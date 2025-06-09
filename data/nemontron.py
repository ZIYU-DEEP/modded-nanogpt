"""
Nemontron dataset (for pretraining)
https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/
Using only the SFT subset, using ["code", "math", "science"] splits.
The total number of tokens is around 200B.
We use the first shard (0.1B) as validation and the rest as training.
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import argparse
import numpy as np


# ------------------------------------------

def tokenize(doc):
    """
    Tokenizes a single document and returns a numpy array of uint16 tokens.
    """

    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2 ** 31, "token count too large" # ~2.1B tokens
    
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2 ** 16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
        
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
        

def apply_chat_template(sample):
    """
    Helper function to format the input messages and output text.
    """
    input_messages = sample["input"]
    output_text = sample["output"]

    conversation = ""
    for msg in input_messages:
        role = msg["role"]
        content = msg["content"].strip()
        conversation += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    
    # Add the assistant's final answer
    conversation += f"<|im_start|>assistant\n{output_text.strip()}\n<|im_end|>"
    return {"text": conversation}

# -------------------------------------------

parser = argparse.ArgumentParser(description="Nemontron dataset preprocessing")
parser.add_argument(
    "-n", "--num_bins", type=int,
    default=8,
    help="Number of .bin files to download. Each file holds up to `shard_size` tokens."
)
parser.add_argument(
    "-d", "--dataset_name", type=str, 
    default="nvidia/Llama-Nemotron-Post-Training-Dataset", 
    help="The name of the dataset to load from Hugging Face Hub."
)
parser.add_argument(
    "-sp", "--split", type=str, 
    default="math,code,science", 
    help="The split of the dataset."
)
parser.add_argument(
    "-l", "--local_dir", type=str, 
    default="nemontron", 
    help="The local directory to save the data."
)
parser.add_argument(
    "-c", "--data_cache_dir", type=str, 
    default=".", 
    help="The local directory to save the data."
)
parser.add_argument(
    "-s", "--shard_size", type=int, 
    default=10 ** 8, 
    help="Size of each shard in tokens; usually 1/100 of the dataset size."
)


args = parser.parse_args()
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
splits = [s.strip() for s in args.split.split(",")]
DATA_CACHE_DIR = os.path.join(
    os.path.expanduser(args.data_cache_dir), 
    args.local_dir
)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ------------------------------------------

# Step 1: download and process the dataset
fw = load_dataset(
    args.dataset_name, 
    split=splits,  # by default, we use math, code, and science splits
)
fw = concatenate_datasets(fw)
fw = fw.shuffle(seed=8964) 
fw = fw.map(
    apply_chat_template, 
    num_proc=nprocs,
    desc="Concatenating input and output to text with delimiters."
)


# Step 2: init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token


# Step 3: tokenize all documents and write output shards
# each of shard_size tokens (last shard has remainder)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty(
        (args.shard_size,), 
        dtype=np.uint16
    )
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count += len(tokens)
            
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(
                    total=args.shard_size, 
                    unit="tokens", 
                    desc=f"Shard {shard_index}"
                )
                
            progress_bar.update(len(tokens))
            
        else:
            # write the current shard and start a new one
            if args.num_bins is not None and shard_index >= args.num_bins:
                break

            split = "val" if shard_index == 0 else "train"  # use the first as val
            filename = os.path.join(
                DATA_CACHE_DIR, 
                f"{args.local_dir}_{split}_{shard_index:06d}.bin"
            )
            
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0 and (args.num_bins is None or shard_index < args.num_bins):
        split = "val" if shard_index == 0 else "train"
        
        filename = os.path.join(
            DATA_CACHE_DIR, 
            f"{args.local_dir}_{split}_{shard_index:06d}.bin"
        )
        
        write_datafile(
            filename=filename, 
            toks=all_tokens_np[:token_count]
        )
