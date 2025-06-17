# infer_gpt.py — inference for models trained with train_gpt.py
# Handles padded vocab (50304) and FlexAttention’s 128‑token block requirement.

import argparse
from pathlib import Path

import torch
import tiktoken

from train_gpt import GPT  # full model definition

EOS_ID = 50256           # <|endoftext|>
PADDED_VOCAB = 50304      # 50257 rounded up to nearest 128
BLOCK_SIZE = 128          # FlexAttention block length
MAX_SEQ_LEN = 49_152      # 48 k used in training

# -----------------------------------------------------------------------------
# Forward monkey‑patch: when target_seq=None, return logits
# -----------------------------------------------------------------------------

def patch_forward_for_inference(model: GPT):
    def forward_patched(self, input_seq, target_seq=None, sliding_window_num_blocks=None):
        # identical to train_gpt but returns logits if target_seq is None
        assert input_seq.ndim == 1
        ve = [emb(input_seq) for emb in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        bm = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm,
              short_bm, short_bm, short_bm, long_bm]
        x = x0 = torch.nn.functional.rms_norm(self.embed(input_seq)[None], (self.embed.embedding_dim,))
        skip = []
        n = len(self.skip_weights)
        for i, blk in enumerate(self.blocks):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip.pop()
            x = blk(x, ve[i], x0, bm[i])
            if i < n:
                skip.append(x)
        x = torch.nn.functional.rms_norm(x, (self.embed.embedding_dim,))
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
        if target_seq is not None:
            return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="mean")
        return logits

    model.forward = forward_patched.__get__(model, GPT)

# -----------------------------------------------------------------------------
# Greedy generation (keeps sequence length multiple of 128 each step)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate(model: GPT, prompt_ids: torch.Tensor, *, max_new_tokens: int = 100, window_blocks: int = 64):
    """Greedy decoding that keeps the *context* (not the growing sequence) length
    a multiple of 128 for FlexAttention without polluting the generated output
    with tons of <|endoftext|> tokens.
    """
    device = prompt_ids.device
    ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Use only the recent MAX_SEQ_LEN tokens as context
        ctx = ids[-MAX_SEQ_LEN:]

        # Pad **a COPY** of the context on the *left* so its length is a multiple
        # of 128, but **do not** add that padding back into the growing `ids`.
        pad_ctx = (-len(ctx)) % BLOCK_SIZE
        if pad_ctx:
            ctx = torch.cat([
                torch.full((pad_ctx,), EOS_ID, dtype=ids.dtype, device=device),
                ctx,
            ])

        num_blocks = torch.tensor(window_blocks, dtype=torch.int32, device=device)
        logits = model(ctx, target_seq=None, sliding_window_num_blocks=num_blocks)
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # [T, V]

        next_tok = torch.argmax(logits[-1], dim=-1).to(torch.int32)
        ids = torch.cat([ids, next_tok.view(1)])
        if next_tok.item() == EOS_ID:
            break

    return ids

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="logs/2025-06-17_15-25-33/state_step001770.pt",
                        help="Path to .pt checkpoint file")
    parser.add_argument("--prompt", type=str,
                        default="<|im_start|>user\nYou are given two strings s and t, each of length n and consisting of lowercase Latin alphabets.")
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor(enc.encode(args.prompt), dtype=torch.int32, device="cuda")

    model = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=MAX_SEQ_LEN).cuda()

    ckpt = torch.load(Path(args.ckpt), map_location="cuda")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    patch_forward_for_inference(model)

    out_ids = generate(model, prompt_ids, max_new_tokens=args.max_tokens)

    decoded = enc.decode([t for t in out_ids.tolist() if t < 50257])
    print("\n=== Completion ===\n")
    print(decoded)


if __name__ == "__main__":
    main()
