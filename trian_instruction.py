import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tiktoken
from model import MyLLM
from config import GPTConfig

# -----------------------------
# CONFIG
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CKPT = "checkpoints/ckpt_006000.pt"   # your base LM
OUT_CKPT  = "checkpoints/instruction_tuned.pt"

DATA_PATH = "InstructionData/train.bin"

batch_size = 8
steps = 4000
learning_rate = 2e-5
weight_decay = 0.01

# -----------------------------
# TOKENIZER & RESPONSE MARKER
# -----------------------------
tokenizer = tiktoken.get_encoding("gpt2")
RESPONSE_PREFIX = tokenizer.encode("### Response:\n")

# -----------------------------
# LOAD DATA (memmap)
# -----------------------------
data = np.memmap(
    DATA_PATH,
    dtype=np.uint16,
    mode="r"
)

block_size = GPTConfig.block_size

# -----------------------------
# BATCH CREATION WITH LOSS MASK
# -----------------------------
def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    X, Y, M = [], [], []

    for i in ix:
        seq = torch.from_numpy(
            data[i:i+block_size].copy()
        ).long()

        tgt = torch.from_numpy(
            data[i+1:i+block_size+1].copy()
        ).long()

        # loss mask: 0 = ignore, 1 = train
        mask = torch.zeros_like(tgt)

        # find "### Response:" and enable loss AFTER it
        for j in range(len(seq) - len(RESPONSE_PREFIX)):
            if seq[j:j+len(RESPONSE_PREFIX)].tolist() == RESPONSE_PREFIX:
                mask[j+len(RESPONSE_PREFIX):] = 1
                break

        X.append(seq)
        Y.append(tgt)
        M.append(mask)

    return (
        torch.stack(X).to(device),
        torch.stack(Y).to(device),
        torch.stack(M).to(device),
    )

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MyLLM(GPTConfig()).to(device)

state = torch.load(
    BASE_CKPT,
    map_location=device,
    weights_only=True
)
model.load_state_dict(state)
model.train()

# -----------------------------
# OPTIMIZER
# -----------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# -----------------------------
# TRAIN LOOP (MASKED LOSS)
# -----------------------------
for step in range(steps):
    X, Y, mask = get_batch()

    logits = model(X)

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = Y.view(-1)
    mask_flat = mask.view(-1)

    loss = (
        F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none"
        ) * mask_flat
    ).sum() / mask_flat.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"[SFT] step {step} | loss {loss.item():.4f}")

# -----------------------------
# SAVE
# -----------------------------
torch.save(model.state_dict(), OUT_CKPT)
print("✅ Instruction tuning complete")
print(f"Saved to: {OUT_CKPT}")
