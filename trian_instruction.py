import torch
import torch.nn as nn
from model import MyLLM
from config import GPTConfig
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load instruction dataset
# -----------------------------
data = np.memmap(
    "InstructionData/train.bin",
    dtype=np.uint16,
    mode="r"
)

block_size = GPTConfig.block_size
batch_size = 8  # small on purpose

def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+block_size]).long()
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+block_size+1]).long()
        for i in ix
    ])
    return x.to(device), y.to(device)

# -----------------------------
# Load model (FROM BASE LM)
# -----------------------------
model = MyLLM(GPTConfig()).to(device)
model.load_state_dict(torch.load("checkpoints/ckpt_006000.pt", map_location=device))
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-6,           # VERY IMPORTANT (low LR)
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()

# -----------------------------
# Instruction fine-tuning
# -----------------------------
steps = 2000   # small on purpose

for step in range(steps):
    X, Y = get_batch()
    logits = model(X)
    loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | loss {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/instruction_tuned.pt")
print("✅ Instruction tuning complete")
