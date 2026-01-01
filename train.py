import torch
import torch.nn as nn
import json
from config import GPTConfig
from model import MyLLM
from dataset import get_batch   
import os

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

start_step = 0
resume_path = "checkpoints/latest.pt"




with open("DataSet/Data/meta.json") as f:
    meta = json.load(f)

vocab_size = meta["vocab_size"]
config = GPTConfig()

model = MyLLM(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=GPTConfig.learning_rate,
    weight_decay=GPTConfig.weight_decay
)

criterion = nn.CrossEntropyLoss()

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    step = ckpt["step"]
    print(f"✅ Resumed from step {step}")

    return step

if os.path.exists(resume_path):
    start_step = load_checkpoint(model, optimizer, resume_path)




def save_checkpoint(model, optimizer, step, config):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": config,
    }

    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step:06d}.pt")
    torch.save(ckpt, path)

    # also update latest pointer
    torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "latest.pt"))

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = torch.zeros(GPTConfig.eval_iters)

        for k in range(GPTConfig.eval_iters):
            X, Y = get_batch(split)
            logits,kv_cache = model(X)

            B, T, V = logits.shape
            loss = criterion(
                logits.view(B*T, V),
                Y.view(B*T)
            )
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out

for step in range(GPTConfig.max_iters):

    if step % GPTConfig.eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Step {step} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    X, Y = get_batch("train")
    logits,kv_cache= model(X)

    B, T, V = logits.shape
    loss = criterion(
        logits.view(B*T, V),
        Y.view(B*T)
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping (important for stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GPTConfig.grad_clip)

    optimizer.step()
    if step % GPTConfig.eval_interval == 0 and step > 0:
        save_checkpoint(model, optimizer, step, config)
    # if step % 1000 == 0:
    #     torch.save(model.state_dict(), "checkpoint.pt")