import torch
import torch.nn as nn
import os
from model import MyLLM
from config import GPTConfig
from dataset import get_batch
import tiktoken
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyLLM(GPTConfig()).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=GPTConfig.learning_rate,
    weight_decay=GPTConfig.weight_decay
)
tokenizer=tiktoken.get_encoding("gpt2")

criterion = nn.CrossEntropyLoss()
os.makedirs("checkpoints", exist_ok=True)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(GPTConfig.eval_iters)
        for k in range(GPTConfig.eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

for step in range(GPTConfig.max_iters):
    if step % GPTConfig.eval_interval == 0:
        losses = estimate_loss()
        print(f"{step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    X, Y = get_batch("train")
    logits = model(X)
    loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
    
    if step % 2000 == 0:
        with torch.no_grad():
            print("\n🔍 SANITY CHECK")
            print("TARGET:")
            print(tokenizer.decode(X[0][:50].tolist()))
            print("PREDICTION:")
            print(tokenizer.decode(torch.argmax(logits[0], dim=-1)[:50].tolist()))
            print("-" * 40)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GPTConfig.grad_clip)
    optimizer.step()
    
    if step % GPTConfig.eval_interval == 0 and step > 0:
        torch.save(model.state_dict(), f"checkpoints/ckpt_{step:06d}.pt")
