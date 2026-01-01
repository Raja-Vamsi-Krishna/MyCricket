import numpy as np
import json
import torch 
from config import GPTConfig

with open("DataSet/Data/meta.json") as f:
    meta = json.load(f)

dtype = np.uint16 if "uint16" in meta["dtype"] else np.uint32

train_data = np.memmap("DataSet/Data/train.bin", dtype=dtype, mode="r")
val_data   = np.memmap("DataSet/Data/value.bin", dtype=dtype, mode="r")

print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

batch_size = GPTConfig.batch_size     
block_size = GPTConfig.block_size

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split: str):
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        low=0,
        high=len(data) - block_size - 1,
        size=(batch_size,)
    )

    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)