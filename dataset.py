import numpy as np
import json
import torch
from config import GPTConfig

with open("DataSet/Data/meta.json") as f:
    meta = json.load(f)

dtype = np.uint16 if "uint16" in meta["dtype"] else np.uint32

train_data = np.memmap("DataSet/Data/train.bin", dtype=dtype, mode="r")
val_data   = np.memmap("DataSet/Data/value.bin", dtype=dtype, mode="r")

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - GPTConfig.block_size - 1, (GPTConfig.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+GPTConfig.block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+GPTConfig.block_size+1]) for i in ix])
    return x.to(device), y.to(device)
