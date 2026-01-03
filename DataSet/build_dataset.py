import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from BPE.bpe_encoder import BPETokenizer
import tiktoken

BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_PATH = BASE_DIR / "BPE" / "bpe_vocab.json"
MERGES_PATH = BASE_DIR / "BPE" / "bpe_merges.txt"
CORPUS_PATH = BASE_DIR / "BPE" / "final_corpus.txt"
OUTPUT_DIR = BASE_DIR / "DataSet" / "Data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   
TRAIN_RATIO=0.9
tokenizer=tiktoken.get_encoding("gpt2")

DTYPE = np.uint16 if tokenizer.n_vocab < 65536 else np.uint32

tokenizer = tiktoken.get_encoding("gpt2")

# tokenizer=BPETokenizer.load(VOCAB_PATH,MERGES_PATH)
vocab_size=tokenizer.n_vocab
print(vocab_size)
all_tokens=[]
text = CORPUS_PATH.read_text(encoding="utf-8")
assert "<|endoftext|>" not in text, "❌ <|endoftext|> found in corpus!"

with open(CORPUS_PATH,"r",encoding="utf-8") as f:
    for line in tqdm(f,desc="Tokenizing corpus"):
        line=line.strip()
        if not line:
            continue
        ids=tokenizer.encode(line)
        ids.append(tokenizer.eot_token)

        all_tokens.extend(ids)

print(f"Total tokens: {len(all_tokens)}")

split_idx=int(len(all_tokens)*TRAIN_RATIO)
train_idx=np.array(all_tokens[:split_idx],dtype=DTYPE)
val_idx=np.array(all_tokens[split_idx:],dtype=DTYPE)

train_idx.tofile(OUTPUT_DIR / "train.bin")
val_idx.tofile(OUTPUT_DIR / "val.bin")

meta = {
    "vocab_size": tokenizer.n_vocab,
    "dtype": str(DTYPE),
    "eos_token_id": tokenizer.eot_token,
    "tokenizer": "tiktoken_gpt2"
}

with open(OUTPUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Dataset built successfully")