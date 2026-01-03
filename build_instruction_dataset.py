import numpy as np
from pathlib import Path
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
EOS = tokenizer.eot_token

INPUT_FILE = "instruction_data.txt"
OUTPUT_DIR = Path("InstructionData")
OUTPUT_DIR.mkdir(exist_ok=True)

text = Path(INPUT_FILE).read_text(encoding="utf-8")

samples = text.split("\n\n")
tokens = []

for sample in samples:
    sample = sample.strip()
    if not sample:
        continue
    ids = tokenizer.encode(sample)
    ids.append(EOS)
    tokens.extend(ids)

tokens = np.array(tokens, dtype=np.uint16)
tokens.tofile(OUTPUT_DIR / "train.bin")

print("✅ Instruction dataset built")
print("Total tokens:", len(tokens))
