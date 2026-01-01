from collections import defaultdict
from pathlib import Path
import json
import heapq
from tqdm import tqdm

# ============================================================
# CONFIG (FREEZE THESE)
# ============================================================

CORPUS_FILE = Path("final_corpus.txt")
END_WORD = "</w>"
EOS_TOKEN = "<eos>"
TARGET_VOCAB_SIZE = 50000

MERGES_FILE = "bpe_merges.txt"
VOCAB_FILE = "bpe_vocab.json"

# ============================================================
# STEP 1 — CORPUS → WORD VOCAB
# ============================================================

def read_corpus_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def build_word_vocab(path):
    vocab = defaultdict(int)
    for line in read_corpus_lines(path):
        for word in line.split():
            vocab[word] += 1
    return vocab

# ============================================================
# STEP 2 — WORD → SYMBOL VOCAB
# ============================================================

def word_to_symbols(word):
    return tuple(list(word) + [END_WORD])

def build_symbol_vocab(word_vocab):
    symbol_vocab = defaultdict(int)
    for word, freq in word_vocab.items():
        symbol_vocab[word_to_symbols(word)] += freq
    return symbol_vocab

# ============================================================
# STEP 3 — WORD TABLES (OPTIMIZATION A)
# ============================================================

def build_word_tables(symbol_vocab):
    words = []
    freqs = []
    for symbols, freq in symbol_vocab.items():
        words.append(symbols)
        freqs.append(freq)
    return words, freqs

# ============================================================
# STEP 4 — PAIR → WORD MAPPING (OPTIMIZATION B)
# ============================================================

def build_pair_to_words(words):
    pair_to_words = defaultdict(set)
    for wid, symbols in enumerate(words):
        for i in range(len(symbols) - 1):
            pair_to_words[(symbols[i], symbols[i + 1])].add(wid)
    return pair_to_words

# ============================================================
# STEP 5 — PAIR FREQUENCIES (OPTIMIZATION C)
# ============================================================

def build_pair_freq(pair_to_words, freqs):
    pair_freq = defaultdict(int)
    for pair, word_ids in pair_to_words.items():
        total = 0
        for wid in word_ids:
            total += freqs[wid]
        pair_freq[pair] = total
    return pair_freq

# ============================================================
# STEP 6 — HEAP UTILITIES (OPTIMIZATION E)
# ============================================================

def build_pair_heap(pair_freq):
    heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)
    return heap

def get_best_pair_from_heap(heap, pair_freq):
    while heap:
        neg_freq, pair = heapq.heappop(heap)
        freq = -neg_freq
        if pair in pair_freq and pair_freq[pair] == freq:
            return pair
    return None

# ============================================================
# STEP 7 — APPLY ONE OPTIMIZED MERGE (OPTIMIZATION D)
# ============================================================

def merge_pair_in_word(symbols, pair, new_symbol):
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(new_symbol)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)

def apply_merge_once(best_pair, words, freqs, pair_to_words, pair_freq, heap):
    a, b = best_pair
    new_symbol = a + b

    affected_word_ids = list(pair_to_words[best_pair])

    del pair_to_words[best_pair]
    del pair_freq[best_pair]

    for wid in affected_word_ids:
        old_symbols = words[wid]
        new_symbols = merge_pair_in_word(old_symbols, best_pair, new_symbol)
        words[wid] = new_symbols
        freq = freqs[wid]

        # Remove old pairs
        for i in range(len(old_symbols) - 1):
            old_pair = (old_symbols[i], old_symbols[i + 1])
            if old_pair in pair_to_words:
                pair_to_words[old_pair].discard(wid)
                pair_freq[old_pair] -= freq
                if pair_freq[old_pair] <= 0:
                    pair_to_words.pop(old_pair, None)
                    pair_freq.pop(old_pair, None)

        # Add new pairs
        for i in range(len(new_symbols) - 1):
            new_pair = (new_symbols[i], new_symbols[i + 1])
            pair_to_words.setdefault(new_pair, set()).add(wid)
            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
            heapq.heappush(heap, (-pair_freq[new_pair], new_pair))

# ============================================================
# STEP 8 — BPE TRAINING LOOP (FINAL)
# ============================================================

def train_bpe_fast(words, freqs, pair_to_words, pair_freq, target_vocab_size):
    heap = build_pair_heap(pair_freq)

    symbols = set(s for w in words for s in w)
    merges = []

    max_merges = target_vocab_size - len(symbols)

    with tqdm(total=max_merges, desc="BPE merges") as pbar:
        while len(symbols) < target_vocab_size:
            best_pair = get_best_pair_from_heap(heap, pair_freq)
            if best_pair is None:
                break

            merges.append(best_pair)
            new_symbol = best_pair[0] + best_pair[1]
            symbols.add(new_symbol)

            apply_merge_once(
                best_pair,
                words,
                freqs,
                pair_to_words,
                pair_freq,
                heap
            )

            pbar.update(1)

    return merges, symbols

# ============================================================
# STEP 9 — SAVE ARTIFACTS (FREEZE POINT)
# ============================================================

def save_merges(merges, path):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

def save_vocab(symbols, path):
    token_to_id = {tok: i for i, tok in enumerate(sorted(symbols))}

    eos_id = len(token_to_id)
    token_to_id[EOS_TOKEN] = eos_id

    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=2)

# ============================================================
# MAIN (SINGLE ENTRY POINT)
# ============================================================

def main():
    print("📥 Building word vocabulary...")
    word_vocab = build_word_vocab(CORPUS_FILE)

    print("🔤 Building symbol vocabulary...")
    symbol_vocab = build_symbol_vocab(word_vocab)

    print("📊 Building optimized structures...")
    words, freqs = build_word_tables(symbol_vocab)
    pair_to_words = build_pair_to_words(words)
    pair_freq = build_pair_freq(pair_to_words, freqs)

    print("🚀 Training optimized BPE tokenizer...")
    merges, symbols = train_bpe_fast(
        words,
        freqs,
        pair_to_words,
        pair_freq,
        TARGET_VOCAB_SIZE
    )

    print("💾 Saving tokenizer artifacts...")
    save_merges(merges, MERGES_FILE)
    save_vocab(symbols, VOCAB_FILE)

    print("✅ BPE tokenizer training complete")
    print(f"Saved: {MERGES_FILE}")
    print(f"Saved: {VOCAB_FILE}")

if __name__ == "__main__":
    main()
