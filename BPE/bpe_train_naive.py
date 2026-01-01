from pathlib import Path #--window safe file handling
from collections import defaultdict #--counting frequencies safely
from tqdm import tqdm
import json
#path to corpus file
CORPUS_FILE=Path("final_corpus.txt")
END_WORD="<\\w>"
TARGET_VOCAB_SIZE = 32000

def read_corpus_lines(path):
    with open(path,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield line

#creating word vocabulary

def words_to_vocabulary(path):
    vocab=defaultdict(int)

    for line in read_corpus_lines(path):
        words=line.split()
        for word in words:
            vocab[word]+=1
    return vocab        

def vocab_to_symbol_vocab(vocab):
    symbol_vocab=defaultdict(int)

    for word,freq in vocab.items():
        symbols=tuple(list(word)+[END_WORD])
        symbol_vocab[symbols]+=freq
    return symbol_vocab    

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            #NAIVE
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx            
def get_pair_stats(symbol_vocab):
    pair_freq=defaultdict(int)

    for symbol,freq in symbol_vocab.items():
        for i in range(len(symbol)-1):
            pair=(symbol[i],symbol[i+1])
            pair_freq[pair]+=freq
    return pair_freq;      
  
def get_most_frequent_pair(pair_freq):
    return max(pair_freq, key=pair_freq.get)
 
def merge_pair_in_symbols(symbols,pair):
    merged=[]
    i=0
    
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1

    return tuple(merged)


def merge_symbol_vocab(symbol_vocab,pair):
    new_vocab=defaultdict(int)

    for symbols,freq in symbol_vocab.items():
        new_symbols=merge_pair_in_symbols(symbols,pair)
        new_vocab[new_symbols]=+freq
    return new_vocab

def get_initial_symbols(symbol_vocab):
    symbols = set()
    for word_symbols in symbol_vocab:
        for s in word_symbols:
            symbols.add(s)
    return symbols

def train_bpe(symbol_vocab, target_vocab_size):
    merges = []  # ordered list of merge rules
    symbols = get_initial_symbols(symbol_vocab)
    max_merges = target_vocab_size - len(symbols)
    with tqdm(total=max_merges, desc="BPE merges") as pbar:
        while len(symbols) < target_vocab_size:
            pair_stats = get_pair_stats(symbol_vocab)

            if not pair_stats:
             break  # no more merges possible

            best_pair = get_most_frequent_pair(pair_stats)
            merges.append(best_pair)

            # Apply the merge
            symbol_vocab = merge_symbol_vocab(symbol_vocab, best_pair)

            # Add new symbol
            new_symbol = best_pair[0] + best_pair[1]
            symbols.add(new_symbol)

            # Optional progress log
            if len(merges) % 500 == 0:
                print(f"Merges: {len(merges)} | Symbols: {len(symbols)}")
            
            pbar.update(1)
        return merges, symbol_vocab

def save_merges(merges, path):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n") 
def save_vocab(symbol_vocab, path):
    symbols = set()
    for word_symbols in symbol_vocab:
        for s in word_symbols:
            symbols.add(s)

    # deterministic ordering
    token_to_id = {token: idx for idx, token in enumerate(sorted(symbols))}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=2)

if __name__=="__main__":
    vocab=words_to_vocabulary(CORPUS_FILE)
    print("Unique words:", len(vocab))
    symbol_vocab=vocab_to_symbol_vocab(vocab)
    # pair_stats=get_pair_stats(symbol_vocab)
    # most_freq_pair =get_most_frequent_pair(pair_stats)
    # new_vocab=merge_symbol_vocab(symbol_vocab,most_freq_pair)


    # # print a word vocab 
    # for i, (word, freq) in enumerate(vocab.items()):
    #     print(word, freq)
    #     if i == 5:
    #         break

    #  # print a word symbol vocab 
    # for i, (word, freq) in enumerate(symbol_vocab.items()):
    #     print(word, freq)
    #     if i == 5:
    #         break    
    # print("================================= ")

    # # print top 10 most frequent pairs
    # sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1], reverse=True)
    # for pair, freq in sorted_pairs[:10]:
    #     print(pair, freq)
    # print("===================================")
    # print(most_freq_pair)
    # print("===================================")

    # # Print before / after for a sample
    # for (old, freq), (new, _) in zip(symbol_vocab.items(), new_vocab.items()):
    #     print("BEFORE:", old,freq)
    #     print("AFTER :", new,freq)
    #     break

    merges, final_vocab = train_bpe(symbol_vocab, TARGET_VOCAB_SIZE)

    # print("\nFirst 10 merges:")
    # for m in merges[:10]:
    #     print(m)

    # print("\nTotal merges:", len(merges))
    save_merges(merges, "bpe_merges.txt")
    save_vocab(final_vocab, "bpe_vocab.json")

    print("✅ BPE tokenizer training complete")
    print("Saved: bpe_merges.txt")
    print("Saved: bpe_vocab.json")