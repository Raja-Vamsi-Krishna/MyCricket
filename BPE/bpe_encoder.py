import json
from pathlib import Path
from typing import List
from functools import lru_cache
import unicodedata
# ---------------- CONSTANTS ----------------
END_WORD = "</w>"        # BPE word boundary
EOS_TOKEN = "<eos>"      # document boundary (for LLM training)


# ---------------- TOKENIZER CLASS ----------------
class BPETokenizer:
    """
    Byte Pair Encoding tokenizer compatible with GPT-style training.
    """

    def __init__(self, merges: List[tuple], token_to_id: dict):
        self.merges = merges                    # list of (a, b)
        self.token_to_id = token_to_id          # token -> id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        # self.cache = {}
        if EOS_TOKEN not in token_to_id:
            raise ValueError("❌ EOS token '<eos>' missing in vocab")

        self.eos_token_id = token_to_id[EOS_TOKEN]

    # -------- REQUIRED BY DATASET BUILDER --------
    @property
    def vocab_size(self):
        return len(self.token_to_id)

    # ---------------- ENCODE ----------------
    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """
        text -> token ids
        """
        text = unicodedata.normalize("NFKC", text)

        # Remove ALL combining characters (like ̇)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) != "Mn"
        )   
        text = text.strip().lower()
        tokens = self._encode_text_to_tokens(text)
        ids=self._tokens_to_ids(tokens)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    # ---------------- DECODE ----------------
    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
     words = []
     current = ""

     for i in ids:
        if stop_at_eos and i == self.eos_token_id:
            break

        tok = self.id_to_token[i]

        if tok.endswith(END_WORD):
            current += tok.replace(END_WORD, "")
            words.append(current)
            current = ""
        else:
            current += tok

     if current:
        words.append(current)
     return " ".join(words)

    # ---------------- INTERNAL BPE LOGIC ----------------
    def _word_to_symbols(self, word: str) -> List[str]:
        return list(word) + [END_WORD]

    def _apply_merge(self, symbols: List[str], pair: tuple, new_symbol: str) -> List[str]:
        i = 0
        merged = []

        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(new_symbol)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1

        return merged
    @lru_cache(maxsize=500_000)
    def _encode_word(self, word: str) -> List[str]:
        # if word in self.cache:
        #     return self.cache[word]
        

        symbols = self._word_to_symbols(word)

        for a, b in self.merges:
            symbols = self._apply_merge(symbols, (a, b), a + b)
        # self.cache[word] = symbols
        return symbols

    def _encode_text_to_tokens(self, text: str) -> List[str]:
        tokens = []
        for word in text.split():
            tokens.extend(self._encode_word(word))
        return tokens

    # ---------------- TOKEN RESOLUTION ----------------
    def _resolve_token(self, token: str) -> List[str]:
        """
        Ensures token exists in vocab.
        Fallback: split into characters.
        """
        if token in self.token_to_id:
            return [token]
        return list(token)

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        ids = []

        for tok in tokens:
            resolved = self._resolve_token(tok)
            for sub in resolved:
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                elif " " in self.token_to_id:
                        ids.append(self.token_to_id[" "])
                    # raise ValueError(f"❌ Unresolvable token: {sub}")
                else:
                    continue

        return ids

    # ---------------- LOAD ----------------
    @classmethod
    def load(cls, vocab_path: str, merges_path: str):
        """
        Loads tokenizer from files
        """
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)

        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split()
                merges.append((a, b))

        return cls(merges, vocab)


# ---------------- TEST ----------------
if __name__ == "__main__":
    tokenizer = BPETokenizer.load(
        "bpe_vocab.json",
        "bpe_merges.txt"
    )

    sentence = "^"
    ids = tokenizer.encode(sentence)
    decoded = tokenizer.decode(ids)
    tokens = tokenizer._encode_text_to_tokens(sentence)
    print("Original :", sentence)
    print("Tokens:",tokens)
    print("Token IDs:", ids)
    print("Decoded  :", decoded)
    print("Vocab size:", tokenizer.vocab_size)
    print("EOS ID:", tokenizer.eos_token_id)

