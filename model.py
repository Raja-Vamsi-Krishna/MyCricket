import torch
import torch.nn as nn
from block import TransformerBlock
from config import GPTConfig
class MyLLM(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def forward(self, idx , kv_cache=None,start_pos=0):
        batch_size,no_of_tokens= idx.shape
        assert no_of_tokens <= self.pos_emb.num_embeddings

        pos = torch.arange(0,no_of_tokens, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # new_cache=[]

        # for i, block in enumerate(self.blocks):
        #     past_kv = None if kv_cache is None else kv_cache.get(i)
        #     x, kv = block(x, past_kv)
        #     new_cache.append(kv)

        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits