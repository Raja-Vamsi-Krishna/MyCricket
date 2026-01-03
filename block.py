import torch.nn as nn
from attention import CasualMultiHeadAttention
from ffn import FeedForward
from config import GPTConfig

class TransformerBlock(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = CasualMultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x):
       
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
        