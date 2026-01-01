from dataclasses import dataclass

@dataclass
class GPTConfig:
    batch_size:int=64
    vocab_size: int=50001
    block_size: int=256 #Context Length
    n_layers: int = 4
    n_heads: int = 4    
    n_embd: int = 256
    dropout: float = 0.05    
    
    max_iters = 60_000
    eval_interval = 1000
    eval_iters = 200
    learning_rate = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0