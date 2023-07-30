from dataclasses import dataclass

@dataclass
class ModelArgs:
    max_len: int = 512
    ext_factor: int = 2
    
    n_layers: int = 12
    n_heads: int = 32
    hidden_states: int = 768
    dropout: float = 0.1