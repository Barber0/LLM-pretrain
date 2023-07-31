from dataclasses import dataclass
from enum import Enum


class PositionEmbeddingType(Enum):
    COMPLEX_ROPE = 'COMPLEX_ROPE'
    SIMULATED_ROPE = 'SIMULATED_ROPE'
    LUCIDRAINS_ROPE = 'LUCIDRAINS_ROPE'


@dataclass
class ModelArgs:
    max_len: int = 1024
    ext_factor: int = 1

    n_layers: int = 12
    n_heads: int = 32
    hidden_states: int = 768
    dropout: float = 0.1
    position_encoding: PositionEmbeddingType = PositionEmbeddingType.COMPLEX_ROPE
    rope_interpolate_factor: float = 1.
    rope_theta: float = 1e4