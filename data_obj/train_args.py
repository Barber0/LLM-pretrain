from dataclasses import dataclass
import os

@dataclass
class TrainArgs:
    local_rank: int = int(os.getenv('LOCAL_RANK', default='0'))
    world_size: int = int(os.getenv('WORLD_SIZE', default='1'))

    start_batch: int = 0
    start_epoch: int = 0
    
    batch_size: int = 4
    grad_accum_period: int = 30

    epochs: int = 1
    log_period: int = 50
    save_period: int = 300
    
    validate_period: int = 200
    validate_batch_num: int = 15
    
    replicate_period: int = 3000

    ckpt_home: str = None
    deepspeed_ckpt_tag: str = None

    torch_ckpt_tag: str = None
    torch_ckpt_opt_prefix: str = 'optim'
    
    deepspeed_module_key: str = 'module'