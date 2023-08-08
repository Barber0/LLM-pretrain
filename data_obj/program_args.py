from dataclasses import dataclass


@dataclass
class ProgramArgs:
    deepspeed_cfg: str = None
    train_path: str = None
    data_loader_workers: int = 10

    tokenizer_path: str = None
    tensorboard_path: str = None
    log_path: str = None
