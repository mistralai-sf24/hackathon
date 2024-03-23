import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from simple_parsing.helpers import Serializable

from finetune.data.args import DataArgs


@dataclass
class OptimArgs(Serializable):
    lr: float = 3e-4
    weight_decay: float = 0.1
    pct_start: float = 0.3


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    # generic
    rank: int = 16
    dropout: float = 0.0
    scaling: float = 2.0
    # quantization
    quantized: bool = False
    block_size: int = 64

    def __post_init__(self):
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class TrainArgs(Serializable):
    data: DataArgs
    initial_model_path: str  # Path to the directory containing the initial model.

    run_dir: str  # Path to the directory where everything will be saved. It needs to be empty.
    wandb_project: Optional[str] = None  # Fill this argument to use wandb.
    wandb_offline: bool = False

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    num_microbatches: int = 1  # Number of steps to accumulate gradients before calling doing an optimizer step.
    seq_len: int = 2048  # Number of tokens per batch per device.
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 10  # Number of steps between each logging.

    # ckpt
    ckpt_freq: int = 0  # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    no_ckpt: bool = False  # /!\ If True, no checkpoint will be saved. This is useful for development.

    # Efficiency
    n_replica: int = 1  # Parameter to control Fully Sharded Data Parallelism (FSDP). It sets how many copies of the model will be made for distributed training. The total number of GPUs used (word_size) divided by n_replica gives the number of GPUs over which one replica will be sharded. A common practice is to set n_replica to the number of nodes.
    checkpoint: bool = False  # Determines whether gradient checkpointing should be utilized or not during the training process. Gradient checkpointing can be beneficial in reducing memory usage at the cost of slightly longer training times.
    reshard_after_forward: bool = False  # If True, the sharding strategy will be Zero 2. Otherwise, it will be Zero 3.

    # Will be filled automatically by the code.
    world_size: int = field(init=False)

    # LoRA
    lora: LoraArgs = field(default_factory=LoraArgs)

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb_offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        if self.initial_model_path is not None:
            Path(self.initial_model_path).exists()
