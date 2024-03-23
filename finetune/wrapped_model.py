import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.distributed.fsdp.fully_sharded_data_parallel as torch_fsdp
import torch.distributed.fsdp.wrap as torch_wrap
import torch.nn.parallel as torch_ddp
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.api import ShardingStrategy

from finetune.args import TrainArgs
from finetune.distributed import get_rank, get_replica_parallel_group, get_shard_group
from mistral.model import ModelArgs, Transformer

logger = logging.getLogger(__name__)
PARALLEL_MODEL = Union[
    torch_fsdp.FullyShardedDataParallel, torch_ddp.DistributedDataParallel
]


def build_model(
    folder: Path,
    train_args: TrainArgs,
) -> PARALLEL_MODEL:
    with open(folder / "params.json", "r") as f:
        args = json.loads(f.read())
        model_args = ModelArgs(
            max_batch_size=train_args.seq_len,
            lora=train_args.lora,
            **args,
        )

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore

    # LoRA only supports DDP and not FSDP
    if model_args.lora.enable:
        model = build_model_ddp(model_args, train_args)
    else:
        model = build_model_fsdp(model_args, train_args)

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Rank {get_rank():.0f} has {train_params:,.0f} params to finetune")

    return model


def build_model_ddp(
    model_args: ModelArgs, train_args: TrainArgs
) -> torch_ddp.DistributedDataParallel:
    ddp_params: Dict[str, Any] = {
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
    }
    with torch_wrap.enable_wrap(
        wrapper_cls=torch_ddp.DistributedDataParallel, **ddp_params
    ):
        model = Transformer(args=model_args, checkpoint=train_args.checkpoint)

        # only finetune LoRA parameters and freeze before wrapping
        for name, param in model.named_parameters():
            if "lora" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model = torch_wrap.wrap(model)

    assert isinstance(model, torch_ddp.DistributedDataParallel)
    logger.info(f"Wrapped model with DDP: {model}")
    return model


def build_model_fsdp(
    model_args: ModelArgs, train_args: TrainArgs
) -> torch_fsdp.FullyShardedDataParallel:
    assert not model_args.lora.enable, "Use DDP with LoRA"
    hybrid_reshard_to_sharding: Dict[Tuple[bool, bool], ShardingStrategy] = {
        (False, False): ShardingStrategy.FULL_SHARD,
        (False, True): ShardingStrategy.SHARD_GRAD_OP,
        (True, False): ShardingStrategy.HYBRID_SHARD,
        (True, True): ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    use_hybrid = train_args.n_replica > 1
    if use_hybrid:
        pg: Union[
            torch.distributed.ProcessGroup,
            Tuple[torch.distributed.ProcessGroup, torch.distributed.ProcessGroup],
        ] = (get_shard_group(), get_replica_parallel_group())
    else:
        pg = get_shard_group()
    sharding = hybrid_reshard_to_sharding[
        (use_hybrid, train_args.reshard_after_forward)
    ]
    logger.info(f"Using hybrid: {use_hybrid} with strategy: {sharding}")

    fsdp_params: Dict[str, Any] = {
        "process_group": pg,
        "sharding_strategy": sharding,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
    }

    # useful to shard in-place at initialization time
    with torch_wrap.enable_wrap(
        wrapper_cls=torch_fsdp.FullyShardedDataParallel, **fsdp_params
    ):
        model = torch_wrap.wrap(
            Transformer(args=model_args, checkpoint=train_args.checkpoint)
        )

    assert isinstance(model, torch_fsdp.FullyShardedDataParallel)
    logger.info(f"Wrapped model with FSDP: {model}")
    return model


@torch.no_grad()
def load_initial_model(
    model: PARALLEL_MODEL,
    model_path: str,
) -> None:
    path = Path(model_path)
    assert path.is_dir(), path

    this_path = path / f"consolidated.{0:02d}.pth"
    assert this_path.exists(), this_path
    logger.info(f"Going to reload consolidated model from {this_path}")

    # load directly on GPU for FSDP since it will shard on-the-fly
    if isinstance(model, torch_fsdp.FullyShardedDataParallel):
        model_state_dict = torch.load(this_path)
        model.load_state_dict(model_state_dict)

    # load to CPU first to avoid GPU OOM and for correct QLoRA loading
    elif isinstance(model, torch_ddp.DistributedDataParallel):
        model_state_dict = torch.load(this_path, map_location="cpu")
        model.module.load_state_dict(model_state_dict)
    else:
        raise TypeError(model)
