import dataclasses
import json
import logging
from pathlib import Path

import torch
import torch.distributed.fsdp.fully_sharded_data_parallel as torch_fsdp
from torch.distributed import barrier

from finetune.distributed import get_rank
from finetune.utils import TrainState
from finetune.wrapped_model import PARALLEL_MODEL

logger = logging.getLogger("checkpointing")


@torch.no_grad()
def save_checkpoint(
    model: PARALLEL_MODEL,
    state: TrainState,
    run_dir: Path,
):
    _safe_save(
        model,
        run_dir=run_dir,
        step=state.step,
        rank=get_rank(),
    )


def _safe_save(
    model,
    run_dir: Path,
    step: int,
    rank: int,
    dtype: torch.dtype = torch.float16,
):
    """
    We save in `float16` and not `bfloat16` to have a checkpoint that
    is compatible with older hardware.

    For (Q)LoRA, `state_dict()` returns CPU tensors to avoid GPU OOM.
    Make sure you have enough CPU RAM: for a 70B model you need more
    than 280GB (140GB for bfloat16 and 140GB for casting to half),
    and the checkpointing process can take a few minutes.
    """

    dst = ckpt_dir(run_dir, step)
    tmp_dst = _tmp(dst)
    logger.info(f"Dumping checkpoint in {dst} using tmp name: {tmp_dst.name}")

    assert not dst.exists(), f"dst exists {dst}"
    tmp_dst.mkdir(parents=True, exist_ok=True)
    model_path = _model_path(tmp_dst)

    # distinguish betwen FSDP and DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        this_model = model.module
    elif isinstance(model, torch_fsdp.FullyShardedDataParallel):
        this_model = model
    else:
        raise TypeError(f"Unknown model type: {type(model)}")

    if rank == 0:
        # make sure you have enough CPU RAM for this for LoRA
        states = {k: v.to(dtype=dtype) for k, v in this_model.state_dict().items()}
        torch.save(states, model_path)

        # parameters files
        params_path = _params_path(tmp_dst)
        with open(params_path, "w") as f:
            model_args = dataclasses.asdict(this_model.args)
            model_args.pop("max_batch_size")
            f.write(json.dumps(model_args, indent=4))

    barrier()
    logger.info("Done!")

    if rank == 0:
        assert not dst.exists(), f"should not happen! {dst}"
        tmp_dst.rename(dst)

        logger.info(f"Done dumping checkpoint in {dst}")


def ckpt_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"


def _tmp(ckpt_dir: Path):
    return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")


def _model_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "consolidated.00.pth"


def _params_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "params.json"
