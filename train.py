import dataclasses
import logging
import pprint
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import fire
import torch.cuda
import torch.distributed as dist
import torch.distributed.fsdp.fully_sharded_data_parallel as torch_fsdp
from torch.distributed import barrier
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs
from finetune.checkpointing import save_checkpoint
from finetune.data.build import build_data_loader
from finetune.distributed import (
    avg_aggregate,
    get_rank,
    get_world_size,
    our_initialize_model_parallel,
    set_device,
)
from finetune.loss import compute_loss_with_mask
from finetune.monitoring.metrics_logger import MetricsLogger
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import build_model, load_initial_model
from mistral.tokenizer import Tokenizer

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger("train")
GB = 1024**3


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    print(f"args: {args}")
    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # Initial setup and checks
    set_random_seed(args.seed)
    set_device()

    # Init NCCL
    logger.info("Going to init comms...")
    dist.init_process_group(backend="nccl")

    our_initialize_model_parallel("nccl", args.n_replica)

    # Init run dir
    logger.info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if run_dir.exists():
        raise RuntimeError(f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}.")

    barrier()

    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)
    logger.info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_project=args.wandb_project,
        wandb_offline=args.wandb_offline,
        config=dataclasses.asdict(args),
    )

    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    # tokenizer / data loader
    tokenizer = Tokenizer(
        model_path=str(Path(args.initial_model_path) / "tokenizer.model")
    )

    data_loader = build_data_loader(
        tokenizer=tokenizer,
        args=args.data,
        seq_len=args.seq_len,
        rank=get_rank(),
        world_size=get_world_size(),
    )

    # model / optimizer
    model = build_model(folder=Path(args.initial_model_path), train_args=args)
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    # state (created just before ckpt reloading)
    state = TrainState()

    # load weights
    load_initial_model(model, args.initial_model_path)
    # train
    model.train()

    torch.cuda.empty_cache()

    while state.step < args.max_steps:
        state.step += 1
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()
        loss = torch.tensor([0.0], device="cuda")

        for i in range(args.num_microbatches):
            is_last_micro_batch = i == args.num_microbatches - 1

            # batch
            batch = next(data_loader)

            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )

            # forward / backward
            output = model(
                input_ids=x,
                seqlens=batch.sizes,
                cache=None,
            )
            mb_loss = compute_loss_with_mask(output, y, y_mask)

            mb_loss.backward()
            loss += mb_loss.detach()

            if not is_last_micro_batch:
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        if isinstance(model, torch_fsdp.FullyShardedDataParallel):
            model.clip_grad_norm_(max_norm=args.max_norm)
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

        # optimizer step
        optimizer.step()
        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if state.step % args.log_freq == 0:
            train_logs = _train_logs(
                state,
                avg_loss,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            logger.info(_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if not args.no_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            save_checkpoint(
                model,
                state,
                run_dir,
            )

    logger.info("done!")


def _train_logs(
    state: TrainState,
    loss: float,
    lr: float,
    peak_allocated_mem: float,
    allocated_mem: float,
    train_args: TrainArgs,
) -> Dict[str, Union[float, int]]:
    metrics = {
        "lr": lr,
        "step": state.step,
        "loss": loss,
        "percent_done": 100 * state.step / train_args.max_steps,
        "peak_allocated_mem": peak_allocated_mem / GB,
        "allocated_mem": allocated_mem / GB,
    }

    return metrics


def _log_msg(state: TrainState, logs: Dict[str, Union[float, int]], loss: float) -> str:
    metrics: Dict[str, Union[float, int, datetime]] = dict(logs)  # shallow copy

    metrics["step"] = state.step
    metrics["loss"] = loss

    parts = []
    for key, fmt, new_name in [
        ("step", "06", None),
        ("percent_done", "03.1f", "done (%)"),
        ("loss", ".3f", None),
        ("lr", ".1e", None),
        ("peak_allocated_mem", ".1f", "peak_alloc_mem (GB)"),
        ("allocated_mem", ".1f", "alloc_mem (GB)"),
    ]:
        name = key if new_name is None else new_name
        try:
            parts.append(f"{name}: {metrics[key]:>{fmt}}")
        except KeyError:
            logger.error(f"{key} not found in {sorted(metrics.keys())}")
            raise

    return " - ".join(parts)


if __name__ == "__main__":
    """See README.md for usage."""
    set_logger(logging.INFO)

    fire.Fire(train)