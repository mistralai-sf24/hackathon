import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import wandb
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("metrics_logger")


class MetricsLogger:
    def __init__(
        self,
        dst_dir: Path,
        tag: str,
        is_master: bool,
        wandb_project: Optional[str],
        wandb_offline: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.dst_dir = dst_dir
        self.tag = tag
        self.is_master = is_master
        self.jsonl_path = dst_dir / f"metrics.{tag}.jsonl"
        self.tb_dir = dst_dir / "tb"
        self.summary_writer: Optional[SummaryWriter] = None

        if not self.is_master:
            return

        filename_suffix = f".{tag}"
        self.tb_dir.mkdir(exist_ok=True)
        self.summary_writer = SummaryWriter(
            log_dir=str(self.tb_dir),
            max_queue=1000,
            filename_suffix=filename_suffix,
        )
        self.is_wandb = wandb_project is not None

        if wandb_project:
            if wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            if wandb.run is None:
                logger.info("initializing wandb")
                wandb.init(
                    config=config,
                    dir=dst_dir,
                    project=wandb_project,
                    job_type="training",
                    name=f"{dst_dir.name}",
                    resume=False,
                )

    def log(self, metrics: Dict[str, Union[float, int]], step: int):
        if not self.is_master:
            return

        metrics_to_ignore = {"step"}
        assert self.summary_writer is not None
        for key, value in metrics.items():
            if key in metrics_to_ignore:
                continue
            assert isinstance(value, (int, float)), (key, value)
            self.summary_writer.add_scalar(
                tag=f"{self.tag}.{key}", scalar_value=value, global_step=step
            )
        if self.is_wandb:
            # grouping in wandb is done with /
            wandb.log(
                {
                    f"{self.tag}/{key}": value
                    for key, value in metrics.items()
                    if key not in metrics_to_ignore
                },
                step=step,
            )

        metrics_: Dict[str, Any] = dict(metrics)  # shallow copy
        if "step" in metrics_:
            assert step == metrics_["step"]
        else:
            metrics_["step"] = step
        metrics_["at"] = datetime.datetime.utcnow().isoformat()
        with self.jsonl_path.open("a") as fp:
            fp.write(f"{json.dumps(metrics_)}\n")

    def close(self):
        if not self.is_master:
            return

        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None

        if self.is_wandb:
            # to be sure we are not hanging while finishing
            wandb.finish()

    def __del__(self):
        if self.summary_writer is not None:
            raise RuntimeError(
                "MetricsLogger not closed properly! You should "
                "make sure the close() method is called!"
            )
