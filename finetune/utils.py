import contextlib
import dataclasses
import datetime
import logging
import os
from typing import List, Protocol

import torch

logger = logging.getLogger("utils")


@dataclasses.dataclass
class TrainState:
    step: int = 0


def visible_devices() -> List[int]:
    return [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Closable(Protocol):
    def close(self):
        pass


@contextlib.contextmanager
def logged_closing(thing: Closable, name: str):
    """
    Logging the closing to be sure something is not hanging at exit time
    """
    try:
        setattr(thing, "wrapped_by_closing", True)
        yield
    finally:
        logger.info(f"Closing: {name}")
        try:
            thing.close()
        except Exception:
            logger.error(f"Error while closing {name}!")
            raise
        logger.info(f"Closed: {name}")


def now_as_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
