import logging
import os
from functools import lru_cache
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup

from finetune.utils import visible_devices

logger = logging.getLogger("distributed")

_SHARD_GROUP: Optional[torch.distributed.ProcessGroup] = None
_REPLICA_GROUP: Optional[torch.distributed.ProcessGroup] = None


@lru_cache()
def get_rank() -> int:
    return dist.get_rank()


@lru_cache()
def get_world_size() -> int:
    return dist.get_world_size()


def our_initialize_model_parallel(
    _backend: Optional[str] = None,
    n_replica: int = 1,
) -> None:
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    assert (
        world_size % n_replica == 0
    ), f"{world_size=} is not divisible by {n_replica=}"
    rank = torch.distributed.get_rank()

    shard_size = int(world_size / n_replica)

    logger.info("> initializing shard with size {}".format(shard_size))
    logger.info("> initializing replica with size {}".format(n_replica))

    groups = torch.LongTensor(range(world_size)).reshape(n_replica, shard_size)

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    global _REPLICA_GROUP
    assert _REPLICA_GROUP is None, "replica group is already initialized"
    for j in range(shard_size):
        ranks = groups[:, j].tolist()
        group = torch.distributed.new_group(ranks, backend=_backend)
        if j == found[1]:
            _REPLICA_GROUP = group

    assert get_replica_parallel_group() is not None

    global _SHARD_GROUP
    assert _SHARD_GROUP is None, "shard group is already initialized"
    for i in range(n_replica):
        group = torch.distributed.new_group(groups[i, :].tolist(), backend=_backend)
        if i == found[0]:
            _SHARD_GROUP = group

    assert get_shard_group() is not None


@lru_cache()
def get_replica_parallel_group() -> ProcessGroup:
    """Get the replica group the caller rank belongs to."""
    assert _REPLICA_GROUP is not None, "replica group is not initialized"
    return _REPLICA_GROUP


@lru_cache()
def get_shard_group() -> ProcessGroup:
    """Get the shard group the caller rank belongs to."""
    assert _SHARD_GROUP is not None, "shard group is not initialized"
    return _SHARD_GROUP


@lru_cache()
def get_shard_rank() -> int:
    return torch.distributed.get_rank(group=get_shard_group())


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")

    assert torch.cuda.is_available()

    assert len(visible_devices()) == torch.cuda.device_count()

    if torch.cuda.device_count() == 1:
        # gpus-per-task set to 1
        torch.cuda.set_device(0)
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Set cuda device to {local_rank}")

    assert 0 <= local_rank < torch.cuda.device_count(), (
        local_rank,
        torch.cuda.device_count(),
    )
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.AVG)
    return buffer[0].item()
