import dataclasses
from typing import Iterator, List, Optional

import numpy as np

from finetune.data.args import DataArgs
from finetune.data.dataset import build_dataset
from mistral.tokenizer import Tokenizer


@dataclasses.dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    sizes: List[int]
    y_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.x.ndim == 1
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert isinstance(self.sizes, list)
        # no padding for the moment
        assert sum(self.sizes) == self.x.size == self.y.size

        if self.y_mask is not None:
            assert self.y_mask.size == self.y.size, (self.y_mask.shape, self.y.shape)
            assert self.y_mask.dtype == bool
            assert sum(self.sizes) == self.y_mask.size
            assert not self.y_mask.all()
            assert self.y_mask.any()


def build_data_loader(
    tokenizer: Tokenizer,
    args: DataArgs,
    seq_len: int,
    rank: int,
    world_size: int,
) -> Iterator[Batch]:
    dataset = build_dataset(
        pretrain_data=args.data,
        instruct_data=args.instruct_data,
        instruct_args=args.instruct,
        tokenizer=tokenizer,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
    )

    for sample in dataset:
        np_mask = np.array(sample.mask, dtype=bool)

        assert all(s >= 0 for s in sample.sizes)

        yield Batch(
            x=np.array(sample.x, dtype=np.int64),
            y=np.array(sample.y, dtype=np.int64),
            y_mask=None if np_mask.all() else np_mask,
            sizes=sample.sizes,
        )
