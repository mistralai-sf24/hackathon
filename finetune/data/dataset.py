import dataclasses
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import datasets
import numpy as np

from finetune.data.args import InstructArgs
from mistral.tokenizer import Tokenizer

datasets.utils.logging.set_verbosity_info()

logger = logging.getLogger("dataset")


@dataclass()
class Interaction:
    text: str
    is_user: bool

    @property
    def is_bot(self) -> bool:
        return not self.is_user

    def __post_init__(self):
        assert type(self.text) is str, self.text
        assert type(self.is_user) is bool, self.is_user


@dataclass()
class Sample:
    data: Dict[str, Any]  # json.loads

    @property
    def text(self) -> str:
        text = self.data.get("text", self.data.get("content"))
        assert type(text) is str
        return text

    @property
    def interactions(self) -> List[Interaction]:
        interactions = self.data.get("interactions")
        assert isinstance(interactions, list)
        processed_interactions = (
            [
                Interaction(
                    text=interaction["text"],
                    is_user=interaction["is_user"],
                )
                for interaction in interactions
            ]
            if interactions is not None
            else None
        )
        return processed_interactions


Sequence = List[int]
Mask = List[bool]


@dataclass()
class TokenSample:
    tokens: Sequence
    masks: Mask


class SampleType(str, Enum):
    PRETRAIN = "pretrain"
    INSTRUCT = "instruct"


def tokenize(
    sample: Sample,
    tokenizer: Tokenizer,
    as_type: SampleType,
    instruct_arg: InstructArgs,
) -> TokenSample:
    if as_type == SampleType.PRETRAIN:
        return tokenize_pretrain(sample, tokenizer)
    else:
        assert as_type == SampleType.INSTRUCT
        return tokenize_instruct(sample, tokenizer, instruct_arg)


def tokenize_pretrain(sample: Sample, tokenizer: Tokenizer) -> TokenSample:
    assert sample.text is not None
    tokens = tokenizer.encode(sample.text, bos=True, eos=True)
    masks = [True] * len(tokens)
    return TokenSample(tokens, masks)


def tokenize_instruct(
    sample: Sample,
    tokenizer: Tokenizer,
    instruct_arg: InstructArgs,
) -> TokenSample:
    assert sample.interactions is not None
    tokens: List[int] = [tokenizer.bos_id]
    masks: List[bool] = [False]

    for interaction in sample.interactions:
        msg = interaction.text
        if interaction.is_user and instruct_arg.add_flag:
            msg = f"[INST] {msg} [/INST]"

        # this will do user-utterance bot-utterance <eos> for every turn
        curr_tokens = tokenizer.encode(msg, bos=False, eos=interaction.is_bot)
        mask = interaction.is_bot
        curr_masks = [mask] * len(curr_tokens)  # only predict bot answers

        tokens.extend(curr_tokens)
        masks.extend(curr_masks)

    return TokenSample(tokens, masks)


@dataclass
class DataDir:
    data_dir: Path
    sample_type: SampleType


def parse_data_sources(
    pretrain_data: str,
    instruct_data: str,
) -> Tuple[List[DataDir], List[float]]:
    seen: Set[str] = set()
    sources: List[DataDir] = []
    weights: List[float] = []
    for sample_sources, sample_type in [
        (pretrain_data, SampleType.PRETRAIN),
        (instruct_data, SampleType.INSTRUCT),
    ]:
        for source in [src for src in sample_sources.split(",") if src]:
            path_, weight_ = source.strip().split(":")
            weight = float(weight_)
            assert path_ not in seen, path_
            assert weight > 0
            seen.add(path_)

            sources.append(DataDir(data_dir=Path(path_), sample_type=sample_type))
            weights.append(weight)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]
    assert min(n_weights) > 0
    assert abs(sum(n_weights) - 1) < 1e-8
    return sources, n_weights


@dataclasses.dataclass()
class SequenceMaskAndSizes:
    """
    Concatenation of samples to reach a given size
    """

    x: List[int]
    y: List[int]
    mask: Mask
    sizes: List[int]

    def __post_init__(self):
        assert sum(self.sizes) == len(self.x) == len(self.y) == len(self.mask)


def encode(
    data: Dict[str, Any],
    instruct_args: InstructArgs,
    tokenizer: Tokenizer,
    as_type: SampleType,
):
    sample = Sample(data=data)
    return tokenize(
        sample=sample,
        tokenizer=tokenizer,
        as_type=as_type,
        instruct_arg=instruct_args,
    )


def sequence_iterator(
    ds_it: Iterator[TokenSample],
    seq_len: int,
) -> Iterator[SequenceMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    x_buffer: List[int] = []
    y_buffer: List[int] = []
    mask_buffer: Mask = []

    sizes: List[int] = []
    n_missing = seq_len
    for sample in ds_it:
        assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
        assert n_missing == seq_len - len(x_buffer)

        tokens, mask = sample.tokens, sample.masks[1:]
        x, y = tokens[:-1], tokens[1:]
        cur_pos = 0

        while cur_pos < len(x):
            size = len(x[cur_pos : cur_pos + n_missing])

            curr_mask = mask[cur_pos : cur_pos + n_missing]
            if not any(curr_mask):
                cur_pos += size
                # we have a sequence with a mask filled with False
                continue

            x_buffer.extend(x[cur_pos : cur_pos + n_missing])
            y_buffer.extend(y[cur_pos : cur_pos + n_missing])
            mask_buffer.extend(curr_mask)
            n_missing -= size
            sizes.append(size)

            cur_pos += size

            if n_missing == 0:
                assert len(mask_buffer) == len(x_buffer) == seq_len == len(y_buffer)
                assert sum(sizes) == seq_len
                # we don't want to yield sequences with a mask filled with False
                if any(mask_buffer):
                    yield SequenceMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                    )
                x_buffer, y_buffer = [], []
                mask_buffer = []
                sizes = []
                n_missing = seq_len


def build_dataset(
    pretrain_data: str,
    instruct_data: str,
    instruct_args: InstructArgs,
    tokenizer: Tokenizer,
    seq_len: int,
    rank: int,
    world_size: int,
) -> Iterator[SequenceMaskAndSizes]:
    sources, probabilities = parse_data_sources(
        pretrain_data=pretrain_data, instruct_data=instruct_data
    )

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_args=instruct_args,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
        )
        for source in sources
    ]

    sequence_iterators = [
        sequence_iterator(
            ds_it=it,
            seq_len=seq_len,
        )
        for it in dataset_iterators
    ]

    rng = np.random.RandomState(seed=rank)
    interleaved_iterator = interleave_iterators(
        sequence_iterators, probabilities=probabilities, rng=rng
    )

    return interleaved_iterator


def get_dataset_iterator(
    source: DataDir,
    instruct_args: InstructArgs,
    tokenizer: Tokenizer,
    rank: int,
    world_size: int,
) -> Iterator[TokenSample]:
    jsonl_files = list(source.data_dir.rglob("*jsonl"))
    assert len(jsonl_files) != 0

    while True:
        for jsonl_file in jsonl_files:
            with jsonl_file.open() as f:
                for idx, line in enumerate(f):
                    if not idx % world_size == rank:
                        continue
                    data = json.loads(line)
                    yield encode(
                        data,
                        instruct_args=instruct_args,
                        tokenizer=tokenizer,
                        as_type=source.sample_type,
                    )


def interleave_iterators(iterators: List[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
