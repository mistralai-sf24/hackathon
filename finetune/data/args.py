import logging
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class InstructArgs(Serializable):
    add_flag: bool = True  # If True, the flags "[INST]" and "[/INST]" will be added at the beginning and end of the interaction with `is_bot` set to False.


@dataclass()
class DataArgs(Serializable):
    # The data arguments `data` and `instruct_data` are a string in the format
    # "data_source_dir_1:weight_1,data_source_dir_2:weight_2,...". The weight
    # will be used to sample the data sources. If the sum of the weights is
    # not 1 when concatenating the two arguments `data` and `instruct_data`,
    # it will be normalized. The data sources folders must contain jsonl files.
    # If the value is an empty string, no data will be used for the corresponding
    # data type.
    data: str = ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key. See Readme for more details. Can be left empty.
    instruct_data: str = ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "interactions" key. See Readme for more details. Can be left empty.
    instruct: InstructArgs = field(default_factory=InstructArgs)
