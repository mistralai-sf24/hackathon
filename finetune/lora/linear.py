import math

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.args import LoraArgs


def maybe_to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    else:
        return x


def apply_recursively(lst, func):
    if isinstance(lst, list):
        return [apply_recursively(sublist, func) for sublist in lst]
    else:
        return func(lst)


class LoRALinear(nn.Module):
    """
    Implementation of:
        - LoRA: https://arxiv.org/abs/2106.09685
        - QLoRA: https://arxiv.org/abs/2305.14314

    Notes:
        - Freezing is handled at network level, not layer level.
        - Scaling factor controls relative importance of LoRA skip
          connection versus original frozen weight. General guidance is
          to keep it to 2.0 and sweep over learning rate when changing
          the rank.
        - For the quantized version, we not rely on MatMul4bits from
          bitsandbytes since we wish to do activation checkpointing properly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_args: LoraArgs,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert not bias
        self.rank = lora_args.rank
        self.scaling = lora_args.scaling
        self.quantized = lora_args.quantized
        self.block_size = lora_args.block_size
        self.dropout = nn.Dropout(p=lora_args.dropout)
        self.lora_A = nn.Parameter(torch.empty((lora_args.rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, lora_args.rank)))
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge_weight(self):
        weight = self.lora_B.mm(self.lora_A).cpu() * self.scaling

        if not self.quantized:
            weight += self.weight.detach().cpu()
        else:
            weight += (
                bnb.functional.dequantize_4bit(self.qweight, self.quant_state)
                .bfloat16()
                .detach()
                .cpu()
            )

        return weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_name = prefix + "weight"

        # full checkpoint
        if key_name in state_dict:
            w_ref = state_dict[key_name].contiguous().bfloat16().cuda()

            if not self.quantized:
                self.weight = w_ref
            else:
                print(f"Quantizing {key_name} on-the-fly")
                w_4bit, quant_state = bnb.functional.quantize_4bit(
                    w_ref,
                    blocksize=self.block_size,
                    compress_statistics=True,
                    quant_type="nf4",
                )
                self.qweight = w_4bit
                self.quant_state = quant_state
                # avoid AttributeError: 'Params4bit' object has no attribute '_mp_param'
                self.qweight._ddp_ignored = True

        # quantized checkpoint, only for QLoRA
        else:
            assert self.quantized, "Provided checkpoint is already quantized"
            qweight_name = prefix + "qweight"
            quant_state_name = prefix + "quant_state"
            assert (
                qweight_name in state_dict
            ), f"Provided checkpoint in wrong format, need {qweight_name}"
            assert (
                quant_state_name in state_dict
            ), f"Provided checkpoint in wrong format, need {quant_state_name}"
            print(f"Reloading already quantized layer {key_name}")
            self.qweight = state_dict[qweight_name].cuda()
            self.quant_state = apply_recursively(
                state_dict[quant_state_name], maybe_to_cuda
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        del destination[prefix + "lora_A"]
        del destination[prefix + "lora_B"]
        key_name = prefix + "weight"
        weight = self.merge_weight()
        destination[key_name] = weight if keep_vars else weight.detach()

    @property
    def frozen_weight(self):
        if not self.quantized:
            frozen_weight = self.weight
        else:
            assert hasattr(
                self, "qweight"
            ), "Please call `load_state_dict` to trigger quantization"
            frozen_weight = bnb.functional.dequantize_4bit(
                self.qweight, self.quant_state
            )
        return frozen_weight

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.frozen_weight)
        lora = self.dropout(x).mm(self.lora_A.t()).mm(self.lora_B.t())
        result += lora * self.scaling
        return result

    def __repr__(self) -> str:
        name = "LoRA" if not self.quantized else "QLoRA"
        return "{}Linear(in_features={}, out_features={}, r={}, dropout={})".format(
            name, self.in_features, self.out_features, self.rank, self.dropout.p
        )
