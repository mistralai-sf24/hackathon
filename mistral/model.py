import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
import torch.distributed.fsdp.wrap as torch_wrap
from simple_parsing.helpers import Serializable
from torch import nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalCausalMask

from finetune.args import LoraArgs
from finetune.lora import LoRALinear
from mistral.cache import CacheView, RotatingBufferCache
from mistral.rope import apply_rotary_emb, precompute_freqs_cis


@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0

    max_batch_size: int = 0

    lora: LoraArgs = field(default_factory=LoraArgs)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def maybe_lora_layer(args: ModelArgs) -> nn.Module:
    if args.lora.enable:
        MaybeLora = partial(
            LoRALinear,
            lora_args=args.lora,
        )
    else:
        MaybeLora = nn.Linear
    return MaybeLora


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        MaybeLora = maybe_lora_layer(args)
        self.wq = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = MaybeLora(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: AttentionBias,
        cache: Optional[CacheView] = None,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.args.head_dim
            )
            val = val.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.args.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, mask)

        return self.wo(output.view_as(x))


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        MaybeLora = maybe_lora_layer(args)
        self.w1 = MaybeLora(args.dim, args.hidden_dim, bias=False)
        self.w2 = MaybeLora(args.hidden_dim, args.dim, bias=False)
        self.w3 = MaybeLora(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        att_mask: AttentionBias,
        cache: Optional[CacheView] = None,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, att_mask, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, checkpoint: bool = False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            block: torch.nn.Module = TransformerBlock(args=args)
            if checkpoint:
                non_reentrant_wrapper = partial(
                    torch_ckpt.checkpoint_wrapper,
                    checkpoint_impl=torch_ckpt.CheckpointImpl.NO_REENTRANT,
                )
                block = non_reentrant_wrapper(block)

            # LoRA only relies on DDP, not FSDP
            if self.training and not args.lora.enable:
                block = torch_wrap.wrap(block)
            self.layers.append(block)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000, theta=args.rope_theta)

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        h = self.tok_embeddings(input_ids)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
            positions = input_metadata.positions.to(device=self.freqs_cis.device)
            att_mask = input_metadata.mask
        else:
            positions = positions_from_sizes(seqlens, self.freqs_cis.device)
            att_mask = BlockDiagonalCausalMask.from_seqlens(
                seqlens
            ).make_local_attention(self.args.sliding_window)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer_id, layer in enumerate(self.layers):
            cache_view = (
                cache.get_view(layer_id, input_metadata) if cache is not None else None
            )
            h = layer(h, freqs_cis, att_mask, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)

        return self.output(self.norm(h)).float()

    @staticmethod
    def from_folder(
        folder: Path, max_batch_size: int = 1, device="cuda", dtype=torch.float16
    ) -> "Transformer":
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size
        model = Transformer(model_args).to(device=device, dtype=dtype)
        loaded = torch.load(folder / "consolidated.00.pth")
        model.load_state_dict(loaded)
        return model


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        sum([list(range(s)) for s in sizes], []), dtype=torch.long, device=device
    )