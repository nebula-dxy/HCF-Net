from typing import Optional

import torch


def _infer_dim_size(index: torch.Tensor, dim_size: Optional[int]) -> int:
    if dim_size is not None:
        return dim_size
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("scatter_add fallback only supports dim=0")
    dim_size = _infer_dim_size(index, dim_size)
    out_shape = list(src.shape)
    out_shape[0] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_add(src, index, dim=dim, dim_size=dim_size)
    dim_size = out.shape[0]
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    ones = torch.ones(index.shape[0], dtype=src.dtype, device=src.device)
    count.index_add_(0, index, ones)
    count = count.clamp_min_(1.0)
    while count.dim() < out.dim():
        count = count.unsqueeze(-1)
    return out / count


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None):
    if dim != 0:
        raise NotImplementedError("scatter_max fallback only supports dim=0")
    dim_size = _infer_dim_size(index, dim_size)
    if src.dim() == 1:
        out = torch.full((dim_size,), -torch.inf, dtype=src.dtype, device=src.device)
        argmax = torch.full((dim_size,), -1, dtype=torch.long, device=src.device)
        for i in range(src.shape[0]):
            idx = int(index[i].item())
            if src[i] > out[idx]:
                out[idx] = src[i]
                argmax[idx] = i
        return out, argmax
    out = torch.full((dim_size, src.shape[1]), -torch.inf, dtype=src.dtype, device=src.device)
    argmax = torch.full((dim_size, src.shape[1]), -1, dtype=torch.long, device=src.device)
    for i in range(src.shape[0]):
        idx = int(index[i].item())
        better = src[i] > out[idx]
        out[idx] = torch.where(better, src[i], out[idx])
        argmax[idx] = torch.where(better, torch.full_like(argmax[idx], i), argmax[idx])
    return out, argmax


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("scatter_softmax fallback only supports dim=0")
    max_per_group, _ = scatter_max(src, index, dim=dim)
    shifted = src - max_per_group[index]
    exp = torch.exp(shifted)
    denom = scatter_add(exp, index, dim=dim)
    return exp / denom[index].clamp_min(1e-12)
