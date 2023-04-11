# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch


def gather_indices(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    IPU-friendly gather function like torch.take_along_dim
    for 2-dimensional tensors (indices along dim=1).

    :param x: shape: (a, e)
    :param index: shape: (b, k)
    :return: shape: (b, k)
        For all rows of `x`, take the `k` elements on
        the row with the indices specified by
        the corresponding row of `index`.
        If `b == 1`, the same indices are gathered from all rows of `x`;
        if `a == 1`, all rows in `index` gather from `x[0]`;
        otherwise `a == b` is required.
    """
    bs, sq = x.shape
    _, mask_size = index.shape
    index_flattened = (index + torch.arange(bs).mul(sq).unsqueeze(1)).view(-1)
    x = torch.index_select(x.view(-1), 0, index_flattened)
    return x.view(-1, mask_size)


def complex_multiplication(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Batched complex multiplication.

    :param v1: shape: (a, 2*e)
        `v1[:,:e]` real part, `v1[:,e:]` imaginary part.
    :param v2: shape: (a, 2*e)
        `v2[:,:e]` real part, `v2[:,e:]` imaginary part.
    :return: shape: (a, 2*e)
        Row-wise complex multiplcation.
    """
    cutpoint = v1.shape[-1] // 2
    v1_re, v1_im = torch.split(v1, cutpoint, dim=-1)
    v2_re, v2_im = torch.split(v2, cutpoint, dim=-1)

    return torch.concat(
        [v1_re * v2_re - v1_im * v2_im, v1_re * v2_im + v1_im * v2_re], dim=-1
    )


def complex_rotation(v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""
    Batched rotation by unitary tensors.

    :param v: shape: (a, 2*e)
        Complex tensor to rotate:
        `v[:,:e]` real part, `v[:,e:]` imaginary part.
    :param r: shape: (a, e)
        Rotate `v[i]` by
    .. math::
        \exp{i * \pi * r[i]}
    :return: shape: (a, 2*e)
        Row-wise rotated tensors.
    """
    r_complex = torch.concat([torch.cos(r), torch.sin(r)], dim=-1)
    return complex_multiplication(v, r_complex)
