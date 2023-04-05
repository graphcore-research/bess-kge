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
