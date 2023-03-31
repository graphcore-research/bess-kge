# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import poptorch
import torch


def gather_indices(x, index):
    """
    IPU-friendly gather function like torch.take_along_dim
    for 2-dimensional tensors (indices along dim=1).

    :param x: shape: (a, e)
    :param index: shape: (a, k) or (1, k)
    :return: shape: (a, k)
        For all rows of `x`, take the `k` elements on
        the row with the indices specified by
        the corresponding row of `index`.
        If `index.shape == 1`, the same mask is applied to all rows.
    """
    bs, sq = x.shape
    _, mask_size = index.shape
    index_flattened = (index + torch.arange(bs).mul(sq).unsqueeze(1)).view(-1)
    x = torch.index_select(x.view(-1), 0, index_flattened)
    return x.view(bs, mask_size)
