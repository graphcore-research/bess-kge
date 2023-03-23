# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import poptorch
import torch


def all_to_all(x: torch.FloatTensor) -> torch.FloatTensor:
    """
    x -- float[n_shard, n_shard, ...]
    returns -- float[n_shard, n_shard, ...]
    """
    if x.shape[0] == 1:
        y = x
    elif poptorch.isRunningOnIpu():
        y = poptorch.custom_op(
            name=f"ReplicatedAllToAll",
            domain_version=1,
            domain="custom.ops",
            inputs=[x],
            example_outputs=[x],
        )[0]
    else:
        y = x.transpose(0, 1)
    return y


def all_gather(x: torch.FloatTensor, n_shard: int) -> torch.FloatTensor:
    """
    x -- float[...]
    returns -- float[n_shard, ...]
    """
    if n_shard == 1:
        return x
    else:
        y = poptorch.custom_op(
            name=f"ReplicatedAllGather",
            domain_version=1,
            domain="custom.ops",
            inputs=[x],
            example_outputs=[torch.zeros(dtype=x.dtype, size=(n_shard, *x.shape))],
        )[0]
        y = y.reshape(n_shard, *x.shape)
        return y
