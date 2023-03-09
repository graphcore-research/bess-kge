import torch
import poptorch

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
        y = x.transpose(0,1)
    return  y