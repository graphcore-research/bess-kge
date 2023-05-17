# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
BESS-KGE is a Python package for knowledge graph embedding on IPUs using
`PopTorch <https://github.com/graphcore/poptorch>`_ and the distributed framework
`BESS <https://arxiv.org/abs/2211.12281>`_.
"""


def load_custom_ops_so() -> None:
    import ctypes
    import pathlib
    import sysconfig

    """
    Code derived from
    https://github.com/graphcore-research/poptorch-experimental-addons/
    Copyright (c) 2023 Graphcore Ltd
    Licensed under the MIT License
    """

    root = pathlib.Path(__file__).parent.parent.absolute()
    name = "besskge_custom_ops.so"
    paths = [
        root / "build" / name,
        (root / name).with_suffix(sysconfig.get_config_vars()["SO"]),
    ]
    for path in paths:
        if path.exists():
            ctypes.cdll.LoadLibrary(str(path))
            return
    raise ImportError(  # pragma: no cover
        f"Cannot find extension library {name} - tried {[str(p) for p in paths]}"
    )


load_custom_ops_so()


from . import (  # NOQA:F401,E402
    batch_sampler,
    bess,
    dataset,
    embedding,
    loss,
    metric,
    negative_sampler,
    scoring,
    sharding,
    utils,
)
