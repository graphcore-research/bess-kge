# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
General purpose utilities.
"""

import torch


def gather_indices(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    IPU-friendly gather function like :func:`torch.take_along_dim`
    for 2-dimensional tensors (indices along dim=1).

    :param x: shape: (a, e)
    :param index: shape: (b, k)
    :return: shape: (b, k)
        For all rows of :code:`x`, take the `k` elements on the row with the
        indices specified by the corresponding row of :code:`index`.
        If :code:`b == 1`, the same indices are gathered from all rows of
        :code:`x`; if :code:`a == 1`, all rows in :code:`index` gather from
        :code:`x[0]`; otherwise :code:`a == b` is required.
    """
    bs, sq = x.shape
    _, mask_size = index.shape
    index_flattened = (
        index
        + torch.arange(bs, dtype=torch.int32, device=index.device)
        .mul(torch.tensor(sq, dtype=torch.int32, device=index.device))
        .unsqueeze(1)
    ).view(-1)
    x = torch.index_select(x.view(-1), 0, index_flattened)
    return x.view(-1, mask_size)


def get_entity_filter(
    triples: torch.Tensor, filter_triples: torch.Tensor, filter_mode: str
) -> torch.Tensor:
    """
    Compare two sets of triples: for each triple (h,r,t) in the first set, find
    the entities `e` such that (e,r,t) (or (h,r,e), depending on `filter_mode`)
    appears in the second set of triples.

    :param triples: shape (x, 3)
        The set of triples to construct filters for.
    :param filter_triples: shape (y, 3)
        The set of triples determining the head/tail entities to filter.
    :param filter_mode:
        Set to "h" to look for entities appearing as heads of the same (r,t) pair,
        or to "t" to look for entities appearing as tails of the same (h,r) pair.

    :return: shape (z, 2)
        The sparse filters. Each row is given by a tuple (i, j), with i the index
        of the triple in `triples` to which the filter applies to and j the global
        ID of the entity to filter.
    """
    if filter_mode == "t":
        ent_col = 0
    elif filter_mode == "h":
        ent_col = 2
    else:
        raise ValueError("`filter_mode` needs to be either 'h' or 't'")
    relation_filter = (filter_triples[:, 1]) == triples[:, 1].view(-1, 1)
    entity_filter = (filter_triples[:, ent_col]) == triples[:, ent_col].view(-1, 1)

    filter = (entity_filter & relation_filter).nonzero(as_tuple=False)
    filter[:, 1] = filter_triples[:, 2 - ent_col].view(1, -1)[:, filter[:, 1]]

    return filter


def complex_multiplication(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Batched complex multiplication.

    :param v1: shape: (a, 2*e)
        :code:`v1[:,:e]` real part, :code:`v1[:,e:]` imaginary part.
    :param v2: shape: (a, 2*e)
        :code:`v2[:,:e]` real part, :code:`v2[:,e:]` imaginary part.
    :return: shape: (a, 2*e)
        Row-wise complex multiplication.
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
        :code:`v[:,:e]` real part, :code:`v[:,e:]` imaginary part.
    :param r: shape: (a, e)
        Rotate :code:`v[k]` by :math:`e^{i \pi r[k]}`
    :return: shape: (a, 2*e)
        Row-wise rotated tensors.
    """
    # Always compute sin and cos in fp16, as faster on IPU
    if r.dtype == torch.float32 and r.device.type == "ipu":
        r_cos = torch.cos(r.to(dtype=torch.float16)).to(dtype=torch.float32)
        r_sin = torch.sin(r.to(dtype=torch.float16)).to(dtype=torch.float32)
    else:
        r_cos = torch.cos(r)
        r_sin = torch.sin(r)
    r_complex = torch.concat([r_cos, r_sin], dim=-1)
    return complex_multiplication(v, r_complex)
