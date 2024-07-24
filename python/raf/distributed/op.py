# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, invalid-name
"""Collective communication operators"""
from .._op import sym
from .context import get_context


def allreduce(x, computation="sum"):
    """General allreduce operators, take tensor or list of tensors as input."""
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._allreduce(x, computation)


def allgather(x, axis):
    """It performs concatenation across replicas.

    Parameters
    ----------
    x : Tensor | [Tensor]
        The tensor(s) to be concatenated across replicas
    axis : int
        The axis over which concatenation is to be performed

    Returns
    -------
    ret: Tensor | [Tensor]
        Concatenation results
    """
    if not isinstance(x, (tuple, list)):
        x = [x]

    return sym._allgather(x, axis=0)


def reduce(x, root, computation="sum"):
    """Performs reduce operation. Collect data to root rank

    Parameters
    ----------
    x : Tensor or list of Tensor
        Tensor(s) to be reduced
    root: int
        The root rank
    computation: string
        The reduction operation, default is sum

    Returns
    -------
    ret: Tensor
        reduction result
    """
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._reduce(x, root, computation)


def reduce_scatter(x, shapes, shape_indices, computation="sum"):
    """Performs reduction then scatter

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors of equal shape
        replica i receives reduction of x[i] over all replicas
    
    shapes : List[int]
        Concatenated output tensor shapes

    shape_indices : List[int]
        End index of each output tensor shape
    
    computation: string
        The reduction operation, default is sum

    Returns
    -------
    ret: Tensor
        reduction result of x[rank] over all replicas,
        where rank represents rank number of the current process
    """
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._reduce_scatter(x, shapes, shape_indices, computation)


def broadcast(x, root):
    """Performs broadcast

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors on rank root to broadcast
    root : int
        root rank

    Returns
    -------
    ret: List[Tensor]
        broadcast-ed results
    """
    if not isinstance(x, (tuple, list)):
        x = [x]
    return sym._broadcast(x, root)


def send(x, peer, token=None):
    """Send x to peer.
    This operation is blocking for GPU.

    Parameters
    ----------
    x : Tensor
        The tensor to be sent

    peer : int
        The send destination

    token : OptionalTensor
        A frame of data that introduces data dependency so that send will not be reordered

    Returns
    -------
    ret: Tensor
        a tensor of zero dimension, which is equivalent to "no return value"
    """
    return sym._send(x, peer=peer, token=token)


def recv(peer, shape, dtype, token=None):
    """Receive a tensor from peer
    This operation is blocking for GPU.

    Parameters
    ----------
    peer : int
        The peer who sends the tensor

    shape : Tuple[int]
        The shape of the tensor to be received

    dtype : String
        The dtype of the tensor to be received

    token : OptionalTensor
        A frame of data that introduces data dependency so that recv will not be reordered

    Returns
    -------
    ret: Tensor
        the received tensor
    """
    return sym._recv(peer=peer, shape=shape, dtype=dtype, token=token)


def all_to_all(x):
    if not isinstance(x, (tuple, list)):
        x = [x]

    return sym._all_to_all(x)

def all_to_allv(x, send_counts):
    if not isinstance(x, (tuple, list)):
        x = [x]
    if not isinstance(send_counts, (tuple, list)):
        send_counts = [send_counts]

    return sym._all_to_allv(x, send_counts)