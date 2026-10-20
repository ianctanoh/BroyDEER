"""
scan_trees.py
Alternative tree topologies for associative (prefix) scans.

jax.lax.associative_scan implements Blelloch's (1990) work-efficient
up-sweep/down-sweep tree. In exact arithmetic any tree gives an identical
result, since associativity guarantees the bracketing of `combine` calls
doesn't matter. Once `combine` performs a *lossy* rank compression, however,
the tree shape determines which intermediate quantities get compressed and
in what order, so different trees can accumulate compression error
differently. This module provides a drop-in alternative tree (Hillis-Steele
/ Sklansky) with the same call signature as jax.lax.associative_scan, so the
two can be swapped behind a `scan_tree` flag for direct comparison.
"""

from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import numpy as jnp


def hillis_steele_scan(fn, elems, reverse: bool = False, axis: int = 0):
    """Inclusive prefix scan via the Hillis-Steele (Sklansky) tree.

    Same interface and semantics as jax.lax.associative_scan: `fn` must be
    an associative binary combine function, `elems` a (possibly nested)
    pytree of arrays each with a leading `axis` dimension of size T.

    Unlike Blelloch's balanced up-sweep/down-sweep tree, Hillis-Steele
    performs ceil(log2(T)) rounds where at round k (0-indexed), element i
    is combined with element i - 2^k for all i >= 2^k, and passed through
    unchanged otherwise. This is a different bracketing of the same prefix
    computation, useful for testing sensitivity to scan-tree topology when
    `fn` is only approximately associative (e.g. under rank compression).
    """
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [lax.rev(elem, [axis]) for elem in elems_flat]

    axis = axis % elems_flat[0].ndim
    T = elems_flat[0].shape[axis]

    def combine(a_flat, b_flat):
        a = tree_unflatten(tree, a_flat)
        b = tree_unflatten(tree, b_flat)
        c = fn(a, b)
        c_flat, _ = tree_flatten(c)
        return c_flat

    def _pad_shift(elem, offset):
        # shift elem forward by `offset` along axis, padding the front with
        # the identity slot (element itself at position 0, unused since the
        # mask below prevents combining out-of-range indices).
        pad_widths = [(0, 0)] * elem.ndim
        pad_widths[axis] = (offset, 0)
        shifted = jnp.pad(elem, pad_widths, mode="edge")
        return lax.slice_in_dim(shifted, 0, T, axis=axis)

    def _mask_for(elem):
        mask_shape = [1] * elem.ndim
        mask_shape[axis] = T
        return mask_shape

    n_rounds = max(1, (T - 1).bit_length())
    current = list(elems_flat)
    idx = jnp.arange(T)
    for k in range(n_rounds):
        offset = 1 << k
        shifted = [_pad_shift(e, offset) for e in current]
        combined = combine(shifted, current)
        active = idx >= offset
        current = [
            jnp.where(active.reshape(_mask_for(orig)), c, orig)
            for c, orig in zip(combined, current)
        ]

    if reverse:
        current = [lax.rev(elem, [axis]) for elem in current]

    return tree_unflatten(tree, current)


def get_scan_fn(scan_tree: str = "blelloch"):
    """Return the associative-scan implementation named by `scan_tree`."""
    if scan_tree == "blelloch":
        return lax.associative_scan
    elif scan_tree == "hillis_steele":
        return hillis_steele_scan
    else:
        raise ValueError(f"Unknown scan_tree: {scan_tree!r}. Choose from 'blelloch', 'hillis_steele'.")
