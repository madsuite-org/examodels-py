"""Helpers for the test suite. Not part of the user-facing API."""
from . import _bridge as _b
from .node import Node

__all__ = ["same_structure", "reference_trace", "full_types"]


def full_types(on=True):
    """Show complete parametric types (they are abbreviated by default)."""
    _b.seval(f"ExaModels.fulltype_display!({str(bool(on)).lower()})")


def same_structure(a, b):
    """Exact structural identity of two expressions.

    Compares the underlying types directly rather than their printed form: the
    printed form is abbreviated by a custom `show` method, so string comparison
    silently passes for almost any pair.
    """
    return bool(_b.same_ty(_b.unwrap(a), _b.unwrap(b)))


def reference_trace(var, generator_src, **extra):
    """Trace a *Julia* generator the way ExaModels does, as a reference oracle.

    This is the one thing the user-facing API deliberately cannot express, and the
    only reason the test suite needs it: it produces the ground truth that a
    Python-built expression is compared against.
    """
    names = ["x", *extra]
    f = _b.seval(
        f"({', '.join(names)}) -> begin gen = ({generator_src}); "
        f"gen.f(ExaModels.DataSource()) end")
    return Node(f(_b.unwrap(var), *(_b.unwrap(v) for v in extra.values())))
