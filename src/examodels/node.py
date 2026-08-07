"""Python handles for ExaModels' Julia expression nodes.

A `Node` owns no structure of its own: every operator forwards straight into
Julia, so the expression tree is built by ExaModels itself and the algebraic
rewrites in `specialization.jl` fire during tracing.
"""
import numpy as _np

from . import _bridge as _b

__all__ = ["Node", "Variable", "Constant"]


class Node:
    """Handle to an `ExaModels.AbstractNode`. Operators dispatch into Julia."""

    __slots__ = ("_jl",)

    def __init__(self, jlobj):
        self._jl = jlobj

    def __add__(self, o):      return Node(_b.ops["+"](self._jl, _b.unwrap(o)))
    def __radd__(self, o):     return Node(_b.ops["+"](_b.unwrap(o), self._jl))
    def __sub__(self, o):      return Node(_b.ops["-"](self._jl, _b.unwrap(o)))
    def __rsub__(self, o):     return Node(_b.ops["-"](_b.unwrap(o), self._jl))
    def __mul__(self, o):      return Node(_b.ops["*"](self._jl, _b.unwrap(o)))
    def __rmul__(self, o):     return Node(_b.ops["*"](_b.unwrap(o), self._jl))
    def __truediv__(self, o):  return Node(_b.ops["/"](self._jl, _b.unwrap(o)))
    def __rtruediv__(self, o): return Node(_b.ops["/"](_b.unwrap(o), self._jl))
    def __neg__(self):         return Node(_b.ops["-"](self._jl))
    def __pos__(self):         return self

    def __pow__(self, o):
        if type(o) is int:      # mirror Julia's literal-exponent lowering
            return Node(_b.literal_pow(self._jl, o))
        return Node(_b.ops["^"](self._jl, _b.unwrap(o)))

    def __rpow__(self, o):     return Node(_b.ops["^"](_b.unwrap(o), self._jl))
    def __getitem__(self, i):  return Node(_b.getidx(self._jl, _b.unwrap(i)))

    def __bool__(self):
        raise TypeError(
            "an ExaModels expression has no truth value: the function is traced ONCE "
            "with a symbolic index, so `if i ...` cannot be evaluated at trace time. "
            "Branch in the data (start / lcon / ucon / the index set) instead."
        )

    @property
    def julia_type(self):
        """Full parametric Julia type — the structural fingerprint of the expression."""
        return str(_b.typestr(self._jl))

    def __repr__(self):
        return f"<Node {self.julia_type[:80]}>"


class Variable:
    """Handle to an `ExaModels.Variable`; `x[i]` builds a Julia `Var` node."""

    __slots__ = ("_jl", "_n")

    def __init__(self, jlobj, n=None):
        self._jl, self._n = jlobj, n

    def __getitem__(self, i):
        # This package is 0-based. The backend is 1-based, and the +1 is applied in
        # exactly one place per index kind: here for a concrete integer, and on the
        # index set for a symbolic one (see core._index_set). Doing it here for
        # symbolic indices too would add a redundant node to every expression.
        if isinstance(i, (int, _np.integer)):
            i = int(i)
            if self._n is not None:
                if not -self._n <= i < self._n:
                    raise IndexError(
                        f"index {i} is out of range for a block of {self._n} variables")
                i %= self._n                      # allow the usual negative indexing
            return Node(_b.getidx(self._jl, i + 1))
        return Node(_b.getidx(self._jl, _b.unwrap(i)))

    def __len__(self):
        if self._n is None:
            raise TypeError("length unknown for this variable block")
        return self._n

    @property
    def julia_type(self):
        return str(_b.typestr(self._jl))

    def __repr__(self):
        return f"<Variable n={self._n}>"


def Constant(v):
    """`ExaModels.Constant{T}` — the value is carried as a Julia *type* parameter,
    which is what enables the rewrites in `specialization.jl` (`x*Constant(1) -> x`)."""
    return Node(_b.EM.Constant(v))
