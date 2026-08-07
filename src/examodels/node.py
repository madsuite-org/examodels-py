"""Python handles for the backend's expression nodes.

A handle owns no structure of its own: operators forward straight into the
backend, so the expression tree is built there and its algebraic simplifications
apply during tracing.
"""
import numpy as _np

from . import _bridge as _b

__all__ = ["Node", "Block", "Constraint", "Expression", "Records", "Constant"]

_BINARY = {"add": "+", "sub": "-", "mul": "*", "truediv": "/"}


class Node:
    """Handle to a backend expression node."""

    __slots__ = ("_jl",)

    def __init__(self, jlobj):
        self._jl = jlobj

    def __pow__(self, o):
        # Julia lowers `x^2` with a LITERAL exponent to `literal_pow(^, x, Val(2))`,
        # which is what puts the exponent in the type and enables `^2 -> abs2`.
        # Python's `**` would otherwise pass a runtime integer and lose that.
        if type(o) is int:
            return Node(_b.literal_pow(self._jl, o))
        return Node(_b.ops["^"](self._jl, _b.unwrap(o)))

    def __rpow__(self, o):    return Node(_b.ops["^"](_b.unwrap(o), self._jl))
    def __neg__(self):        return Node(_b.ops["-"](self._jl))
    def __pos__(self):        return self
    def __getitem__(self, i): return Node(_b.getidx(self._jl, _b.unwrap(i)))

    def __getattr__(self, name):
        """`row.field` inside a traced function — a lookup into the index set."""
        if name.startswith("_"):
            raise AttributeError(name)
        return Node(_b.getfield_(self._jl, name))

    def __bool__(self):
        raise TypeError(
            "an expression has no truth value: the function is traced ONCE with a "
            "symbolic index, so `if i ...` cannot be evaluated at trace time. Branch "
            "in the data (start / lower / upper / the index set) instead."
        )

    @property
    def julia_type(self):
        """Full parametric backend type — the structural fingerprint of the expression."""
        return str(_b.typestr(self._jl))

    def __repr__(self):
        return f"<Node {self.julia_type[:80]}>"


def _binop(op, swap):
    def f(self, o):
        a, b = (_b.unwrap(o), self._jl) if swap else (self._jl, _b.unwrap(o))
        return Node(_b.ops[op](a, b))
    return f


for _name, _op in _BINARY.items():
    setattr(Node, f"__{_name}__", _binop(_op, False))
    setattr(Node, f"__r{_name}__", _binop(_op, True))


class Block:
    """A block of variables or parameters. Index it to reference one element."""

    __slots__ = ("_jl", "_n", "_kind")

    def __init__(self, jlobj, n, kind="variable"):
        self._jl, self._n, self._kind = jlobj, n, kind

    def __getitem__(self, i):
        # This package is 0-based, the backend is 1-based. A concrete index is
        # shifted here; a symbolic one is shifted once on the index set instead
        # (see core._index_set), which keeps the traced expression identical to
        # the one the backend builds for itself.
        if isinstance(i, (int, _np.integer)):
            i = int(i)
            if not -self._n <= i < self._n:
                raise IndexError(f"index {i} is out of range for {self!r}")
            return Node(_b.getidx(self._jl, i % self._n + 1))
        return Node(_b.getidx(self._jl, _b.unwrap(i)))

    def __len__(self):
        return self._n

    def __repr__(self):
        return f"<{self._kind} block of {self._n}>"


class Constraint:
    """A block of constraint rows. Pass it back to `add_con` to add terms to it."""

    __slots__ = ("_jl", "_n")

    def __init__(self, jlobj, n):
        self._jl, self._n = jlobj, n

    def __len__(self):
        return self._n

    def __repr__(self):
        return f"<constraint block of {self._n}>"


class Expression:
    """A reusable subexpression.

    Subexpressions are inlined at each use — no auxiliary variable, no equality
    constraint — so this is held entirely on the Python side: `s[i]` just applies
    the function again. Uses sharing a structure share derivative code, exactly as
    if the backend had built them.
    """

    __slots__ = ("_f", "_over")

    def __init__(self, f, over):
        self._f = f
        self._over = over if isinstance(over, tuple) else (over,)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(idx) != len(self._over):
            raise IndexError(f"{self!r} is indexed by {len(self._over)}, got {len(idx)}")
        for i, over in zip(idx, self._over):
            if isinstance(i, (int, _np.integer)) and i not in over:
                raise IndexError(f"index {i} is outside {over}")
        return self._f(*idx)

    def __len__(self):
        n = 1
        for o in self._over:
            n *= len(o)
        return n

    def __repr__(self):
        return f"<subexpression {' x '.join(str(len(o)) for o in self._over)}>"


class Records:
    """A table of rows to index a model over, instead of a plain range.

        arcs = Records({"bus": [...], "i": [...]}, index=["bus", "i"])
        core.constrain(lambda a: p[a.i] - ..., over=arcs)

    Columns named in `index` hold positions of variables; they are 0-based like
    everything else here, and converted for the backend once, on the way in.
    """

    __slots__ = ("_jl", "_n", "_fields")

    def __init__(self, columns, index=()):
        import numpy as np
        index = set(index)
        names, cols, n = [], [], None
        for name, values in columns.items():
            a = np.asarray(values)
            a = np.ascontiguousarray(a, dtype=np.int64) + 1 if name in index else \
                np.ascontiguousarray(a, dtype=np.float64)
            if n is None:
                n = a.size
            elif a.size != n:
                raise ValueError(f"column {name!r} has {a.size} rows, expected {n}")
            names.append(name)
            cols.append(a)
        self._jl = _b.mkrecords(names, cols)
        self._n, self._fields = n, tuple(names)

    def __len__(self):
        return self._n

    def __repr__(self):
        return f"<Records {self._n} rows: {', '.join(self._fields)}>"


def Constant(v):
    """A constant whose value is carried in the backend *type*, enabling the
    algebraic simplifications (`x * Constant(1) -> x`)."""
    return Node(_b.EM.Constant(v))
