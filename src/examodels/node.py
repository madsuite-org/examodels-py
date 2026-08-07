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
    """A block of variables or parameters, of one or more dimensions.

        x = core.add_var(10)          # x[i]
        y = core.add_var((T, N))      # y[t, i]
    """

    __slots__ = ("_jl", "_shape", "_kind")

    def __init__(self, jlobj, shape, kind="variable"):
        self._jl = jlobj
        self._shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self._kind = kind

    @property
    def shape(self):
        return self._shape

    def _axis(self, i, axis):
        """0-based here, 1-based in the backend.

        A concrete index is shifted here; a symbolic one is left alone, because the
        index set it comes from has already been shifted (see core._index_set).
        Doing both would double-count.
        """
        if isinstance(i, (int, _np.integer)):
            i, n = int(i), self._shape[axis]
            if not -n <= i < n:
                raise IndexError(
                    f"index {i} is out of range for axis {axis} of {self!r}")
            return i % n + 1
        return _b.unwrap(i)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(idx) != len(self._shape):
            raise IndexError(
                f"{self!r} takes {len(self._shape)} "
                f"ind{'ices' if len(self._shape) > 1 else 'ex'}, got {len(idx)}")
        shifted = [self._axis(i, k) for k, i in enumerate(idx)]
        if len(shifted) == 1:
            return Node(_b.getidx(self._jl, shifted[0]))
        return Node(_b.getidxn(self._jl, *shifted))

    def __len__(self):
        n = 1
        for d in self._shape:
            n *= d
        return n

    def __repr__(self):
        dims = " x ".join(str(d) for d in self._shape)
        return f"<{self._kind} block {dims}>"


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


class _RecordCursor:
    """What iterating a `Records` yields to a generator expression.

    A generator expression evaluates its outermost iterable eagerly and keeps the
    *iterator*. Handing it one of these means the index set can be recovered as the
    `Records` itself, rather than as the rows it was built from — so the converted
    table is reused instead of rebuilt.
    """

    __slots__ = ("records",)

    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class Records:
    """A table of rows to index a model over, instead of a range.

        Gen = namedtuple("Gen", "i bus cost")
        gens = Records([Gen(0, 1, 5.0), Gen(1, 2, 6.0)], index=["i", "bus"])
        core.add_obj(g.cost * pg[g.i]**2 for g in gens)

    Rows are named tuples, mirroring how the backend holds them. Fields named in
    `index` hold positions of variables; they are 0-based like everything else and
    converted once, on the way in.
    """

    __slots__ = ("_jl", "_n", "_fields")

    def __init__(self, rows, index=()):
        import numpy as np
        rows = list(rows)
        if not rows:
            raise ValueError("a Records table needs at least one row")
        if isinstance(rows[0], dict):
            raise TypeError(
                "rows must be named tuples, not dicts — define one with "
                "collections.namedtuple so the field names travel with the data")
        fields = getattr(type(rows[0]), "_fields", None)
        if fields is None:
            raise TypeError(f"rows must be named tuples, got {type(rows[0]).__name__}")
        index = set(index)
        cols = []
        for k, name in enumerate(fields):
            values = [r[k] for r in rows]
            cols.append(np.ascontiguousarray(values, dtype=np.int64) + 1 if name in index
                        else np.ascontiguousarray(values, dtype=np.float64))
        self._jl = _b.mkrecords(list(fields), cols)
        self._n, self._fields = len(rows), tuple(fields)

    def __len__(self):
        return self._n

    def __iter__(self):
        return _RecordCursor(self)

    def __repr__(self):
        return f"<Records {self._n} rows: {', '.join(self._fields)}>"


def Constant(v):
    """A constant whose value is carried in the backend *type*, enabling the
    algebraic simplifications (`x * Constant(1) -> x`)."""
    return Node(_b.EM.Constant(v))
