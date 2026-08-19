"""Python handles for the backend's expression nodes.

A handle owns no structure of its own: operators forward straight into the
backend, so the expression tree is built there and its algebraic simplifications
apply during tracing.
"""
import numpy as _np

from . import _bridge as _b

__all__ = ["Node", "TupleNode", "Block", "Constraint", "Expression", "Product",
           "Constant", "sum", "prod"]

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

    __slots__ = ("_jl", "_axes", "_kind")

    def __init__(self, jlobj, axes, kind="variable"):
        self._jl = jlobj
        #: one `range` per dimension, in this package's own 0-based terms
        self._axes = tuple(range(d) if isinstance(d, int) else d for d in
                           (axes if isinstance(axes, tuple) else (axes,)))
        self._kind = kind

    @property
    def shape(self):
        return tuple(len(a) for a in self._axes)

    @property
    def axes(self):
        return self._axes

    def _axis(self, i, axis):
        """Indices are passed through unchanged.

        The block's backend dimension is declared with these same bounds, so an
        index means the same thing on both sides -- whether it addresses a variable
        or is used as a number in the surrounding arithmetic.
        """
        if isinstance(i, (int, _np.integer)):
            i, ax = int(i), self._axes[axis]
            if i < 0 and ax.start == 0:
                i += len(ax)                       # ordinary negative indexing
            if i not in ax:
                raise IndexError(
                    f"index {i} is out of range for axis {axis} "
                    f"({ax.start}..{ax.stop - 1}) of {self!r}")
            return i
        from ._record import PNode
        if isinstance(i, PNode):
            raise TypeError(
                f"{self!r} belongs to a plain Core, but this index is being "
                f"recorded for a Core(cache=...); a cached model must build "
                f"every block from its own core")
        return _b.unwrap(i)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(idx) != len(self._axes):
            raise IndexError(
                f"{self!r} takes {len(self._axes)} "
                f"ind{'ices' if len(self._axes) > 1 else 'ex'}, got {len(idx)}")
        shifted = [self._axis(i, k) for k, i in enumerate(idx)]
        if len(shifted) == 1:
            return Node(_b.getidx(self._jl, shifted[0]))
        return Node(_b.getidxn(self._jl, *shifted))

    def __len__(self):
        n = 1
        for a in self._axes:
            n *= len(a)
        return n

    def __repr__(self):
        # A placeholder-sized axis has no length to print until the model is built.
        dims = " x ".join(
            "?" if not isinstance(a, range)
            else str(len(a)) if a.start == 0 else f"{a.start}..{a.stop - 1}"
            for a in self._axes)
        return f"<{self._kind} block {dims}>"

class TupleNode(Node):
    """The symbolic index for a product index set.

    Unpacking it (`t, i = ...`) yields one positional lookup per dimension, which
    is what the backend does when it traces `expr for t in .., i in ..`.
    """

    __slots__ = ("_arity",)

    def __init__(self, jlobj, arity):
        super().__init__(jlobj)
        self._arity = arity

    def __iter__(self):
        # tuple components are looked up positionally, and the backend numbers
        # those from 1 regardless of what the index values are
        return iter(tuple(Node(_b.getidx(self._jl, k + 1)) for k in range(self._arity)))

    def __len__(self):
        return self._arity

class _ProductCursor:
    """What iterating a `Product` yields to a generator expression."""

    __slots__ = ("product",)

    def __init__(self, product):
        self.product = product

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

class Product:
    """A rectangular index set: one range per dimension.

        core.add_con(x[t, i] - x[t-1, i] for t, i in exa.product(range(1, T), range(N)))
    """

    __slots__ = ("axes",)

    def __init__(self, *axes):
        if not axes or not all(isinstance(a, range) and a.step == 1 for a in axes):
            raise TypeError("give one unit-step range per dimension, "
                            "e.g. product(range(1, T), range(N))")
        self.axes = tuple(axes)

    def __len__(self):
        n = 1
        for a in self.axes:
            n *= len(a)
        return n

    def __iter__(self):
        return _ProductCursor(self)

    def __repr__(self):
        return f"<product {' x '.join(str(len(a)) for a in self.axes)}>"

class Constraint:
    """A block of constraint rows. Pass it back to `add_con` to add terms to it."""

    __slots__ = ("_jl", "_n", "row_offset")

    def __init__(self, jlobj, n, row_offset=0):
        self._jl, self._n = jlobj, n
        #: added to a row index when adding terms — see Core.add_con
        self.row_offset = row_offset

    def __len__(self):
        if self._n is None:
            raise TypeError(
                "this constraint block was written over a placeholder index set, "
                "so its row count is not known until the model is built — read it "
                "from the model (`model.ncon`) instead."
            )
        return self._n

    def __repr__(self):
        return ("<constraint block, sized when the model is built>"
                if self._n is None else f"<constraint block of {self._n}>")

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
        axes = getattr(over, "axes", None)           # a Product indexes per axis
        self._over = axes if axes is not None else (
            over if isinstance(over, tuple) else (over,))

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

def _field_names(row):
    """The ordered field names of a row, whatever kind of record it is.

    Python has several ways to spell what Julia writes as a struct or a named
    tuple, and they are all reasonable things to hold model data in.
    """
    for attr in ("_fields",):                       # namedtuple, typing.NamedTuple
        names = getattr(type(row), attr, None)
        if names:
            return list(names)
    if hasattr(type(row), "__dataclass_fields__"):  # dataclasses.dataclass
        import dataclasses
        return [f.name for f in dataclasses.fields(row)]
    if hasattr(type(row), "__attrs_attrs__"):       # attrs
        return [a.name for a in type(row).__attrs_attrs__]
    slots = getattr(type(row), "__slots__", None)   # a plain class with __slots__
    if slots:
        return [slots] if isinstance(slots, str) else list(slots)
    return None

def _columns(rows, index=()):
    """A sequence of named rows -> `(fields, cols, n)` numpy columns.

    Accepts a numpy structured array (which is already exactly that) or anything
    iterable of named tuples. Column types come from the values: whole numbers stay
    integers, so a field can be used as a variable index, and anything else becomes
    a float.
    """
    import numpy as np
    if isinstance(rows, np.ndarray) and rows.dtype.names:
        fields = list(rows.dtype.names)
        cols = [np.ascontiguousarray(rows[f]) for f in fields]
        n = len(rows)
    else:
        rows = list(rows)
        if not rows:
            raise ValueError("an index set needs at least one row")
        if isinstance(rows[0], dict):
            raise TypeError(
                "rows must be named tuples, not dicts -- define one with "
                "collections.namedtuple so the field names travel with the data")
        fields = _field_names(rows[0])
        if fields is None:
            raise TypeError(
                f"an index set must be a range, a product, a numpy structured array, "
                f"or a sequence of rows with named fields -- a named tuple, a "
                f"dataclass, or anything with __slots__; got "
                f"{type(rows[0]).__name__}")
        index, cols, n = set(index), [], len(rows)
        for name in fields:
            values = [getattr(r, name) for r in rows]
            whole = all(isinstance(x, (int, np.integer)) for x in values)
            cols.append(np.ascontiguousarray(
                values, dtype=np.int64 if (name in index or (not index and whole))
                else np.float64))
    return fields, cols, n

def _table(rows, index=()):
    """A sequence of named rows -> the backend's table of named, typed fields.

    Columns cross the boundary as whole arrays BEFORE the row loop: element
    access on a Python-wrapped column dynamic-dispatches per value, and the
    row builder touches every value of every column — measured at 24 s for a
    9 241-bus OPF case (fully warm) against ~1 s with the bulk conversion."""
    fields, cols, n = _columns(rows, index)
    jcols = tuple((_b.vec_i64 if c.dtype.kind == "i" else _b.vec_f64)(c)
                  for c in cols)
    return _b.mkrecords(fields, jcols), n

def is_table(over):
    """Is this a table of named rows?

    A sequence of dicts is refused rather than ignored: it is the obvious thing to
    reach for, and letting it fall through would fail much further along.
    """
    import numpy as np
    if isinstance(over, np.ndarray):
        return bool(over.dtype.names)
    if isinstance(over, (str, bytes, range)):
        return False
    try:
        first = next(iter(over))
    except (TypeError, StopIteration):
        return False
    if isinstance(first, dict):
        raise TypeError(
            "an index set of dicts has no field order -- use a named tuple, a "
            "dataclass, or a numpy structured array, so the field names travel "
            "with the data")
    return _field_names(first) is not None

def sum(nodes):
    """A single summation node over `nodes` (the backend's `exa_sum`)."""
    nodes = list(nodes)
    from ._record import PNode
    if any(isinstance(n, PNode) for n in nodes):
        return PNode("sum", *nodes)
    return Node(_b.exa_sum([_b.unwrap(n) for n in nodes]))

def prod(nodes):
    """A single product node over `nodes` (the backend's `exa_prod`)."""
    nodes = list(nodes)
    from ._record import PNode
    if any(isinstance(n, PNode) for n in nodes):
        return PNode("prod", *nodes)
    return Node(_b.exa_prod([_b.unwrap(n) for n in nodes]))

def Constant(v):
    """A constant whose value is carried in the backend *type*, enabling the
    algebraic simplifications (`x * Constant(1) -> x`)."""
    from ._record import PNode, tracing
    if tracing():
        return PNode("const", v)
    return Node(_b.EM.Constant(v))
