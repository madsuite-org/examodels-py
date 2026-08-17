"""Recording model construction — the lazy half of `Core(cache=...)`.

A recording core accepts the same calls as `Core` but forwards nothing to the
backend: expressions become small pure-Python trees, data stays in numpy, and
Julia is never loaded (no `_bridge` attribute is ever touched).  The record
then has two futures:

  - canonicalized into a structural FINGERPRINT plus a DATA DIGEST and
    compared against the sidecar of a previously compiled shared library — a
    match loads the library through cnlpmodels, still with no Julia in the
    process;
  - on a mismatch, REPLAYED through the ordinary eager path, which makes the
    existing implementation the renderer for compilation.

The fingerprint must be at least as fine as the backend's expression *type*:
finer only costs a spurious recompile, coarser would reuse wrong derivative
code.  So everything the type carries is structural here — operator identity
and arity, literal-integer exponents (`x**2` lowers through `literal_pow`;
the exponent lives in the type), `Constant` values, the numeric *type* of
embedded literals, block identity and dimensionality, index-set kind, table
field names and column types, block names.  Values — literals, starts,
bounds, lcon/ucon, index-set contents, table columns — go to the data
digest: the compiled library bakes them, so any change is a miss, but they
do not change the generated code.
"""
import contextvars
import hashlib
import types

import numpy as np

from .node import Block, Constraint, Expression, Product, _columns, is_table

__all__ = ["RecordingCore", "PNode", "tracing"]

_TRACING = contextvars.ContextVar("examodels_recording", default=False)


def tracing():
    """Is a recording trace in progress?  Consulted by the pieces of the eager
    surface that cannot dispatch on an argument type (`Constant`, the lazily
    generated math functions)."""
    return _TRACING.get()


def math_fn(name, opsfn=None):
    """A math function by name that records when given recorded nodes and
    otherwise forwards to the backend-registered operator (loading it — and
    Julia — on first eager use)."""
    def f(*args):
        if any(isinstance(a, PNode) for a in args):
            return PNode(name, *args)
        fn = opsfn
        if fn is None:
            from . import ops  # noqa: PLC0415 — boots Julia; validates the name
            fn = getattr(ops, name)
        return fn(*args)

    f.__name__ = name
    f.__doc__ = f"ExaModels-registered operator `{name}` (records under `Core(cache=...)`)."
    return f


class PNode:
    """A recorded expression node: an operator tag plus children.

    Children are PNodes or plain numbers; numbers keep their Python type,
    because int-vs-float is part of the backend expression type (structure)
    while the value is instance data.
    """

    __slots__ = ("op", "args")

    def __init__(self, op, *args):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", args)

    def __pow__(self, o):
        # Mirror Node.__pow__: a literal int exponent lowers via literal_pow,
        # which puts the exponent in the *type* — structural, not data.
        if type(o) is int:
            return PNode("powi", self, o)
        return PNode("^", self, o)

    def __rpow__(self, o):    return PNode("^", o, self)
    def __neg__(self):        return PNode("neg", self)
    def __pos__(self):        return self
    def __getitem__(self, i): return PNode("at", self, i)

    def __getattr__(self, name):
        """`row.field` inside a traced function — a lookup into the index set."""
        if name.startswith("_"):
            raise AttributeError(name)
        return PNode("field", self, name)

    def __bool__(self):
        raise TypeError(
            "an expression has no truth value: the function is traced ONCE with a "
            "symbolic index, so `if i ...` cannot be evaluated at trace time. Branch "
            "in the data (start / lower / upper / the index set) instead."
        )

    def __repr__(self):
        return f"<recorded {self.op}/{len(self.args)}>"


def _pbin(op, swap):
    def f(self, o):
        return PNode(op, o, self) if swap else PNode(op, self, o)
    return f


for _name, _op in {"add": "+", "sub": "-", "mul": "*", "truediv": "/"}.items():
    setattr(PNode, f"__{_name}__", _pbin(_op, False))
    setattr(PNode, f"__r{_name}__", _pbin(_op, True))


class PTupleSym(PNode):
    """The recorded symbolic index for a product index set (mirrors TupleNode):
    unpacking yields one positional lookup per dimension, numbered from 1."""

    __slots__ = ("_arity",)

    def __init__(self, arity):
        super().__init__("sym")
        object.__setattr__(self, "_arity", arity)

    def __iter__(self):
        return iter(tuple(PNode("at", self, k + 1) for k in range(self._arity)))

    def __len__(self):
        return self._arity


class RBlock(Block):
    """A recorded block of variables or parameters.  Indexing produces recorded
    nodes; the index arithmetic and bounds checks are `Block`'s own."""

    __slots__ = ("_key",)

    def __init__(self, key, axes, kind="variable"):
        super().__init__(None, axes, kind)
        self._key = key                       # ("var" | "par", ordinal)

    def _axis(self, i, axis):
        if isinstance(i, (int, np.integer)):
            return super()._axis(i, axis)
        return i                              # an index expression, recorded as-is

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(idx) != len(self._axes):
            raise IndexError(
                f"{self!r} takes {len(self._axes)} "
                f"ind{'ices' if len(self._axes) > 1 else 'ex'}, got {len(idx)}")
        return PNode("ref", self, *(self._axis(i, k) for k, i in enumerate(idx)))


class RConstraint(Constraint):
    """A recorded block of constraint rows."""

    __slots__ = ("_key",)

    def __init__(self, key, n, row_offset=0):
        super().__init__(None, n, row_offset)
        self._key = key                       # ("con", ordinal)


def _refuse(args, what):
    from .advanced import EachScenario
    from .node import Node
    for a in args:
        if isinstance(a, EachScenario):
            raise NotImplementedError(
                "two-stage / per-scenario declarations cannot be cached yet; "
                "build without cache=")
        if isinstance(a, Node):
            raise TypeError(
                f"{what} mixes an eager expression into a caching core; with "
                f"Core(cache=...) every block must come from this same core")


def _is_oracle(v):
    from .advanced import Oracle
    return isinstance(v, Oracle)


from .core import Core as _Core  # noqa: E402  (cycle-safe: core never imports us at module level)


class RecordingCore(_Core):
    """`Core(cache=...)` — records the model instead of building it.

    Same surface as `Core`; nothing touches Julia.  `replay()` re-runs the
    record through the ordinary eager path when the cache misses.
    """

    nscen = 0

    def __init__(self, backend=None, minimize=True, nargs=0, cache=True):
        if nargs:
            raise NotImplementedError(
                "recipes (nargs=) cannot be cached yet; build without cache=")
        if backend is not None and (not isinstance(backend, str)
                                    or backend not in ("serial", "cpu")):
            raise ValueError(
                f"a cached model compiles to a CPU shared library; backend "
                f"{backend!r} cannot be cached — build without cache=")
        self._backend = backend
        self._minimize = minimize
        self.cache = cache
        self.args = ()
        self._records = []
        self._counts = {"var": 0, "par": 0, "con": 0}
        self._named = {}

    # -- recording helpers ----------------------------------------------------
    def _ordinal(self, kind):
        k = self._counts[kind]
        self._counts[kind] = k + 1
        return k

    def _trace(self, f, over):
        if isinstance(f, PNode):
            return f
        arity = len(over.axes) if isinstance(over, Product) else 1
        tok = _TRACING.set(True)
        try:
            if arity == 1:
                return f(PNode("sym"))
            sym = PTupleSym(arity)
            code = getattr(f, "__code__", None)
            return f(*sym) if code is not None and code.co_argcount == arity else f(sym)
        finally:
            _TRACING.reset(tok)

    def _over_desc(self, over):
        """(structural kind + data, row count) for an index set."""
        from .recipe import SRange
        if isinstance(over, SRange):
            raise NotImplementedError(
                "recipes (srange) cannot be cached yet; build without cache=")
        if isinstance(over, range):
            if over.step != 1:
                return ("steprange", over.start, over.stop, over.step), len(over)
            return ("range", over.start, over.stop - 1), len(over)
        if is_table(over):
            fields, cols, n = _columns(over)
            return ("table", tuple(fields), tuple(str(c.dtype) for c in cols),
                    tuple(cols)), n
        if isinstance(over, Product):
            return ("product", tuple((a.start, a.stop - 1) for a in over.axes)), len(over)
        raise TypeError(
            f"cannot record the index set {over!r}; with Core(cache=...) use a "
            f"range, a product, or a table of named rows")

    @staticmethod
    def _over_obj(desc):
        """The index set back from its description, for replay."""
        kind = desc[0]
        if kind == "range":
            return range(desc[1], desc[2] + 1)
        if kind == "steprange":
            return range(desc[1], desc[2], desc[3])
        if kind == "product":
            return Product(*(range(lo, hi + 1) for lo, hi in desc[1]))
        fields, dtypes, cols = desc[1], desc[2], desc[3]
        arr = np.empty(len(cols[0]), dtype=[(f, c.dtype) for f, c in zip(fields, cols)])
        for f, c in zip(fields, cols):
            arr[f] = c
        return arr

    # -- the Core surface -----------------------------------------------------
    def add_var(self, *dims, start=0.0, lvar=None, uvar=None, tag=None, name=None):
        from .core import _data
        if tag is not None:
            raise NotImplementedError("tags cannot be cached yet; build without cache=")
        _refuse(dims, "add_var")
        if len(dims) == 1 and (isinstance(dims[0], types.GeneratorType)
                               or callable(dims[0])):
            return self._defined_var(dims[0], start, lvar, uvar)
        for d in dims:
            if isinstance(d, range) and d.step != 1:
                raise TypeError(
                    f"a dimension cannot have a step ({d}); the backend defines "
                    f"lengths only for whole ranges. Declare the whole range and "
                    f"use the stepped one as an index set instead")
        if not dims or not all(isinstance(d, (int, range)) for d in dims):
            raise TypeError("give one integer or range per dimension, "
                            "e.g. add_var(T, N) or add_var(range(2, 11))")
        axes = [range(d) if isinstance(d, int) else d for d in dims]
        k = self._ordinal("var")
        self._records.append({
            "kind": "var", "ordinal": k,
            "axes": tuple((a.start, a.stop - 1) for a in axes),
            "start": _data(start, "start"),
            "lvar": None if lvar is None else _data(lvar, "lvar"),
            "uvar": None if uvar is None else _data(uvar, "uvar"),
            "name": None if name is None else str(name)})
        return self._remember(name, RBlock(("var", k), tuple(axes)))

    def _defined_var(self, f, start, lvar, uvar):
        from .core import _as_function
        f, over = _as_function(f, None)
        if not isinstance(over, range):
            raise TypeError("variables defined by expressions need a range index set")
        block = self.add_var(over, start=start, lvar=lvar, uvar=uvar)
        self.add_con(lambda i: block[i] - f(i), over=over)
        return block

    def add_par(self, values, name=None):
        arr = np.ascontiguousarray(values, dtype=np.float64).ravel()
        k = self._ordinal("par")
        self._records.append({"kind": "par", "ordinal": k, "values": arr,
                              "name": None if name is None else str(name)})
        return self._remember(name, RBlock(("par", k), arr.size, "parameter"))

    def add_expr(self, f, over=None, name=None):
        from .core import _as_function
        f, over = _as_function(f, over)
        return self._remember(name, Expression(f, over))

    def add_obj(self, f, over=None, name=None):
        from .core import _as_function
        if _is_oracle(f):
            raise NotImplementedError(
                "an oracle is a Python callback — it cannot live in a compiled "
                "library; build without cache=")
        f, over = _as_function(f, over)
        over = range(1) if over is None else over
        tree = self._trace(f, over)
        desc, _ = self._over_desc(over)
        self._records.append({"kind": "obj", "tree": tree, "over": desc,
                              "name": None if name is None else str(name)})
        return self

    def add_con(self, *args, over=None, lcon=0.0, ucon=0.0, name=None):
        from .core import _data
        if args and _is_oracle(args[0]):
            raise NotImplementedError(
                "an oracle is a Python callback — it cannot live in a compiled "
                "library; build without cache=")
        _refuse(args, "add_con")
        if args and all(isinstance(a, (int, range)) for a in args):
            axes = [range(d) if isinstance(d, int) else d for d in args]
            k = self._ordinal("con")
            n = 1
            for a in axes:
                n *= len(a)
            self._records.append({
                "kind": "con_dims", "ordinal": k,
                "axes": tuple((a.start, a.stop - 1) for a in axes),
                "lcon": _data(lcon, "lcon"), "ucon": _data(ucon, "ucon")})
            return RConstraint(("con", k), n, 1 - axes[0].start)
        if args and isinstance(args[0], Constraint):
            constraint, f, *rest = args
            return self._augment(constraint, f, rest[0] if rest else over)
        f, *rest = args
        from .core import _as_function
        f, over = _as_function(f, rest[0] if rest else over)
        tree = self._trace(f, over)
        desc, n = self._over_desc(over)
        k = self._ordinal("con")
        start = over.start if isinstance(over, range) else 0
        self._records.append({
            "kind": "con", "ordinal": k, "tree": tree, "over": desc,
            "lcon": _data(lcon, "lcon"), "ucon": _data(ucon, "ucon"),
            "name": None if name is None else str(name)})
        return self._remember(name, RConstraint(("con", k), n, 1 - start))

    def _augment(self, constraint, f, over):
        from .core import _as_function
        if not isinstance(constraint, RConstraint):
            raise TypeError(
                "add_con(handle, ...) on a caching core needs a handle from this "
                "same core")
        f, over = _as_function(f, over)
        tok = _TRACING.set(True)
        try:
            idx, expr = f(PNode("sym"))
        finally:
            _TRACING.reset(tok)
        desc, _ = self._over_desc(over)
        self._records.append({"kind": "aug", "con": constraint._key,
                              "idx": idx, "expr": expr, "over": desc})
        return constraint

    # -- fingerprinting -------------------------------------------------------
    def fingerprint(self):
        """`(fingerprint, data_digest)` — hex sha256 pair.

        The fingerprint hashes what determines the generated code (see the
        module docstring); the digest hashes the values the library bakes.
        Parameter *values* are deliberately excluded from both: they stay live
        through the ABI setter on a loaded library.
        """
        s, d = hashlib.sha256(), hashlib.sha256()
        s.update(f"examodels-record-v0;minimize={self._minimize};"
                 f"backend={self._backend}".encode())
        for r in self._records:
            kind = r["kind"]
            s.update(f"|{kind}".encode())
            if kind == "var":
                s.update(f";ndims={len(r['axes'])};name={r['name']}".encode())
                _d_val(d, np.asarray(r["axes"], dtype=np.int64))
                for key in ("start", "lvar", "uvar"):
                    _d_val(d, r[key])
            elif kind == "par":
                s.update(f";n={r['values'].size};name={r['name']}".encode())
                # values excluded: live per-instance through P_set_value
            elif kind in ("obj", "con"):
                _walk(r["tree"], s, d)
                _h_over(r["over"], s, d)
                s.update(f";name={r['name']}".encode())
                if kind == "con":
                    _d_val(d, r["lcon"])
                    _d_val(d, r["ucon"])
            elif kind == "con_dims":
                s.update(f";ndims={len(r['axes'])}".encode())
                _d_val(d, np.asarray(r["axes"], dtype=np.int64))
                _d_val(d, r["lcon"])
                _d_val(d, r["ucon"])
            elif kind == "aug":
                s.update(f";con={r['con'][1]}".encode())
                _walk(r["idx"], s, d)
                _walk(r["expr"], s, d)
                _h_over(r["over"], s, d)
        return s.hexdigest(), d.hexdigest()

    # -- replay ---------------------------------------------------------------
    def replay(self, backend=None):
        """Re-run the record through the ordinary eager path — Julia boots here.
        Returns the eager `Core`, ready for `Model` or `compile_library`."""
        from .core import Core
        core = Core(backend=self._backend if backend is None else backend,
                    minimize=self._minimize)
        handles = {}
        for r in self._records:
            kind = r["kind"]
            if kind == "var":
                kw = {"start": r["start"]}
                if r["lvar"] is not None:
                    kw["lvar"] = r["lvar"]
                if r["uvar"] is not None:
                    kw["uvar"] = r["uvar"]
                if r["name"] is not None:
                    kw["name"] = r["name"]
                handles[("var", r["ordinal"])] = core.add_var(
                    *(range(lo, hi + 1) for lo, hi in r["axes"]), **kw)
            elif kind == "par":
                kw = {"name": r["name"]} if r["name"] is not None else {}
                handles[("par", r["ordinal"])] = core.add_par(r["values"], **kw)
            elif kind == "obj":
                kw = {"name": r["name"]} if r["name"] is not None else {}
                core.add_obj(_render(r["tree"], handles),
                             over=self._over_obj(r["over"]), **kw)
            elif kind == "con":
                kw = {"name": r["name"]} if r["name"] is not None else {}
                handles[("con", r["ordinal"])] = core.add_con(
                    _render(r["tree"], handles), over=self._over_obj(r["over"]),
                    lcon=r["lcon"], ucon=r["ucon"], **kw)
            elif kind == "con_dims":
                handles[("con", r["ordinal"])] = core.add_con(
                    *(range(lo, hi + 1) for lo, hi in r["axes"]),
                    lcon=r["lcon"], ucon=r["ucon"])
            elif kind == "aug":
                core.add_con(
                    handles[r["con"]],
                    lambda i, _r=r: (_ev(_r["idx"], handles, i),
                                     _ev(_r["expr"], handles, i)),
                    self._over_obj(r["over"]))
        return core

    def build(self):
        """Finish this core, returning a `Model`.

        For now this always replays through Julia; the fingerprint lookup and
        the cnlpmodels load land next.
        """
        return self.replay().build()

    def __repr__(self):
        return f"<Core cache={self.cache!r} ({len(self._records)} records)>"


# -- tree evaluation (replay) and hashing -------------------------------------

def _render(tree, handles):
    """The recorded tree as a function of the symbolic index, for the eager path."""
    return lambda i: _ev(tree, handles, i)


def _ev(t, handles, sym):
    if not isinstance(t, PNode):
        return t
    op, a = t.op, t.args
    if op == "sym":
        return sym
    if op == "ref":
        idx = tuple(_ev(i, handles, sym) for i in a[1:])
        return handles[a[0]._key][idx[0] if len(idx) == 1 else idx]
    if op == "at":
        return _ev(a[0], handles, sym)[_ev(a[1], handles, sym)]
    if op == "field":
        return getattr(_ev(a[0], handles, sym), a[1])
    if op == "powi":
        return _ev(a[0], handles, sym) ** a[1]
    if op == "neg":
        return -_ev(a[0], handles, sym)
    if op == "const":
        from .node import Constant
        return Constant(a[0])
    if op in ("sum", "prod"):
        from . import node
        return getattr(node, op)([_ev(x, handles, sym) for x in a])
    if op in _EV_BIN:
        return _EV_BIN[op](_ev(a[0], handles, sym), _ev(a[1], handles, sym))
    from . import ops  # a registered math function, by name — validates it too
    try:
        fn = getattr(ops, op)
    except AttributeError:
        raise TypeError(
            f"{op!r} was recorded as an operator but ExaModels registers no such "
            f"function — is it misspelled?") from None
    return fn(*(_ev(x, handles, sym) for x in a))


import operator as _operator  # noqa: E402

_EV_BIN = {"+": _operator.add, "-": _operator.sub, "*": _operator.mul,
           "/": _operator.truediv, "^": _operator.pow}


def _walk(t, s, d):
    """One canonical serialization pass: structure into `s`, values into `d`."""
    if isinstance(t, PNode):
        op, a = t.op, t.args
        if op == "ref":
            key = a[0]._key
            s.update(f"(ref:{key[0]}:{key[1]}".encode())
            for i in a[1:]:
                _walk(i, s, d)
            s.update(b")")
        elif op == "field":
            s.update(f"(field:{a[1]}".encode())
            _walk(a[0], s, d)
            s.update(b")")
        elif op == "powi":
            s.update(f"(powi:{a[1]}".encode())      # exponent is in the type
            _walk(a[0], s, d)
            s.update(b")")
        elif op == "const":
            s.update(f"(const:{a[0]!r})".encode())  # value is in the type
        else:
            s.update(f"({op}".encode())
            for x in a:
                _walk(x, s, d)
            s.update(b")")
    elif isinstance(t, bool):
        raise TypeError(f"cannot fingerprint leaf {t!r}")
    elif isinstance(t, (int, np.integer)):
        s.update(b"i")
        d.update(repr(int(t)).encode() + b";")
    elif isinstance(t, (float, np.floating)):
        s.update(b"f")
        d.update(np.float64(t).tobytes())
    else:
        raise TypeError(f"cannot fingerprint leaf {t!r}")


def _h_over(desc, s, d):
    kind = desc[0]
    if kind in ("range", "steprange"):
        s.update(f";over={kind}".encode())
        d.update(repr(desc[1:]).encode())
    elif kind == "product":
        s.update(f";over=product{len(desc[1])}".encode())
        d.update(repr(desc[1]).encode())
    else:                                           # table
        fields, dtypes, cols = desc[1], desc[2], desc[3]
        s.update(f";over=table:{','.join(fields)}:{','.join(dtypes)}".encode())
        for c in cols:
            _d_val(d, c)


def _d_val(d, v):
    if v is None:
        d.update(b"~")
    elif isinstance(v, np.ndarray):
        d.update(f"A{v.dtype}{v.shape}".encode())
        d.update(np.ascontiguousarray(v).tobytes())
    else:
        d.update(b"s")
        d.update(np.float64(v).tobytes())
