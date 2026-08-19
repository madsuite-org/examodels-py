"""Model construction: variables, parameters, subexpressions, objective, constraints."""
import dis
import types

import numpy as np

from . import _bridge as _b
from .node import (
    Block,
    Constraint,
    Expression,
    Node,
    Product,
    TupleNode,
    _ProductCursor,
    _table,
    is_table,
)

__all__ = ["Core", "trace", "backends", "install_backend"]


def trace(f):
    """Call `f` ONCE with a symbolic index and return the expression it builds.

    The loop never runs at build time — one traced expression describes every row —
    so `f` must not branch on the index. Anything index-dependent belongs in the
    data (`start`, `lower`, `upper`, or the index set), which is evaluated per index
    in the ordinary way.
    """
    return f(Node(_b.EM.DataSource()))


def _index_set(over):
    """Describe an index set for the backend, using the caller's own bounds.

    Returns *arguments*, not an iterator: a range that round-trips through Python
    comes back as a `StepRange`, which several of the backend's size and dispatch
    paths reject, so the iterator is constructed on the backend side.
    """
    from .recipe import SRange
    if isinstance(over, SRange):
        # Bounds may be placeholders, so the length is unknown here; the backend
        # sizes the block when the model is built.
        return (over._jl(),), None
    if isinstance(over, range):
        if over.step != 1:
            # A stepped range is a fine index *set* -- it is just a list of indices.
            # It cannot be a variable *dimension*: the backend defines `_length`
            # only for Int and UnitRange.
            return (_b.int_vector(list(over)),), len(over)
        return (over.start, over.stop - 1), len(over)
    if is_table(over):
        jl, n = _table(over)
        return (jl,), n
    if isinstance(over, Product):
        los = [a.start for a in over.axes]
        his = [a.stop - 1 for a in over.axes]
        return (_b.product(los, his),), len(over)
    return (_b.unwrap(over),), len(over)


def _data(v, what):
    from .recipe import Arg
    if isinstance(v, Arg):
        # A placeholder is data supplied at build time; it passes through as the
        # backend node it stands for, and is sized when the model is built.
        return v._jl
    if callable(v):
        raise TypeError(
            f"{what} is data, not an expression: pass a number or an array. It is "
            f"evaluated once per index, so build it with a list comprehension or numpy.")
    return float(v) if np.isscalar(v) else np.ascontiguousarray(v, dtype=np.float64)


def _node(f, over=None):
    """Trace `f`, giving it an index of the right arity for `over`."""
    if isinstance(f, Node):
        return f
    arity = len(over.axes) if isinstance(over, Product) else 1
    if arity == 1:
        got = trace(f)
    else:
        sym = TupleNode(_b.EM.DataSource(), arity)
        code = getattr(f, "__code__", None)
        # a user function written `lambda t, i: ...` takes the components; one
        # derived from a generator expression takes the tuple and unpacks it itself
        got = f(*sym) if code is not None and code.co_argcount == arity else f(sym)
    from ._record import PNode
    if isinstance(got, PNode):
        raise TypeError(
            "this expression uses a block from a Core(cache=...), but is being "
            "added to a plain Core; a cached model must build every block from "
            "its own core")
    return got


def _cell(value):
    return (lambda v: (lambda: v).__closure__[0])(value)


def _iterable_of(iterator):
    """The thing a generator expression is iterating, without consuming it."""
    if isinstance(iterator, _ProductCursor):
        return iterator.product
    try:                                    # range_iterator, list_iterator, ...
        fn, args, *_ = iterator.__reduce__()
        if args:
            return args[0]
    except (AttributeError, TypeError):
        pass
    raise TypeError("could not recover the index set from the generator expression; "
                    "pass a function and `over=` instead")


def _as_function(f, over):
    """Accept either `lambda i: expr` with `over=`, or `(expr for i in over)`.

    A generator expression carries both the body and the index set, so it reads
    the way the same model reads in the backend's own syntax. It is not iterated:
    the body is re-invoked once with a symbolic index, exactly as for a function.
    """
    if not isinstance(f, types.GeneratorType):
        return f, over
    frame = f.gi_frame
    if frame is None or ".0" not in frame.f_locals:
        raise TypeError("expected a generator expression, like (x[i]**2 for i in ...)")
    if sum(i.opname == "FOR_ITER" for i in dis.get_instructions(f.gi_code)) > 1:
        raise TypeError(
            "a generator expression with more than one `for` cannot be traced; "
            "use a function and `over=` (with a product or a table for several indices)")
    recovered = _iterable_of(frame.f_locals[".0"])
    body = types.FunctionType(
        f.gi_code, frame.f_globals, "<traced>", (),
        tuple(_cell(frame.f_locals[n]) for n in f.gi_code.co_freevars))
    return (lambda i: next(body(iter([i])))), (recovered if over is None else over)


#: name -> (backend package, constructor). Each is loaded only if it is asked for:
#: importing a GPU backend costs seconds and acquires a device context, so the
#: default CPU path must never touch one.
BACKENDS = {
    "serial": (None, None),
    "cpu": ("KernelAbstractions", "KernelAbstractions.CPU()"),
    "cuda": ("CUDA", "CUDA.CUDABackend()"),
    "rocm": ("AMDGPU", "AMDGPU.ROCBackend()"),
    "oneapi": ("oneAPI", "oneAPI.oneAPIBackend()"),
    "metal": ("Metal", "Metal.MetalBackend()"),
}
_loaded = set()


def backends():
    """Accelerator backends this package knows how to construct."""
    return sorted(BACKENDS)


def _backend(spec):
    """Resolve a backend name, loading its package on first use."""
    if spec is None or not isinstance(spec, str):
        return spec                                   # already a backend object
    try:
        pkg, ctor = BACKENDS[spec]
    except KeyError:
        raise ValueError(f"unknown backend {spec!r}; available: {backends()}") from None
    if pkg is None:
        return None
    if spec not in _loaded:
        try:
            _b.seval(f"using {pkg}")
        except Exception:                             # noqa: BLE001
            raise _b.ModelError(
                f"the {spec!r} backend is not installed in this environment "
                f"(needs {pkg}). Install it with: examodels.install_backend({spec!r})"
            ) from None
        _loaded.add(spec)
    return _b.seval(ctor)


#: What a working DEVICE SOLVE needs beyond the backend package itself: the
#: solver's device side and its linear solver, installed together so the whole
#: tree resolves in ONE environment.  A piece left to the global environment
#: gets picked up through Julia's load path against a different dependency
#: tree, and the mixed stack fails incoherently at the first solve.
_BACKEND_STACKS = {
    "cuda": (("MadNLP", "2621e9c9-9eb4-46b1-8089-e8c72242dfb6"),
             ("MadNLPGPU", "d72a61cc-809d-412f-99be-fd81f4b8a598"),
             ("CUDSS", "45b445bb-4962-46a0-9369-b4df9d0f772e")),
}


def install_backend(name):
    """Install an accelerator backend into this environment (one-off; needs a
    network).  For `"cuda"` this is the whole stack a device solve needs —
    CUDA, MadNLP, MadNLPGPU and CUDSS — resolved into one environment.
    Restart Python afterwards: the environment cannot change under a running
    Julia."""
    pkg, _ = BACKENDS.get(name, (None, None))
    if pkg is None:
        raise ValueError(f"nothing to install for backend {name!r}")
    import juliapkg
    juliapkg.add(pkg, _BACKEND_UUIDS[pkg])
    for extra, uuid in _BACKEND_STACKS.get(name, ()):
        juliapkg.add(extra, uuid)
    juliapkg.resolve()


_BACKEND_UUIDS = {
    "KernelAbstractions": "63c18a36-062a-441e-b654-da1e3ab1ce7c",
    "CUDA": "052768ef-5323-5732-b1bb-66c8b64840ba",
    "AMDGPU": "21141c5a-9bdb-4563-92ae-f87d6854732e",
    "oneAPI": "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b",
    "Metal": "dde4c033-4e86-420c-a63e-0dd931031962",
}


class Core:
    """Accumulates a model, mirroring the backend's `ExaCore`.

        core = Core()
        x = core.add_variables(10, start=0.0)
        core.minimize(lambda i: x[i]**2, over=range(10))
        model = Model(core)

    `Model(core)` finishes it; `core.solve()` is shorthand for both steps.
    """

    #: set by TwoStageCore; a plain core has none
    nscen = 0

    def __new__(cls, *args, cache=None, **kwargs):
        if cls is Core:
            if cache:
                # Caching wants the model *recorded*, not built: RecordingCore
                # keeps the whole construction in Python so a cache hit never
                # boots Julia.  The CPU-only guard lives here, with the cache=
                # that implies it — the recorder itself is backend-neutral
                # (the daemon records accelerator models).
                backend = kwargs.get("backend", args[0] if args else None)
                if backend is not None and (not isinstance(backend, str)
                                            or backend not in ("serial", "cpu")):
                    raise ValueError(
                        f"a cached model compiles to a CPU shared library; "
                        f"backend {backend!r} cannot be cached — build "
                        f"without cache=")
                from ._record import RecordingCore
                return object.__new__(RecordingCore)
            from ._warm import dispatching
            if dispatching():
                # A warm session is running: record for it instead of building
                # eagerly.  Everything a record cannot do falls back to eager
                # construction or an in-process solve; see _warm.py.
                from ._warm import DispatchCore
                return object.__new__(DispatchCore)
        return object.__new__(cls)

    def __init__(self, backend=None, minimize=True, nargs=0, cache=None):
        resolved = _backend(backend)
        kw = {"backend": resolved} if resolved is not None else {}
        if not minimize:
            kw["minimize"] = False
        if nargs:
            # `ExaCore(nargs = Val(k))` returns the core followed by k
            # placeholders. Taken apart here: a Julia tuple indexed from Python
            # would be 1-based, and every index in this package is 0-based.
            from .recipe import Arg
            got = _b.guard(_b.core_with_args, nargs, concrete=_b.valtrue, **kw)
            self._core = _b.at(got, 0)
            self.args = tuple(Arg(_b.at(got, i + 1)) for i in range(nargs))
        else:
            self._core = _b.guard(_b.EM.ExaCore, concrete=_b.valtrue, **kw)
            #: placeholders, when built with `nargs=`; empty otherwise
            self.args = ()
        self._named = {}

    def _add(self, fn, *args, **kwargs):
        self._core, out = _b.guard(fn, self._core, *args, **kwargs)
        return out

    def _named_kw(self, name, kw):
        """`name=` registers the block with the backend and here."""
        if name is not None:
            kw["name"] = _b.val_symbol(str(name))
        return kw

    def _remember(self, name, handle):
        if name is not None:
            self._named.setdefault("", None)
            self._named[str(name)] = handle
        return handle

    def __getattr__(self, name):
        """`core.x` — whatever was added with `name="x"`."""
        named = self.__dict__.get("_named")
        if named and name in named:
            return named[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    # -- variables and parameters ---------------------------------------------
    def _scen(self, args):
        """Strip a leading EachScenario() marker; return (marker_or_None, rest)."""
        from .advanced import EachScenario
        if args and isinstance(args[0], EachScenario):
            return _b.each_scenario, args[1:]
        return None, args

    def add_var(self, *dims, start=0.0, lvar=None, uvar=None, tag=None, name=None):
        """A block of decision variables, one dimension per argument.

            x = core.add_var(10)        # index as x[i]
            y = core.add_var(T, N)      # index as y[t, i]

        `start`, `lvar` and `uvar` are scalars or arrays of that shape.

        >>> import examodels as exa
        >>> core = exa.Core()
        >>> x = core.add_var(3, start=1.0)
        >>> _ = core.add_obj(lambda i: (x[i] - 2.0) ** 2, over=range(3))
        >>> exa.Model(core).get_start(x)
        array([1., 1., 1.])
        """
        scen, dims = self._scen(dims)
        if len(dims) == 1 and (isinstance(dims[0], types.GeneratorType)
                               or callable(dims[0])):
            return self._defined_var(dims[0], start, lvar, uvar)
        for d in dims:
            if isinstance(d, range) and d.step != 1:
                raise TypeError(
                    f"a dimension cannot have a step ({d}); the backend defines "
                    f"lengths only for whole ranges. Declare the whole range and "
                    f"use the stepped one as an index set instead")
        from .recipe import Arg
        if len(dims) == 1 and isinstance(dims[0], Arg):
            # A placeholder dimension cannot be turned into a `range` here: its
            # value is not known yet. The 0-based range is described to the
            # backend instead, and built when the model is.
            kw = {"start": _data(start, "start")}
            if lvar is not None:
                kw["lvar"] = _data(lvar, "lvar")
            if uvar is not None:
                kw["uvar"] = _data(uvar, "uvar")
            if tag is not None:
                kw["tag"] = _b.unwrap(tag)
            self._named_kw(name, kw)
            block = Block(self._add(_b.add_var_arg, 0, dims[0]._jl, **kw), (dims[0],))
            return self._remember(name, block)
        if not dims or not all(isinstance(d, (int, range)) for d in dims):
            raise TypeError("give one integer or range per dimension, "
                            "e.g. add_var(T, N) or add_var(range(2, 11)); a "
                            "placeholder from Core(nargs=...) may be used alone")
        # The backend dimension is declared with the caller's own bounds -- an
        # integer `n` means `0:n-1`, a `range(a, b)` means `a:b-1`. Nothing is
        # shifted anywhere: an index is a number the caller chose, so it reads the
        # same whether it addresses a variable or appears in the arithmetic. The
        # backend folds the offset into the constant it already carries, so this
        # costs no extra node.
        axes = [range(d) if isinstance(d, int) else d for d in dims]
        los = [a.start for a in axes]
        his = [a.stop - 1 for a in axes]
        kw = {"start": _data(start, "start")}
        if lvar is not None:
            kw["lvar"] = _data(lvar, "lvar")
        if uvar is not None:
            kw["uvar"] = _data(uvar, "uvar")
        if tag is not None:
            kw["tag"] = _b.unwrap(tag)
        self._named_kw(name, kw)
        if scen is None:
            return self._remember(
                name, Block(self._add(_b.add_var_dims, los, his, **kw), tuple(axes)))
        # A per-scenario declaration is replicated by the backend, so the block
        # really holds nscen times what was asked for. Index it flat.
        per = 1
        for a in axes:
            per *= len(a)
        blk = self._add(_b.add_var_scen, scen, los, his, **kw)
        return self._remember(name, Block(blk, (range(per * self.nscen),)))

    def _defined_var(self, f, start, lvar, uvar):
        """`add_var(expr for i in over)` — variables tied to those expressions.

        The same thing the backend does: make the block, then constrain each new
        variable to equal its expression.
        """
        f, over = _as_function(f, None)
        if not isinstance(over, range):
            raise TypeError("variables defined by expressions need a range index set")
        block = self.add_var(over, start=start, lvar=lvar, uvar=uvar)
        self.add_con(lambda i: block[i] - f(i), over=over)
        return block

    def add_par(self, values, name=None):
        """A block of parameters — fixed values usable in expressions, changeable
        afterwards with `Model.set_parameters` without rebuilding.

        >>> import examodels as exa
        >>> core = exa.Core()
        >>> x = core.add_var(2)
        >>> p = core.add_par([3.0, 4.0])
        >>> _ = core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(2))
        >>> m = exa.Model(core)
        >>> m.parameters(p)
        array([3., 4.])
        >>> _ = m.set_parameters(p, [5.0, 6.0])
        >>> m.parameters(p)
        array([5., 6.])
        """
        arr = np.ascontiguousarray(values, dtype=np.float64).ravel()
        kw = self._named_kw(name, {})
        return self._remember(name, Block(
            self._add(_b.add_par_range, 0, arr.size - 1, arr, **kw),
            arr.size, "parameter"))

    # -- subexpressions -------------------------------------------------------
    def add_expr(self, f, over=None, name=None):
        """Name a reusable subexpression. Inlined at each use, so it adds no
        variables and no constraints. `over` may be a tuple for `s[t, i]`."""
        f, over = _as_function(f, over)
        return self._remember(name, Expression(f, over))

    # -- objective and constraints --------------------------------------------
    def add_obj(self, f, over=None, name=None):
        """Add `sum(f(i) for i in over)` to the objective.

        Write it either way:  `add_obj(x[i]**2 for i in range(n))`
        or                    `add_obj(lambda i: x[i]**2, over=range(n))`.
        """
        from .advanced import Oracle
        if isinstance(f, Oracle):
            self._core = _b.guard(_b.register_obj_oracle, self._core, f._jl)
            return self
        f, over = _as_function(f, over)
        over = range(1) if over is None else over
        args, _ = _index_set(over)
        fn = _b.obj_range if len(args) == 2 else _b.obj_iter
        kw = self._named_kw(name, {})
        self._core, obj = _b.guard(fn, self._core, _b.unwrap(_node(f, over)), *args, **kw)
        self._remember(name, obj)
        return self

    def add_con(self, *args, over=None, lcon=0.0, ucon=0.0, name=None):
        """Add constraints, or add terms to constraints already added.

            core.add_con(x[i] + x[i+1] for i in range(n - 1))
            core.add_con(lambda i: x[i] + x[i+1], over=range(n - 1))

        `add_con(f, over)` creates one row per index, `lcon <= f(i) <= ucon`, and
        returns a handle.

        `add_con(handle, f, over)` adds terms into those rows: `f(row)` returns
        `(row_index, expression)` and the expression is added to that row. This is
        how a balance is assembled from many sources -- every line and every
        generator at a bus -- without materialising a sum per row. It mirrors the
        backend's `add_con` / `add_con!` pair.
        """
        from .advanced import Oracle
        if args and isinstance(args[0], Oracle):
            self._core = _b.guard(_b.register_con_oracle, self._core, args[0]._jl)
            return args[0]
        scen, args = self._scen(args)
        if scen is not None:
            kwargs = dict(lcon=_data(lcon, "lcon"), ucon=_data(ucon, "ucon"))
            f, *rest = args
            f, over = _as_function(f, rest[0] if rest else over)
            iters, _ = _index_set(over)
            gen = _b.gen_range(_b.unwrap(_node(f, over)), *iters) if len(iters) == 2 \
                else _b.gen_iter(_b.unwrap(_node(f, over)), *iters)
            con = self._add(_b.add_con_scen, scen, gen, **kwargs)
            start = over.start if isinstance(over, range) else 0
            return Constraint(con, len(over), 1 - start)
        if args and all(isinstance(a, (int, range)) for a in args):
            # dimensions only: an empty block, to be filled with add_con(handle, ...)
            axes = [range(d) if isinstance(d, int) else d for d in args]
            con = self._add(_b.con_dims, [a.start for a in axes], [a.stop - 1 for a in axes],
                            lcon=_data(lcon, "lcon"), ucon=_data(ucon, "ucon"))
            n = 1
            for a in axes:
                n *= len(a)
            return Constraint(con, n, 1 - axes[0].start)
        if args and isinstance(args[0], Constraint):
            constraint, f, *rest = args               # add_con(handle, f, over)
            return self._augment(constraint, f, rest[0] if rest else over)
        f, *rest = args
        f, recovered = _as_function(f, rest[0] if rest else over)
        over = recovered
        args, _ = _index_set(over)
        # The backend's `add_con` has no low-level expression form (`add_obj` does),
        # so the expression is wrapped in a constant generator — the same approach
        # its own MathOptInterface backend uses.
        node = _b.unwrap(_node(f, over))
        gen = _b.gen_range(node, *args) if len(args) == 2 else _b.gen_iter(node, *args)
        con = self._add(_b.EM.add_con, gen, **self._named_kw(name, dict(
            lcon=_data(lcon, "lcon"), ucon=_data(ucon, "ucon"))))
        # Rows are addressed by the index that produced them, but the backend
        # numbers a constraint block's rows from 1 whatever the index set was (it
        # takes their count, not their labels). So the label has to be mapped back
        # to a position: an index set starting at `a` puts label `a` in row 1.
        from .recipe import SRange
        start = over.start if isinstance(over, range) else 0
        # An `srange` has placeholder bounds, so the row count is not known here.
        # `nrows=None` says "ask the model" rather than inventing a number; the
        # backend has the real count once the model is built.
        nrows = None if isinstance(over, SRange) else len(over)
        return self._remember(name, Constraint(con, nrows, 1 - start))

    def _augment(self, constraint, f, over):
        f, over = _as_function(f, over)
        idx, expr = f(Node(_b.EM.DataSource()))
        if constraint.row_offset:
            idx = idx + constraint.row_offset
        args, _ = _index_set(over)
        fn = _b.aug_range if len(args) == 2 else _b.aug_iter
        self._add(_b.EM.add_con_b, _b.unwrap(constraint),
                  fn(_b.unwrap(idx), _b.unwrap(expr), *args))
        return constraint

    # -- finalize -------------------------------------------------------------
    def build(self):
        """Finish this core, returning a `Model`. Same as `Model(core)`."""
        from .model import Model
        return Model(self)

    def solve(self, solver=None, **options):
        return self.build().solve(solver=solver, **options)

    def __repr__(self):
        return "<Core>"
