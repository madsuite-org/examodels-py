"""Model construction: variables, parameters, subexpressions, objective, constraints."""
import numpy as np

from . import _bridge as _b
from .node import Block, Constraint, Expression, Node, Records

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
    """Describe an index set for the backend, shifted from 0-based to 1-based.

    Returns *arguments*, not an iterator: a Julia range that round-trips through
    Python comes back as a `StepRange`, which several of the backend's size and
    dispatch paths reject, so the iterator is constructed on the backend side.
    Symbolic indexing is left unshifted, which keeps a traced expression identical
    to the one the backend builds for itself.
    """
    if isinstance(over, range):
        if over.step != 1:
            raise ValueError("index sets must have step 1")
        return (over.start + 1, over.stop), len(over)
    return (_b.unwrap(over),), len(over)


def _data(v, what):
    if callable(v):
        raise TypeError(
            f"{what} is data, not an expression: pass a number or an array. It is "
            f"evaluated once per index, so build it with a list comprehension or numpy.")
    return float(v) if np.isscalar(v) else np.ascontiguousarray(v, dtype=np.float64)


def _node(f):
    return f if isinstance(f, Node) else trace(f)


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


def install_backend(name):
    """Install an accelerator backend into this environment (one-off; needs a network)."""
    pkg, _ = BACKENDS.get(name, (None, None))
    if pkg is None:
        raise ValueError(f"nothing to install for backend {name!r}")
    import juliapkg
    juliapkg.add(pkg, _BACKEND_UUIDS[pkg])
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

    def __init__(self, backend=None):
        resolved = _backend(backend)
        kw = {"backend": resolved} if resolved is not None else {}
        self._core = _b.guard(_b.EM.ExaCore, concrete=_b.valtrue, **kw)

    def _add(self, fn, *args, **kwargs):
        self._core, out = _b.guard(fn, self._core, *args, **kwargs)
        return out

    # -- variables and parameters ---------------------------------------------
    def add_var(self, n, start=0.0, lower=None, upper=None):
        """A block of `n` decision variables; index it with `[i]`."""
        kw = {"start": _data(start, "start")}
        if lower is not None:
            kw["lvar"] = _data(lower, "lower")
        if upper is not None:
            kw["uvar"] = _data(upper, "upper")
        return Block(self._add(_b.EM.add_var, n, **kw), n)

    def add_par(self, values):
        """A block of parameters — fixed values usable in expressions, changeable
        afterwards with `Model.set_parameters` without rebuilding."""
        arr = np.ascontiguousarray(values, dtype=np.float64).ravel()
        return Block(self._add(_b.EM.add_par, arr), arr.size, "parameter")

    # -- subexpressions -------------------------------------------------------
    def add_expr(self, f, over):
        """Name a reusable subexpression. Inlined at each use, so it adds no
        variables and no constraints. `over` may be a tuple for `s[t, i]`."""
        return Expression(f, over)

    # -- objective and constraints --------------------------------------------
    def add_obj(self, f, over=range(1)):
        """Add `sum(f(i) for i in over)` to the objective. `f` is traced once."""
        args, _ = _index_set(over)
        fn = _b.obj_range if len(args) == 2 else _b.obj_iter
        self._core, _ = _b.guard(fn, self._core, _b.unwrap(_node(f)), *args)
        return self

    def add_con(self, *args, over=None, lower=0.0, upper=0.0):
        """Add constraints, or add terms to constraints already added.

        `add_con(f, over)` creates one row per index, `lower <= f(i) <= upper`, and
        returns a handle.

        `add_con(handle, f, over)` adds terms into those rows: `f(row)` returns
        `(row_index, expression)` and the expression is added to that row. This is
        how a balance is assembled from many sources -- every line and every
        generator at a bus -- without materialising a sum per row. It mirrors the
        backend's `add_con` / `add_con!` pair.
        """
        if args and isinstance(args[0], Constraint):
            constraint, f, *rest = args               # add_con(handle, f, over)
            return self._augment(constraint, f, rest[0] if rest else over)
        f, *rest = args
        over = rest[0] if rest else over
        args, _ = _index_set(over)
        # The backend's `add_con` has no low-level expression form (`add_obj` does),
        # so the expression is wrapped in a constant generator — the same approach
        # its own MathOptInterface backend uses.
        gen = _b.gen_range(_b.unwrap(_node(f)), *args) if len(args) == 2 else \
            _b.gen_iter(_b.unwrap(_node(f)), *args)
        con = self._add(_b.EM.add_con, gen,
                        lcon=_data(lower, "lower"), ucon=_data(upper, "upper"))
        return Constraint(con, len(over))

    def _augment(self, constraint, f, over):
        idx, expr = f(Node(_b.EM.DataSource()))
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

    def solve(self, solver="ipopt", **options):
        return self.build().solve(solver=solver, **options)

    def __repr__(self):
        return "<Core>"
