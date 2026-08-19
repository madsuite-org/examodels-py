"""The finished model and the solve result."""
import numpy as np

from . import _bridge as _b

__all__ = ["Model", "Solution"]


def _py(v):
    """Backend value -> a plain Python/numpy value, wherever it was stored."""
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return np.array(_b.tohost(v), dtype=np.float64)
    except Exception:                                        # noqa: BLE001
        return v


class Model:
    """A finished model, built from a `Core` — the backend's `ExaCore` -> `ExaModel`.

    Metadata (`nvar`, `ncon`, `nnzj`, `nnzh`, `x0`, `lvar`, ...) is read straight
    off the backend rather than mirrored here, so nothing needs updating when the
    backend gains a field.
    """

    def __new__(cls, core=None, *args, **kwargs):
        from ._record import RecordingCore
        if cls is Model and isinstance(core, RecordingCore):
            if core.cache:
                # The cache lookup: a hit is a CachedModel, and no Julia has
                # entered the process.  For a recipe, `args` instantiate the
                # loaded library.  (Python then re-invokes __init__ on the
                # returned instance; CachedModel.__init__ absorbs that.)
                from ._cache import attach
                hit = attach(core, args)
                if hit is not None:
                    return hit
            # A cache hit beats the daemon (a local library load beats IPC);
            # on a miss — or for a record made only because a daemon was up —
            # the warm session is next, and MISS below is the catch-all.
            from ._warm import try_daemon
            served = try_daemon(core, args)
            if served is not None:
                return served
        return object.__new__(cls)

    def __init__(self, core, *args):
        """`Model(core)`, or `Model(core, *values)` for a core built with
        `nargs=` — one value per placeholder, in the order they were returned."""
        from ._record import RecordingCore
        from .core import Core
        from .recipe import _unwrap
        if isinstance(core, RecordingCore):
            if core.cache:
                # A miss (a hit never reaches here): replay through the eager
                # path, compile and store the entry synchronously, and carry
                # on as an ordinary eager model for this run — instantiated
                # with `args` below, for a recipe.
                from ._cache import materialize
                core = materialize(core, args)
            else:
                # A daemon record whose daemon went away: plain replay, no
                # cache to fill.  Grafting inside replay keeps the caller's
                # handles working.
                core = core.replay()
        if isinstance(core, Core):
            if args:
                self._jl = _b.guard(
                    _b.model_with_args, core._core, *[_unwrap(a) for a in args])
            else:
                self._jl = _b.guard(_b.EM.ExaModel, core._core)
            self._named = dict(getattr(core, "_named", {}))
        else:
            self._jl, self._named = core, {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        named = self.__dict__.get("_named")
        if named and name in named:                  # `model.x` -> the named block
            return named[name]
        try:
            v = getattr(self._jl.meta, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from None
        return int(v) if isinstance(v, int) else _py(v)

    def __dir__(self):
        return sorted({*super().__dir__(), *(str(f) for f in _b.jl.fieldnames(
            _b.seval("typeof")(self._jl.meta)))})

    # -- evaluation -----------------------------------------------------------
    def objective(self, x):
        return float(_b.guard(_b.EM.obj, self._jl, self._x(x)))

    def _x(self, x):
        """Primal vector, placed wherever this model's arrays live.

        A device array is used where it already is, rather than copied to the host
        and back.
        """
        if hasattr(x, "__cuda_array_interface__"):
            from .advanced import from_cupy
            return from_cupy(x)
        return _b.upload(self._jl, np.ascontiguousarray(x, dtype=np.float64))

    def gradient(self, x):
        return self._inplace("grad!", self.nvar, x)

    def constraints(self, x):
        return self._inplace("cons!", self.ncon, x)

    def violation(self, x):
        """Largest constraint violation at `x`.

        Not `max|c(x)|`: for a one-sided constraint the value itself is unbounded
        and says nothing about feasibility -- only how far outside its own bounds
        each row sits does.
        """
        c = self.constraints(x)
        return float(np.maximum(0.0, np.maximum(self.lcon - c, c - self.ucon)).max())

    def _inplace(self, fn, n, x):
        out = _b.like(self._jl, n)
        _b.guard(_b.seval(f"(m, x, o) -> NLPModels.{fn}(m, x, o)"), self._jl, self._x(x), out)
        return np.array(_b.tohost(out), dtype=np.float64)

    # -- parameters -----------------------------------------------------------
    def parameters(self, block):
        """Current values of a parameter block."""
        return np.array(_b.guard(_b.EM.get_value, self._jl, block._jl), dtype=np.float64)

    def set_parameters(self, block, values):
        """Change a parameter block's values in place; the model is reused as is."""
        # Place the values where the model's arrays live, like every other
        # setter: writing a host array into device memory falls into a scalar
        # path the backend disallows.
        _b.guard(_b.EM.set_value_b, self._jl, block._jl,
                 _b.upload(self._jl, np.asarray(values, dtype=np.float64).ravel()))
        return self

    def solve(self, solver=None, **options):
        from .solve import solve
        return solve(self, solver=solver, **options)

    def __repr__(self):
        return f"<Model nvar={self.nvar} ncon={self.ncon} nnzj={self.nnzj} nnzh={self.nnzh}>"


def _refuse_placeholder_block(handle):
    """Raise the readable error when `handle` is a placeholder-sized block."""
    from .recipe import Arg
    if any(isinstance(a, Arg) for a in getattr(handle, "_axes", ())):
        raise _b.ModelError(
            "a placeholder-sized block cannot be addressed on a built model -- "
            "its handle describes sizes that were only supplied at instantiation "
            "(ExaModels.jl has the same limit). Read the whole vector instead: "
            "`sol.x`, `model.x0`, `model.lvar`, ... , and slice it."
        ) from None


#: What can be read back and changed on a built model, without rebuilding it:
#: parameter values, the starting point, and variable and constraint bounds.
#: Named as the backend names them.
ACCESSORS = ("value", "start", "lvar", "uvar", "lcon", "ucon")


def _shape_of(handle):
    shape = getattr(handle, "shape", None)
    return shape if shape and len(shape) > 1 else None


def _accessors(name):
    def get(self, handle):
        try:
            flat = np.array(_b.tohost(_b.guard(getattr(_b.EM, f"get_{name}"), self._jl,
                                               _b.unwrap(handle))), dtype=np.float64)
        except Exception:
            _refuse_placeholder_block(handle)
            raise
        shape = getattr(handle, "shape", None)
        # The backend stores a block column-major; give it back in the shape it was
        # given in, so a value read out sits where the caller put it.
        return flat.reshape(shape, order="F") if shape else flat

    def set(self, handle, values):
        if hasattr(values, "__cuda_array_interface__"):
            from .advanced import from_cupy
            _b.guard(getattr(_b.EM, f"set_{name}_b"), self._jl, _b.unwrap(handle),
                     from_cupy(values))
            return self
        a = np.asarray(values, dtype=np.float64)
        shape = _shape_of(handle)
        flat = np.ascontiguousarray(a.reshape(shape, order="C").ravel(order="F")) \
            if shape and a.shape == shape else np.ascontiguousarray(a).ravel()
        # Place the values where the model's arrays live: writing a host array
        # into device memory falls into a scalar path the backend disallows.
        try:
            _b.guard(getattr(_b.EM, f"set_{name}_b"), self._jl, _b.unwrap(handle),
                     _b.upload(self._jl, flat))
        except Exception:
            _refuse_placeholder_block(handle)
            raise
        return self

    get.__name__, set.__name__ = f"get_{name}", f"set_{name}"
    get.__doc__ = f"Read the {name} of a variable, parameter or constraint block."
    set.__doc__ = f"Change the {name} of a block in place; the model is reused as is."
    return get, set


for _n in ACCESSORS:
    _g, _s = _accessors(_n)
    setattr(Model, _g.__name__, _g)
    setattr(Model, _s.__name__, _s)


class Solution:
    """Result of a solve. Fields come from the solver's own result object.

    `sol[block]` gives that block's values.
    """

    _ALIASES = {"x": "solution", "y": "multipliers", "iterations": "iter"}
    #: `multipliers` is a per-block accessor here, as it is in the backend; the
    #: whole dual vector stays available as `.y`

    __slots__ = ("_raw", "elapsed")

    def __init__(self, raw, elapsed=float("nan")):
        self._raw = raw
        #: wall-clock seconds, measured here: solvers do not agree on reporting it
        self.elapsed = elapsed

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            v = getattr(self._raw, self._ALIASES.get(name, name))
        except AttributeError:
            raise AttributeError(f"solutions have no attribute {name!r}") from None
        return str(v) if name == "status" else (int(v) if isinstance(v, int) else _py(v))

    #: solvers report success in their own vocabulary
    _SUCCESS = frozenset({"first_order", "acceptable", "SOLVE_SUCCEEDED",
                          "SOLVED_TO_ACCEPTABLE_LEVEL"})

    @property
    def success(self):
        return self.status in self._SUCCESS

    def multipliers(self, constraint):
        """Duals of a constraint block."""
        return self._block(_b.EM.multipliers, constraint)

    def multipliers_L(self, block):
        """Duals of a variable block's lower bounds."""
        return self._block(_b.EM.multipliers_L, block)

    def multipliers_U(self, block):
        """Duals of a variable block's upper bounds."""
        return self._block(_b.EM.multipliers_U, block)

    def _block(self, fn, handle):
        try:
            got = np.array(_b.tohost(_b.guard(fn, self._raw, _b.unwrap(handle))),
                           dtype=np.float64)
        except Exception:
            # The backend (and ExaModels.jl itself, verified) cannot index a
            # result by a placeholder-sized block: the handle describes sizes
            # that were only supplied at instantiation. Name the way that works.
            _refuse_placeholder_block(handle)
            raise
        shape = getattr(handle, "shape", None)
        return got.reshape(shape, order="F") if shape else got

    def __getitem__(self, block):
        """`sol[x]` — the solution values of a variable block."""
        return self._block(_b.EM.solution, block)

    def __repr__(self):
        return (f"<Solution status={self.status!r} objective={self.objective:.6g} "
                f"iterations={self.iterations}>")
