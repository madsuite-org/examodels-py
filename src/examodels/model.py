"""The finished model and the solve result."""
import numpy as np

from . import _bridge as _b

__all__ = ["Model", "Solution"]


def _py(v):
    """Backend value -> a plain Python/numpy value."""
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return np.array(v, dtype=np.float64)
    except Exception:                                        # noqa: BLE001
        return v


class Model:
    """A finished model, built from a `Core` — the backend's `ExaCore` -> `ExaModel`.

    Metadata (`nvar`, `ncon`, `nnzj`, `nnzh`, `x0`, `lvar`, ...) is read straight
    off the backend rather than mirrored here, so nothing needs updating when the
    backend gains a field.
    """

    __slots__ = ("_jl",)

    def __init__(self, core):
        from .core import Core
        self._jl = _b.guard(_b.EM.ExaModel, core._core) if isinstance(core, Core) else core

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
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
        return float(_b.guard(_b.EM.obj, self._jl, np.asarray(x, dtype=np.float64)))

    def gradient(self, x):
        return self._inplace("grad!", self.nvar, x)

    def constraints(self, x):
        return self._inplace("cons!", self.ncon, x)

    def _inplace(self, fn, n, x):
        out = np.zeros(n)
        _b.guard(_b.seval(f"(m, x, o) -> NLPModels.{fn}(m, x, o)"),
                 self._jl, np.asarray(x, dtype=np.float64), out)
        return out

    # -- parameters -----------------------------------------------------------
    def parameters(self, block):
        """Current values of a parameter block."""
        return np.array(_b.guard(_b.EM.get_value, self._jl, block._jl), dtype=np.float64)

    def set_parameters(self, block, values):
        """Change a parameter block's values in place; the model is reused as is."""
        _b.guard(_b.EM.set_value_b, self._jl, block._jl,
                 np.asarray(values, dtype=np.float64).ravel())
        return self

    def solve(self, solver="ipopt", **options):
        from .solve import solve
        return solve(self, solver=solver, **options)

    def __repr__(self):
        return f"<Model nvar={self.nvar} ncon={self.ncon} nnzj={self.nnzj} nnzh={self.nnzh}>"


class Solution:
    """Result of a solve. Fields come from the solver's own result object.

    `sol[block]` gives that block's values.
    """

    _ALIASES = {"x": "solution", "y": "multipliers", "iterations": "iter",
                "elapsed": "elapsed_time"}

    __slots__ = ("_raw",)

    def __init__(self, raw):
        self._raw = raw

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

    def __getitem__(self, block):
        return np.array(_b.guard(_b.EM.solution, self._raw, block._jl), dtype=np.float64)

    def __repr__(self):
        return (f"<Solution status={self.status!r} objective={self.objective:.6g} "
                f"iterations={self.iterations}>")
