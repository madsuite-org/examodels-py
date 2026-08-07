"""The built model and the solution — plain Python objects over numpy arrays."""
import numpy as np

from . import _bridge as _b

__all__ = ["Problem", "Solution"]


def _np(v):
    """Backend vector -> a numpy array the caller owns."""
    return np.asarray(memoryview(v)) if isinstance(v, memoryview) else np.array(v, dtype=np.float64)


class Problem:
    """A finished optimization model. Sizes and evaluations are plain Python/numpy."""

    __slots__ = ("_jl", "_vars")

    def __init__(self, jlobj, variables=()):
        self._jl = jlobj
        self._vars = tuple(variables)

    @property
    def nvar(self):
        return int(self._jl.meta.nvar)

    @property
    def ncon(self):
        return int(self._jl.meta.ncon)

    @property
    def nnzj(self):
        return int(self._jl.meta.nnzj)

    @property
    def nnzh(self):
        return int(self._jl.meta.nnzh)

    @property
    def x0(self):
        return _np(self._jl.meta.x0)

    def objective(self, x):
        """Objective value at `x`."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        return float(_b.guard(_b.EM.obj, self._jl, x))

    def gradient(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        g = np.zeros(self.nvar)
        _b.guard(_b.seval("(m,x,g) -> (NLPModels.grad!(m,x,g); g)"), self._jl, x, g)
        return g

    def constraints(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        c = np.zeros(self.ncon)
        _b.guard(_b.seval("(m,x,c) -> (NLPModels.cons!(m,x,c); c)"), self._jl, x, c)
        return c

    def solve(self, solver="ipopt", **options):
        from .solve import solve
        return solve(self, solver=solver, **options)

    def __repr__(self):
        return f"<Problem nvar={self.nvar} ncon={self.ncon} nnzj={self.nnzj} nnzh={self.nnzh}>"


class Solution:
    """Result of a solve. Index it with a variable to get that block's values."""

    __slots__ = ("status", "objective", "iterations", "x", "y", "elapsed", "_model", "_raw")

    def __init__(self, *, status, objective, iterations, x, y, elapsed, model, raw):
        self.status = status
        self.objective = objective
        self.iterations = iterations
        self.x = x
        self.y = y
        self.elapsed = elapsed
        self._model = model
        self._raw = raw

    @property
    def success(self):
        return self.status in ("first_order", "acceptable")

    def __getitem__(self, var):
        """`sol[x]` -> the values of variable block `x` as a numpy array."""
        return _np(_b.guard(_b.EM.solution, self._raw, _b.unwrap(var)))

    def __repr__(self):
        return (f"<Solution status={self.status!r} objective={self.objective:.6g} "
                f"iterations={self.iterations}>")
