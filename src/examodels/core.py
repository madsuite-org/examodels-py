"""Model construction: variables, objectives, constraints."""
import numpy as np

from . import _bridge as _b
from .node import Node, Variable

__all__ = ["Model", "trace"]


def trace(f):
    """Call `f` ONCE with a symbolic index and return the expression it builds.

    The loop never runs at build time — one traced expression describes every row —
    so `f` must not branch on the index. Put anything index-dependent in the data
    (`start`, `lower`, `upper`, or the index set) instead.
    """
    return f(Node(_b.EM.DataSource()))


def _index_set(over):
    """Python index set -> backend index set.

    The backend counts from 1, this package counts from 0, so the set is shifted by
    one here.  Symbolic indexing (`x[i]`) is then left alone, which keeps the traced
    expression byte-identical to the one the backend builds for itself.
    """
    if isinstance(over, range):
        if over.step != 1:
            raise ValueError("index sets must have step 1")
        return _b.mkrange(over.start + 1, over.stop), len(over)
    n = len(over) if hasattr(over, "__len__") else None
    return _b.unwrap(over), n


def _data(v, n, what):
    if callable(v):
        raise TypeError(
            f"{what} must be a number or an array, not a function — it is data, "
            f"evaluated once per index, so build it with numpy or a comprehension"
        )
    if np.isscalar(v):
        return float(v)
    a = np.ascontiguousarray(v, dtype=np.float64)
    if n is not None and a.size != n:
        raise ValueError(f"{what} has length {a.size}, expected {n}")
    return a


class Model:
    """An optimization model, built up variable by variable.

        m = Model()
        x = m.add_variables(10, start=0.0)
        m.minimize(lambda i: x[i]**2, over=range(10))
        sol = m.solve()
    """

    def __init__(self, backend=None):
        kwargs = {"backend": backend} if backend is not None else {}
        self._core = _b.guard(_b.EM.ExaCore, concrete=_b.valtrue, **kwargs)
        self._vars = []

    # -- variables ------------------------------------------------------------
    def add_variables(self, n, start=0.0, lower=None, upper=None):
        """Add a block of `n` decision variables; returns a handle you index with `[i]`."""
        kw = {"start": _data(start, n, "start")}
        if lower is not None:
            kw["lvar"] = _data(lower, n, "lower")
        if upper is not None:
            kw["uvar"] = _data(upper, n, "upper")
        self._core, x = _b.guard(_b.EM.add_var, self._core, n, **kw)
        v = Variable(x, n)
        self._vars.append(v)
        return v

    # -- objective ------------------------------------------------------------
    def minimize(self, f, over=range(1)):
        """Add `sum(f(i) for i in over)` to the objective. `f` is traced once."""
        node = f if isinstance(f, Node) else trace(f)
        iters, _ = _index_set(over)
        self._core, _ = _b.guard(_b.EM.add_obj, self._core, _b.unwrap(node), iters)
        return self

    # -- constraints ----------------------------------------------------------
    def constrain(self, f, over, lower=0.0, upper=0.0):
        """Add one row per element of `over`, constrained to `lower <= f(i) <= upper`."""
        node = f if isinstance(f, Node) else trace(f)
        iters, n = _index_set(over)
        # ExaModels' `add_con` has no low-level expression form (`add_obj` does), so the
        # expression is wrapped in a constant generator — the same approach the
        # MathOptInterface backend uses.
        gen = _b.mkgen(_b.unwrap(node), iters)
        self._core, _ = _b.guard(
            _b.EM.add_con, self._core, gen,
            lcon=_data(lower, n, "lower"), ucon=_data(upper, n, "upper"),
        )
        return self

    # -- finalize -------------------------------------------------------------
    def build(self):
        """Finish the model, returning a `Problem` you can evaluate or solve."""
        from .problem import Problem
        return Problem(_b.guard(_b.EM.ExaModel, self._core), self._vars)

    def solve(self, solver="ipopt", **options):
        """Build and solve in one step."""
        return self.build().solve(solver=solver, **options)

    def __repr__(self):
        return f"<Model building, {len(self._vars)} variable block(s)>"
