"""Solver front end. Solvers are loaded on demand and named in Python terms."""
import time

import numpy as np

from . import _bridge as _b
from .problem import Solution

__all__ = ["solve", "available_solvers", "install_solver"]

_SOLVERS = {
    # name -> (backend package, uuid, entry point)
    "ipopt": ("NLPModelsIpopt", "f4238b75-b362-5c4c-b852-0801c9a21d71", "ipopt"),
    "madnlp": ("MadNLP", "2621e9c9-9eb4-46b1-8089-e8c72242dfb6", "madnlp"),
}

_loaded = set()


def available_solvers():
    return sorted(_SOLVERS)


def install_solver(name):
    """Install a solver backend into this environment (one-off; needs a network)."""
    if name not in _SOLVERS:
        raise ValueError(f"unknown solver {name!r}; available: {available_solvers()}")
    pkg, uuid, _ = _SOLVERS[name]
    import juliapkg
    juliapkg.add(pkg, uuid)
    juliapkg.resolve()


def _load(name):
    if name in _loaded:
        return
    pkg, _uuid, _ = _SOLVERS[name]
    try:
        _b.seval(f"using {pkg}")
    except Exception:                                        # noqa: BLE001
        raise _b.ModelError(
            f"the {name!r} solver is not installed in this environment. "
            f"Install it with: examodels.install_solver({name!r})"
        ) from None
    _loaded.add(name)


def solve(model, solver="ipopt", print_level=0, **options):
    """Solve `model` and return a `Solution`.

    Extra keyword arguments are passed through to the solver.
    """
    if solver not in _SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; available: {available_solvers()}")
    _load(solver)
    _pkg, _uuid, fname = _SOLVERS[solver]
    fn = getattr(_b.jl, fname)

    if solver == "ipopt":
        options.setdefault("print_level", print_level)
        if not print_level:
            options.setdefault("sb", "yes")   # Ipopt prints its banner even at level 0

    t0 = time.perf_counter()
    raw = _b.guard(fn, model._jl, **options)
    elapsed = time.perf_counter() - t0

    return Solution(
        status=str(raw.status),
        objective=float(raw.objective),
        iterations=int(raw.iter),
        x=np.array(raw.solution, dtype=np.float64),
        y=np.array(raw.multipliers, dtype=np.float64),
        elapsed=elapsed,
        model=model,
        raw=raw,
    )
