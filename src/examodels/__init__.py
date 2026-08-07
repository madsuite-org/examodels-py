"""examodels — Python interface to ExaModels.jl.

    import examodels as exa

    m = exa.Model()
    x = m.add_variables(N, start=[-1.2 if i % 2 else 1.0 for i in range(N)])
    m.minimize(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2, over=range(2, N+1))
    sol = m.solve()
    print(sol.objective, sol[x])

Expressions are written as ordinary Python functions of an index. Each one is
traced **once** with a symbolic index, producing a single structured expression
that is evaluated over the whole index set — on CPU threads or on a GPU.

Because the function is traced rather than looped, it must not branch on the
index; anything index-dependent belongs in the data (`start`, `lower`, `upper`,
or the index set itself).
"""
from .core import Model, trace
from .node import Constant, Node, Variable
from .problem import Problem, Solution
from .solve import available_solvers, install_solver, solve
from ._bridge import ModelError

__version__ = "0.1.0"

__all__ = [
    "Model", "Problem", "Solution", "Variable", "Node", "Constant",
    "trace", "solve", "available_solvers", "install_solver", "ModelError", "__version__",
]


_SUBMODULES = frozenset({"ops", "core", "node", "problem", "solve", "testing", "_bridge"})


def __getattr__(name):
    """Expose ExaModels' registered math functions (sin, exp, log, ...) lazily.

    They are generated from the operator list ExaModels itself registers, so this
    package never carries its own copy of that list.
    """
    if name.startswith("__") or name in _SUBMODULES:
        # Never re-enter for a submodule: `from . import ops` consults this hook while
        # the submodule is still being imported, which would recurse forever.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    ops = importlib.import_module(".ops", __name__)
    if hasattr(ops, name):
        fn = getattr(ops, name)
        globals()[name] = fn
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    import importlib
    ops = importlib.import_module(".ops", __name__)
    return sorted(set(__all__) | set(ops.__all__))
