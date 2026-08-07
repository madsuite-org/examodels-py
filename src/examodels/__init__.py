"""examodels — Python interface to ExaModels.jl.

    import examodels as exa

    core = exa.Core()
    x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])
    core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2, over=range(1, N))

    model = exa.Model(core)
    sol = model.solve()
    print(sol.objective, sol[x])

Expressions are written as ordinary Python functions of an index. Each one is
traced **once** with a symbolic index, producing a single structured expression
that is evaluated over the whole index set — on CPU threads or on a GPU.

Because the function is traced rather than looped, it must not branch on the
index; anything index-dependent belongs in the data (`start`, `lower`, `upper`,
or the index set itself).
"""
from .core import Core, backends, install_backend, trace
from .node import (Block, Constant, Constraint, Expression, Node, Product,
                   Records, TupleNode)
from .node import prod, sum


def product(*axes):
    """A rectangular index set: `product(range(T), range(N))`."""
    return Product(*axes)
from .model import Model, Solution
from .solve import available_solvers, install_solver, solve
from ._bridge import ModelError

__version__ = "0.1.0"

__all__ = [
    "Core", "Model", "Solution", "Block", "Constraint", "Expression", "Records",
    "Product", "product", "Node", "Constant", "sum", "prod",
    "trace", "solve", "available_solvers", "install_solver",
    "backends", "install_backend", "ModelError", "__version__",
]


_SUBMODULES = frozenset({"ops", "core", "node", "model", "solve", "testing", "_bridge"})


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
