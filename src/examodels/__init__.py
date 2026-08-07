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


#: The backend spells these as functions taking the core (or the model, or the
#: result) first, and this package spells them as methods. Both are available and
#: are the same call: the functions below simply forward, so there is one
#: implementation and no chance of the two drifting.
_CORE_FUNCTIONS = ("add_var", "add_par", "add_obj", "add_con", "add_expr")
_MODEL_FUNCTIONS = ("get_value", "set_value", "get_start", "set_start",
                    "get_lvar", "set_lvar", "get_uvar", "set_uvar",
                    "get_lcon", "set_lcon", "get_ucon", "set_ucon",
                    "objective", "gradient", "constraints", "violation", "solve")
_RESULT_FUNCTIONS = ("multipliers", "multipliers_L", "multipliers_U")


def _forward(name, first):
    def fn(obj, *args, **kwargs):
        return getattr(obj, name)(*args, **kwargs)

    fn.__name__ = name
    fn.__doc__ = (f"`{name}({first}, ...)` — the same call as `{first}.{name}(...)`, "
                  f"in the argument order the backend uses.")
    return fn


for _n in _CORE_FUNCTIONS:
    globals()[_n] = _forward(_n, "core")
for _n in _MODEL_FUNCTIONS:
    globals()[_n] = _forward(_n, "model")
for _n in _RESULT_FUNCTIONS:
    globals()[_n] = _forward(_n, "result")


def solution(result, block):
    """`solution(result, x)` — the same as `result[x]`."""
    return result[block]
from .model import Model, Solution
from .solve import available_solvers, install_solver, solve
from ._bridge import ModelError
from .advanced import (as_cupy, from_cupy, CompressedNLPModel, EachScenario, FirstStageConstraintTag,
                       OracleEvaluator, ScalarNonlinearOracle, VectorNonlinearOracle,
                       add_eval, embed_oracle, has_matfree_jac, has_matfree_hess,
                       FirstStageTag, SecondStageConstraintTag, SecondStageTag,
                       TimedNLPModel, TwoStageCore, WrapperNLPModel,
                       get_con_scen, get_nscen, get_var_scen, new_tag, timings)

__version__ = "0.1.0"

__all__ = [
    "Core", "Model", "Solution", "Block", "Constraint", "Expression", "Records",
    "Product", "product", "Node", "Constant", "sum", "prod",
    "trace", "available_solvers", "install_solver", "solution",
    *_CORE_FUNCTIONS, *_MODEL_FUNCTIONS, *_RESULT_FUNCTIONS,
    "backends", "install_backend", "ModelError", "__version__",
    "WrapperNLPModel", "TimedNLPModel", "CompressedNLPModel", "timings",
    "as_cupy", "from_cupy",
    "TwoStageCore", "EachScenario", "get_nscen", "get_var_scen", "get_con_scen",
    "VectorNonlinearOracle", "ScalarNonlinearOracle", "OracleEvaluator",
    "has_matfree_jac", "has_matfree_hess", "add_eval", "embed_oracle",
    "new_tag", "FirstStageTag", "SecondStageTag",
    "FirstStageConstraintTag", "SecondStageConstraintTag",
]


_SUBMODULES = frozenset({"ops", "core", "node", "model", "solve", "testing",
                         "advanced", "_bridge"})


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
