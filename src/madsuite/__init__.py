"""madsuite — Python interface to ExaModels.jl.

    import madsuite as exa

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
from ._bridge import ModelError
from .advanced import (
    CompressedNLPModel,
    EachScenario,
    FirstStageConstraintTag,
    FirstStageTag,
    OracleEvaluator,
    ScalarNonlinearOracle,
    SecondStageConstraintTag,
    SecondStageTag,
    TimedNLPModel,
    TwoStageCore,
    VectorNonlinearOracle,
    WrapperNLPModel,
    add_eval,
    as_cupy,
    embed_oracle,
    from_cupy,
    get_con_scen,
    get_nscen,
    get_var_scen,
    has_matfree_hess,
    has_matfree_jac,
    new_tag,
    timings,
)
from .compile import (
    CompiledLibrary,
    compile_library,
    compiler_available,
    install_compiler,
)
from .core import Core, backends, install_backend, trace
from .model import Model, Solution
from .node import (
    Block,
    Constant,
    Constraint,
    Expression,
    Node,
    Product,
    TupleNode,
    prod,
    sum,
)
from .recipe import Arg, is_placeholder, recipe, srange
from .solve import available_solvers, install_solver, solve


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

__version__ = "0.1.0"

__all__ = [
    "Arg",
    "CompiledLibrary",
    "compile_library",
    "compiler_available",
    "install_compiler",
    "is_placeholder",
    "recipe",
    "srange",
    "Core", "Model", "Solution", "Block", "Constraint", "Expression",
    "Product", "product", "Node", "TupleNode", "Constant", "sum", "prod",
    "trace", "available_solvers", "install_solver", "solution",
    "add_var", "add_par", "add_obj", "add_con", "add_expr",
    "get_value", "set_value", "get_start", "set_start",
    "get_lvar", "set_lvar", "get_uvar", "set_uvar",
    "get_lcon", "set_lcon", "get_ucon", "set_ucon",
    "objective", "gradient", "constraints", "violation", "solve",
    "multipliers", "multipliers_L", "multipliers_U",
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
                         "advanced", "_bridge", "_record", "_cache",
                         "_wire", "_warm", "daemon", "compile", "recipe"})


def __getattr__(name):
    """Expose ExaModels' registered math functions (sin, exp, log, ...) lazily.

    They are generated from the operator list ExaModels itself registers, so this
    package never carries its own copy of that list.
    """
    if name.startswith("__") or name in _SUBMODULES:
        # Never re-enter for a submodule: `from . import ops` consults this hook while
        # the submodule is still being imported, which would recurse forever.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import _bridge, _record
    if _record.tracing() or not _bridge.loaded():
        # Inside a *recorded* trace, importing ops would boot Julia (its operator
        # list comes from the backend). Return a recorder for any name; a name
        # ExaModels does not register fails at replay, against the real list.
        #
        # The same deferral while Julia is simply *not up yet*: a script doing
        # `sin = exa.sin` at the top cannot yet know whether it will record
        # (daemon, cache) or build eagerly, and resolving through ops here
        # would boot Julia in a process that may never need it — measured at
        # ~10 s of what looked like "tracing cost".  The recorder it gets
        # records PNodes and forwards eager calls to the real operator
        # (loading ops then, when Julia is already inevitable); a misspelled
        # name errors at first eager call or at replay instead of here.  Once
        # Julia is up, resolution validates eagerly as before.  (Deferred
        # results are deliberately not cached in globals(): a later access
        # with Julia up gets the validated, cached form.)
        return _record.math_fn(name)
    import importlib
    ops = importlib.import_module(".ops", __name__)
    if hasattr(ops, name):
        # Cache a dispatcher, not the ops function: once `exa.sin` lives in
        # globals(), this hook is never consulted again, and a later recorded
        # trace must still divert PNode arguments.
        fn = _record.math_fn(name, getattr(ops, name))
        globals()[name] = fn
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    import importlib
    ops = importlib.import_module(".ops", __name__)
    return sorted(set(__all__) | set(ops.__all__))
