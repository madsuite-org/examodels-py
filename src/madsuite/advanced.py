"""The parts of the backend beyond ordinary algebraic modelling.

Model wrappers, two-stage stochastic programs, tags, and nonlinear oracles. Kept
apart from `core`/`model` so the everyday path stays small; nothing here is loaded
until it is used.
"""
import numpy as np

from . import _bridge as _b

__all__ = [
    "as_cupy", "from_cupy",
    "VectorNonlinearOracle", "ScalarNonlinearOracle", "OracleEvaluator",
    "has_matfree_jac", "has_matfree_hess", "add_eval", "embed_oracle",
    "WrapperNLPModel", "TimedNLPModel", "CompressedNLPModel",
    "TwoStageCore", "EachScenario", "get_nscen", "get_var_scen", "get_con_scen",
    "new_tag", "FirstStageTag", "SecondStageTag",
    "FirstStageConstraintTag", "SecondStageConstraintTag",
]


# ------------------------------------------------------------- wrappers ------
def _wrap(name, model):
    from .model import Model
    return Model(_b.guard(getattr(_b.EM, name), model._jl))


def WrapperNLPModel(model):
    """Buffer a model's evaluations through host arrays, for a solver that needs it."""
    return _wrap("WrapperNLPModel", model)


def TimedNLPModel(model):
    """Wrap a model so it records how long each evaluation takes."""
    return _wrap("TimedNLPModel", model)


def CompressedNLPModel(model):
    """Wrap a model with duplicate Jacobian and Hessian entries merged."""
    return _wrap("CompressedNLPModel", model)


def timings(model):
    """The timing report of a `TimedNLPModel`, as text.

    Captured from standard output: the backend's report writes there rather than
    to the stream it is handed, so `sprint` comes back empty.
    """
    return str(_b.guard(_b.seval("""m -> begin
        old = stdout
        rd, wr = redirect_stdout()
        try
            print(m)
        finally
            redirect_stdout(old)
            close(wr)
        end
        read(rd, String)
    end"""), model._jl))


# ------------------------------------------------------------ two-stage ------
class EachScenario:
    """Marks a declaration as per-scenario (recourse) rather than shared (design)."""

    __slots__ = ()

    def __repr__(self):
        return "EachScenario()"


def TwoStageCore(nscen, backend=None):
    """A core for a two-stage stochastic program with `nscen` scenarios.

        core = TwoStageCore(3)
        d = add_var(core, 2)                       # design, shared
        v = add_var(core, EachScenario(), 4)       # recourse, per scenario
        add_con(core, EachScenario(), lambda i: v[i] - d[0], over=range(4))

    Everything else about it is an ordinary `Core`; only how it starts differs.
    """
    from .core import Core, _backend
    core = Core.__new__(Core)
    core._core = _b.guard(_b.EM.TwoStageExaCore, int(nscen),
                          backend=_backend(backend), concrete=_b.valtrue)
    core.nscen = int(nscen)
    return core


def get_nscen(model):
    """How many scenarios a two-stage model has."""
    return int(_b.guard(_b.EM.get_nscen, model._jl))


def get_var_scen(model):
    """Which scenario each variable belongs to (0 for the shared first stage)."""
    return np.asarray(_b.tohost(_b.guard(_b.EM.get_var_scen, model._jl)), dtype=np.int64)


def get_con_scen(model):
    """Which scenario each constraint row belongs to (0 for the first stage)."""
    return np.asarray(_b.tohost(_b.guard(_b.EM.get_con_scen, model._jl)), dtype=np.int64)


# ----------------------------------------------------------------- tags ------
def new_tag(name, kind="variable"):
    """Define a tag to mark variables or constraints with.

    The backend dispatches on a tag's *type*, so one is created here rather than
    passed as a value. `name` is checked before use: it is the only caller-supplied
    string in this package that reaches the backend as source rather than as data,
    and an unchecked one would let any Julia code through.
    """
    if not (isinstance(name, str) and name.isidentifier() and name.isascii()):
        raise ValueError(f"a tag name must be a plain identifier, got {name!r}")
    try:
        super_ = {"variable": "AbstractVariableTag",
                  "constraint": "AbstractConstraintTag"}[kind]
    except KeyError:
        raise ValueError(f"kind must be 'variable' or 'constraint', got {kind!r}") from None
    # The type is defined and then used in the same evaluation, so the binding is
    # reached in a world older than the one that defines it. Julia 1.12 tightened
    # that from a warning to an error, and `invokelatest` is the sanctioned way
    # to say "in whatever world exists by the time this runs".
    return _b.seval(f"""begin
        isdefined(Main, :{name}) ||
            Core.eval(Main, :(struct {name} <: ExaModels.{super_} end))
        Base.invokelatest(() -> Main.{name}())
    end""")


def _tag(name):
    return _b.seval(f"ExaModels.{name}()")


def FirstStageTag():
    return _tag("FirstStageTag")


def SecondStageTag():
    return _tag("SecondStageTag")


def FirstStageConstraintTag():
    return _tag("FirstStageConstraintTag")


def SecondStageConstraintTag():
    return _tag("SecondStageConstraintTag")


# -------------------------------------------------------------- oracles ------
class Oracle:
    """A block of constraints, or an objective term, computed by your own code."""

    __slots__ = ("_jl", "kind")

    def __init__(self, jlobj, kind):
        self._jl, self.kind = jlobj, kind

    def __repr__(self):
        return f"<{self.kind} oracle>"


def _ints(v):
    return [int(i) for i in (v if v is not None else ())]


def VectorNonlinearOracle(*, nvar, ncon, f, jac=None, hess=None,
                          jac_rows=(), jac_cols=(), hess_rows=(), hess_cols=(),
                          lcon=None, ucon=None,
                          jvp=None, vjp=None, hvp=None, adapt=True):
    """A constraint block you evaluate yourself.

    `f(c, x)` fills `c` with residuals. Supply either explicit derivatives --
    `jac(vals, x)` and `hess(vals, x, y)` with their sparsity patterns -- or the
    matrix-free products `jvp(Jv, x, v)`, `vjp(Jtv, x, w)`, `hvp(Hv, x, w, v)`.

    `adapt=True` (the default here) copies the arrays to the host before each call,
    which is what a Python callback needs. Set it False only for a callback that
    can run on the device -- which a Python one cannot.
    """
    lcon = np.zeros(ncon) if lcon is None else np.ascontiguousarray(lcon, dtype=np.float64)
    ucon = np.zeros(ncon) if ucon is None else np.ascontiguousarray(ucon, dtype=np.float64)
    return Oracle(_b.guard(_b.vector_oracle, int(nvar), int(ncon),
                           _ints(jac_rows), _ints(jac_cols),
                           _ints(hess_rows), _ints(hess_cols),
                           lcon, ucon, f, jac, hess, jvp, vjp, hvp, bool(adapt)),
                  "constraint")


def ScalarNonlinearOracle(*, nvar, f, grad, hvp=None,
                          hess_rows=(), hess_cols=(), adapt=True):
    """An objective term you evaluate yourself: `f(x)` and `grad(g, x)`."""
    return Oracle(_b.guard(_b.scalar_oracle, int(nvar), f, grad, hvp,
                           _ints(hess_rows), _ints(hess_cols), bool(adapt)),
                  "objective")


def has_matfree_jac(oracle):
    """True when the oracle supplies Jacobian-vector products instead of a matrix."""
    return bool(_b.guard(_b.EM.has_matfree_jac, oracle._jl))


def has_matfree_hess(oracle):
    """True when the oracle supplies Hessian-vector products instead of a matrix."""
    return bool(_b.guard(_b.EM.has_matfree_hess, oracle._jl))


def OracleEvaluator(*args, **kwargs):
    """Constructed by the backend when an oracle is embedded; see `embed_oracle`."""
    return _b.guard(_b.EM.OracleEvaluator, *args, **kwargs)


def add_eval(core, cons, variables, f, **kwargs):
    """Attach an evaluation callback to existing constraints and variables."""
    core._core, out = _b.guard(_b.EM.add_eval, core._core,
                               tuple(_b.unwrap(c) for c in cons),
                               tuple(_b.unwrap(v) for v in variables), f, **kwargs)
    return out


def embed_oracle(core, block, output_dim, **kwargs):
    """Embed an oracle that maps a variable block to `output_dim` outputs."""
    core._core, out = _b.guard(_b.EM.embed_oracle, core._core,
                               _b.unwrap(block), int(output_dim), **kwargs)
    return out


# ------------------------------------------------- device interchange --------
class _CudaView:
    """Publishes a backend device array through CUDA's array interface.

    The backend does not expose that interface itself, so it is built here from
    the array's pointer, length and element size. `cupy.asarray` of this views the
    same memory: no host round-trip, and no copy on either side.
    """

    __slots__ = ("__cuda_array_interface__", "_owner")

    def __init__(self, jl_array):
        ptr, n, typestr = _b.guard(_b.device_info, jl_array)
        self.__cuda_array_interface__ = {
            "shape": (int(n),), "typestr": str(typestr),
            "data": (int(ptr), False), "version": 3, "strides": None,
        }
        #: the backend still owns the memory; hold it so it cannot be freed while
        #: something is looking at it
        self._owner = jl_array


def as_cupy(array):
    """View a backend device array as a CuPy array, sharing the same memory.

    The backend keeps ownership: the result is a view, so writing through it
    writes into the model.
    """
    # Checked before cupy is imported: a host array is a usage error worth
    # reporting even on a machine with no cupy at all.
    if not bool(_b.guard(_b.is_device, array)):
        raise TypeError("this array is in host memory; use the numpy accessors")
    import cupy
    return cupy.asarray(_CudaView(array))


def from_cupy(array):
    """View a CuPy array as a backend device array, sharing the same memory."""
    iface = getattr(array, "__cuda_array_interface__", None)
    if iface is None:
        raise TypeError("expected a CuPy array (or anything exposing "
                        "__cuda_array_interface__)")
    ptr, _read_only = iface["data"]
    n = 1
    for d in iface["shape"]:
        n *= d
    if iface["typestr"] not in ("<f8", "|f8"):
        raise TypeError(f"expected float64, got {iface['typestr']}")
    return _b.guard(_b.wrap_device_ptr, int(ptr), int(n))
