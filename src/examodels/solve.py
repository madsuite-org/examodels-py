"""Solver front end. Backends are loaded on first use."""
from . import _bridge as _b
from .model import Solution

__all__ = ["solve", "available_solvers", "install_solver"]

#: name -> (backend package, uuid, entry point)
SOLVERS = {
    "ipopt": ("NLPModelsIpopt", "f4238b75-b362-5c4c-b852-0801c9a21d71", "ipopt"),
    "madnlp": ("MadNLP", "2621e9c9-9eb4-46b1-8089-e8c72242dfb6", "madnlp"),
}
_loaded = set()


def available_solvers():
    return sorted(SOLVERS)


def install_solver(name):
    """Install a solver backend into this environment (one-off; needs a network)."""
    pkg, uuid, _ = _lookup(name)
    import juliapkg
    juliapkg.add(pkg, uuid)
    juliapkg.resolve()


def _lookup(name):
    try:
        return SOLVERS[name]
    except KeyError:
        raise ValueError(f"unknown solver {name!r}; available: {available_solvers()}") from None


def solve(model, solver=None, print_level=0, **options):
    """Solve `model`, returning a `Solution`. Extra keywords go to the solver.

    Without `solver=`, one is chosen by where the model's arrays live: Ipopt cannot
    take device arrays, so a model built on an accelerator goes to MadNLP.
    """
    if solver is None:
        solver = "madnlp" if _on_device(model) else "ipopt"
    pkg, _uuid, entry = _lookup(solver)
    if solver not in _loaded:
        try:
            _b.seval(f"using {pkg}")
        except Exception:                                    # noqa: BLE001
            raise _b.ModelError(
                f"the {solver!r} solver is not installed in this environment. "
                f"Install it with: examodels.install_solver({solver!r})") from None
        _loaded.add(solver)

    if solver == "ipopt":
        options.setdefault("print_level", print_level)
        if not print_level:
            options.setdefault("sb", "yes")   # Ipopt prints its banner even at level 0
    elif solver == "madnlp":
        if not print_level:
            options.setdefault("print_level", _b.seval("MadNLP.ERROR"))
        if _on_device(model):
            # A model whose arrays live on a device needs a linear solver that also
            # runs there; the default one indexes elementwise and cannot.
            options.setdefault("linear_solver", _device_linear_solver())

    import time
    t0 = time.perf_counter()
    raw = _b.guard(getattr(_b.jl, entry), model._jl, **options)
    return Solution(raw, elapsed=time.perf_counter() - t0)


def _on_device(model):
    """True when the model's arrays are not plain host arrays."""
    return not bool(_b.seval("m -> m.meta.x0 isa Array")(model._jl))


#: preferred first. Some are declared by MadNLPGPU but only *assigned* when their
#: own backing package is present, so availability is checked rather than assumed.
DEVICE_LINEAR_SOLVERS = ("CUDSSSolver", "LapackCUDASolver")


def _device_linear_solver():
    # MadNLPGPU declares these names but only *assigns* them once their backing
    # packages are loaded, so load those too -- best effort, since a given
    # environment may not have all of them.
    for pkgs in ("MadNLPGPU, CUDA, CUDSS", "MadNLPGPU, CUDA", "MadNLPGPU"):
        try:
            _b.seval(f"using {pkgs}")
            break
        except Exception:                                # noqa: BLE001
            continue
    for name in DEVICE_LINEAR_SOLVERS:
        if bool(_b.seval(f"try; getglobal(MadNLPGPU, :{name}); true; catch; false; end")):
            return _b.seval(f"MadNLPGPU.{name}")
    raise _b.ModelError(
        "no device linear solver is available: MadNLPGPU is installed but none of "
        f"{', '.join(DEVICE_LINEAR_SOLVERS)} is usable. Install CUDSS.")
