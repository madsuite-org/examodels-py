"""Ahead-of-time compilation of the backend, to cut time-to-first-solve.

Most of the first-solve latency is fixed per-process cost: starting the runtime,
loading the packages, and compiling their generic machinery. All of it can be
baked into a system image once, after which a process starts with that work
already done. What cannot be removed is the compilation of *your* expression --
each distinct expression structure is a distinct type, created when you write the
function, so its derivative kernels are necessarily compiled at run time.
"""
import os
from pathlib import Path

__all__ = ["build", "path", "active"]

WORKLOAD = """
using ExaModels, NLPModels, NLPModelsIpopt, MadNLP

function _model(N, f, g)
    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, N; start = [i % 2 == 1 ? -1.2 : 1.0 for i = 1:N])
    c, _ = add_obj(c, (f(x, i) for i = 2:N))
    c, _ = add_con(c, (g(x, i) for i = 1:(N-2)))
    ExaModel(c)
end

lv_obj(x, i) = 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2
lv_con(x, i) = 3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) +
               4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3
alt_obj(x, i) = cos(x[i]) * tanh(x[i-1]) + log1p(x[i]^2)
alt_con(x, i) = sqrt(abs(x[i]) + 1) - x[i+1] / (1 + x[i+2]^2)

for (f, g) in ((lv_obj, lv_con), (alt_obj, alt_con))
    m = _model(50, f, g)
    x = similar(m.meta.x0); copyto!(x, m.meta.x0)
    obj(m, x); grad(m, x); cons(m, x)
    ipopt(m; print_level = 0, sb = "yes")
    madnlp(m; print_level = MadNLP.ERROR)
end

# parameters and the low-level expression entry point the Python layer uses
let c = ExaCore(concrete = Val(true))
    c, t = add_par(c, [100.0, 1.0])
    c, x = add_var(c, 20)
    c, _ = add_obj(c, (t[1] * x[i]^2 + t[2] for i = 1:20))
    gen = (x[i]^2 + x[i+1] for i = 1:19)
    c, _ = add_con(c, Base.Generator(_ -> gen.f(ExaModels.DataSource()), 1:19))
    m = ExaModel(c)
    ipopt(m; print_level = 0, sb = "yes")
    set_value!(m, t, [200.0, 0.5])
    ipopt(m; print_level = 0, sb = "yes")
end
"""

#: packages baked in. CUDA is deliberately absent: it is loaded only on request,
#: and including it would multiply the build time and the image size for users who
#: never touch a GPU. Build a separate image with `packages=` if you want one.
PACKAGES = ("PythonCall", "ExaModels", "NLPModels", "NLPModelsIpopt", "MadNLP")


def path():
    """Where the image lives for this environment."""
    if os.environ.get("EXAMODELS_SYSIMAGE"):
        return Path(os.environ["EXAMODELS_SYSIMAGE"])
    import juliapkg
    return Path(juliapkg.project()) / "examodels.so"


def active():
    """True when the current process started from a prebuilt image."""
    from . import _bridge
    if not _bridge.started():
        return bool(os.environ.get("PYTHON_JULIACALL_SYSIMAGE"))
    return bool(_bridge.seval('Base.JLOptions().image_file != ""')) and \
        str(path()) in str(_bridge.seval("unsafe_string(Base.JLOptions().image_file)"))


def build(target=None, packages=PACKAGES, filter_stdlibs=False, incremental=True):
    """Build the image. Takes several minutes and is needed only once per
    environment; afterwards every process picks it up automatically.

    `filter_stdlibs=True` drops unused standard libraries for a smaller image, and
    requires `incremental=False`, which makes the build considerably slower.
    """
    if filter_stdlibs and incremental:
        raise ValueError("filter_stdlibs=True requires incremental=False")
    from . import _bridge
    target = Path(target) if target else path()
    target.parent.mkdir(parents=True, exist_ok=True)

    workload = target.parent / "examodels_precompile.jl"
    workload.write_text(WORKLOAD)

    _bridge.seval("using PackageCompiler")
    _bridge.seval("(pkgs, out, wl, fs, inc) -> PackageCompiler.create_sysimage("
                  "Symbol.(collect(pkgs)); sysimage_path = out, "
                  "precompile_execution_file = wl, filter_stdlibs = fs, incremental = inc)")(
                      list(packages), str(target), str(workload),
                      bool(filter_stdlibs), bool(incremental))
    return target
