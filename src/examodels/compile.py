"""Compile a recipe into a shared library, callable without Julia.

A recipe — a `Core` built with `nargs=` — has its structure and its data
separated, which is exactly what ahead-of-time compilation needs. Compiling one
produces a library exposing the model through a plain C interface:

    core, N = exa.Core(nargs=1)
    x = core.add_var(N, start=1.0)
    core.add_obj(lambda i: (x[i] - 2.0)**2, over=exa.srange(0, N))

    lib = exa.compile_library(core, "rosenrock", arg=1000)

Load the result with [cnlpmodels](https://github.com/MadNLP/cnlpmodels-py) —
ctypes and numpy, no Julia in the process — or from Julia with
[CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl). Having imported
`cnlpmodels`:

    m = cnlpmodels.CModel("rosenrock", args=1000)

Neither consumer is a dependency of this package: the library is a plain
shared object, and who loads it is the caller's business.

The compiler is a backend package rather than a Python one, so it is installed
through this package, once per environment — the same arrangement as solvers.
"""

from . import _bridge as _b

#: The backend package that does the compiling, and where it comes from. It is
#: a subdirectory package of the backend's own repository rather than a
#: registered one, so the source has to be named.
COMPILER = {
    "name": "ExaModelsC",
    "uuid": "3d1e9a26-5b74-4f0c-9a2b-7c8f4e11d3a7",
    "url": "https://github.com/exanauts/ExaModels.jl",
    "subdir": "ExaModelsC",
}


def install_compiler():
    """Install the compiler backend into this environment (one-off; needs a network)."""
    import juliapkg
    juliapkg.add(COMPILER["name"], COMPILER["uuid"],
                 url=COMPILER["url"], subdir=COMPILER["subdir"])
    juliapkg.resolve()


def compiler_available():
    """Whether the compiler backend is present, without importing it."""
    return bool(_b.seval(f'Base.find_package("{COMPILER["name"]}") !== nothing'))


def compile_library(core, out, *, arg, prefix=None, bundle=True, verbose=False):
    """Compile `core` into a shared library under `out`, and return its path.

    `core` must be a recipe — a `Core` built with `nargs=1`, since the C
    interface carries one instantiation argument. `arg` is an *example* value
    for that placeholder: its value is never baked in, but its type is, because
    the compiler needs the call graph resolved statically. The size is supplied
    per instance at run time.

    `out` may be a directory, or a bare **name**, in which case the library is
    installed on the `CNLPMODELS_PATH` search path where both consumers find it
    by that name.

    `bundle=True` (the default) carries a private copy of the Julia runtime, so
    the library needs no Julia installed — and it is the only form loadable from
    Julia itself. `bundle=False` gives a single small library for Python and C
    callers, which is the usual case here.
    """
    from .core import Core
    from .recipe import _unwrap

    if not isinstance(core, Core):
        raise TypeError("compile_library takes a Core built with nargs=")
    if not getattr(core, "args", ()):
        raise ValueError(
            "this core has no placeholders, so there is nothing to supply at run "
            "time — build it with `Core(nargs=1)` and write the model against the "
            "placeholder it returns."
        )
    if not compiler_available():
        raise RuntimeError(
            "the compiler backend is not installed in this environment; run "
            "`examodels.install_compiler()` once (it needs a network)."
        )

    kw = {"arg": _unwrap(arg), "bundle": bundle, "verbose": verbose}
    if prefix is not None:
        kw["prefix"] = str(prefix)
    got = _b.guard(_b.compile_library, core._core, str(out), **kw)
    return str(_b.at_field(got, "libpath"))
