"""Compile models into a shared library, callable without Julia.

A recipe — a `Core` built with `nargs=` — has its structure and its data
separated, which is exactly what ahead-of-time compilation needs. Compiling one
produces a library exposing the model through a plain C interface:

    core, N = exa.Core(nargs=1)
    x = core.add_var(N, start=1.0)
    core.add_obj(lambda i: (x[i] - 2.0)**2, over=exa.srange(0, N))

    lib = exa.compile_library("@rosenrock", core, 1000)

The example value (`1000`) is never baked in: its *type* is, because the
compiler needs the call graph resolved statically, while the size is supplied
per instance at run time.

A core with no placeholders is a **fixed** model — compile it by passing no
example at all:

    lib = exa.compile_library("@fixed", plain_core)

and **several models share one library**, each under its own name, by passing a
mapping instead of a core:

    lib = exa.compile_library("@grid", {"acopf": (ac_core, 100),
                                        "dcopf": (dc_core, 100),
                                        "small": fixed_core})

They share the library file — and, in a bundle, its one privatized copy of the
Julia runtime — which is the reason to co-package them rather than emit a
library each. The consumer selects by name.

Load the result with [cnlpmodels](https://github.com/MadNLP/cnlpmodels-py) —
ctypes and numpy, no Julia in the process — or from Julia with
[CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl). Having imported
`cnlpmodels`:

    m = cnlpmodels.CModel("rosenrock", 1000)   # by name, off CNLPMODELS_PATH
    m = cnlpmodels.CModel("@grid", "acopf", 100)      # one of several

Neither consumer is a dependency of this package: the library is a plain shared
object implementing the cnlp ABI, and who loads it is the caller's business.

The compiler is a backend package rather than a Python one, so it is installed
through this package, once per environment — the same arrangement as solvers.
"""


from . import _bridge as _b

__all__ = ["compile_library", "compiler_available", "install_compiler", "CompiledLibrary"]

#: The backend package that does the compiling, and where it comes from. It is
#: a subdirectory package of the backend's own repository rather than a
#: registered one, so the source has to be named.
COMPILER = {
    "name": "ExaModelsCompiler",
    "uuid": "3d1e9a26-5b74-4f0c-9a2b-7c8f4e11d3a7",
    "url": "https://github.com/madsuite-org/ExaModels.jl",
    "subdir": "ExaModelsCompiler",
}


class CompiledLibrary:
    """What a compile produced: the library, where it went, and its model names.

    Usable directly wherever a path is expected (`str(lib)`, `open(lib)`,
    `os.fspath(lib)`), since the path is what most callers want; the model
    names matter only for a library carrying more than one.
    """

    __slots__ = ("path", "outdir", "prefixes")

    def __init__(self, path, outdir, prefixes):
        self.path = str(path)
        #: the directory the compile wrote into (a bundle is a directory)
        self.outdir = str(outdir)
        #: the name each model answers to, in the order they were given
        self.prefixes = tuple(prefixes)

    def __fspath__(self):
        return self.path

    def __str__(self):
        return self.path

    def __repr__(self):
        return f"<CompiledLibrary {self.path!r} models={list(self.prefixes)}>"

    def __eq__(self, other):
        return self.path == other if isinstance(other, str) else NotImplemented

    def __hash__(self):
        return hash(self.path)


def install_compiler():
    """Install the compiler backend into this environment (one-off; needs a network)."""
    import juliapkg
    juliapkg.add(COMPILER["name"], COMPILER["uuid"],
                 url=COMPILER["url"], subdir=COMPILER["subdir"])
    juliapkg.resolve()


def compiler_available():
    """Whether the compiler backend is present, without importing it."""
    return bool(_b.seval(f'Base.find_package("{COMPILER["name"]}") !== nothing'))


def _spec(name, value):
    """One `name => core, examples...` model, from what the caller wrote."""
    from .core import Core
    from .recipe import _unwrap
    core, args = (value[0], tuple(value[1:])) if isinstance(value, tuple) else (value, ())
    if not isinstance(core, Core):
        raise TypeError(
            f"model {name!r}: expected a Core, or a tuple (core, *examples); "
            f"got {type(core).__name__}")
    return core._core, [_unwrap(a) for a in args]


def compile_library(out, models, *examples, prefix=None, trim="safe", bundle=False,
                    verbose=False, argfun=None):
    """Compile `models` into a shared library under `out`, and return it.

    `models` is either a single `Core` — with its example instantiation values
    as the remaining positional arguments, or none at all for a fixed model —
    or a mapping of name to core (or to a `(core, *examples)` tuple), which
    puts several models in one library.

    `out` is a path, or **`"@name"`** — the sigil asks for the library to be
    installed on the `CNLPMODELS_PATH` search path, where both consumers find it
    by that name. A bare name with no `@` is an ordinary relative path.

    `bundle=False` (the default) emits a single small library linked against
    the Julia the compile ran on, which the consumer's machine must also have.
    `bundle=True` carries a privatized copy of the runtime instead — around
    80 MB, needing no Julia at the far end, and the only form loadable from
    Julia itself.

    `trim` is passed to the compiler as its trimming mode; `argfun` is a *Julia*
    function that turns the examples into the tuple the core is instantiated
    with, for a model whose data cannot cross the boundary.
    """
    if not compiler_available():
        raise RuntimeError(
            "the compiler backend is not installed in this environment; run "
            "`examodels.install_compiler()` once (it needs a network)."
        )
    if callable(argfun) and not _b.is_julia(argfun):
        raise TypeError(
            "argfun must be a Julia function: it is compiled into the library "
            "and called there, where no Python interpreter exists. Build one "
            "with `examodels._bridge.seval(\"...\")`."
        )

    kw = {"trim": str(trim), "bundle": bool(bundle), "verbose": bool(verbose)}
    if hasattr(models, "items"):
        if examples:
            raise TypeError(
                "several models were given as a mapping, so each carries its own "
                "examples — write them as `{'name': (core, 1000)}` rather than "
                "positionally.")
        if prefix is not None:
            raise TypeError(
                "`prefix` has no meaning for several models: the library file is "
                "named by `out` and each model is named by its own key.")
        if argfun is not None:
            raise TypeError(
                "`argfun` belongs to one model; give it per model by writing the "
                "core with its own argument function on the Julia side.")
        if not models:
            raise ValueError("give at least one model")
        names = [str(n) for n in models]
        pairs = []
        for name in names:
            core, args = _spec(name, models[name])
            pairs.append((name, core, args))
        got = _b.guard(_b.compile_models, [p[0] for p in pairs], [p[1] for p in pairs],
                       [p[2] for p in pairs], str(out), **kw)
        return _library(got, names)

    core, args = _spec(out, (models, *examples))
    if prefix is not None:
        kw["prefix"] = str(prefix)
    if argfun is not None:
        kw["argfun"] = argfun
    got = _b.guard(_b.compile_library, str(out), core, args, **kw)
    return _library(got, [str(_b.at_field(got, "prefix"))])


def _library(got, prefixes):
    # `outdir` is read off the result rather than derived from the path: a
    # bundle's directory is the artifact, and it is not the library's parent.
    return CompiledLibrary(_b.at_field(got, "libpath"), _b.at_field(got, "outdir"),
                           prefixes)
