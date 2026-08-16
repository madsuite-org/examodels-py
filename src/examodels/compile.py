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


import re as _re

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


#: Why a Python caller cannot supply `argfun`, and what to do instead. The
#: compiler resolves the function BY NAME from the generated library, so it must
#: belong to a package -- a lambda, a closure, and anything built at run time
#: through the bridge are all refused by it, at spec time, whatever this package
#: does. Python callers want the other route anyway: their data is already in
#: Python, so passing it across the boundary costs them nothing.
_ARGFUN_HELP = (
    "argfun cannot be {what}: the compiled library calls it by name in a "
    "process with no Python in it, and the compiler accepts only a named "
    "function defined at the top level of a Julia PACKAGE (not a lambda, not "
    "one built with seval). Pass the data as example values instead -- "
    "`compile_library(out, core, n, table)` -- which needs no Julia: scalars, "
    "arrays and tables of them all cross the boundary, and the consumer then "
    "supplies them per instance. Use argfun only for data that must stay on "
    "the Julia side, and give it a function from an installed package."
)


#: A package-qualified Julia function name, and nothing else -- it is evaluated
#: as source, so it is checked the way `new_tag` checks a tag name.
_ARGFUN_NAME = _re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")


def _named_argfun(path):
    """`"ExaPowerIO.parse_case"` -> that function, so Python need not write Julia.

    The compiler needs a named function belonging to a package. NAMING one is
    something a Python caller can do without knowing the language, which
    writing one is not -- so a string is resolved here rather than refused.
    """
    if not _ARGFUN_NAME.match(path):
        raise ValueError(
            f"argfun as a string must name a function in a package, qualified "
            f"by it -- 'ExaPowerIO.parse_case'. Got {path!r}.")
    root = path.split(".")[0]
    try:
        return _b.seval(f"import {root}; {path}")
    except Exception:                                        # noqa: BLE001
        raise _b.ModelError(
            f"no function {path!r} is available: is {root!r} installed in this "
            f"environment? Backend packages are installed through juliapkg, the "
            f"way `install_solver` and `install_compiler` do it."
        ) from None


def _example(name, v):
    """One example instantiation value, as the type its storage will have.

    The compiler emits storage of exactly the example's Julia type, so a numpy
    array must arrive as a `Vector{Float64}` (or `{Int64}`) and a table of rows
    as a vector of named tuples. Left alone, both cross as the wrappers
    juliacall makes of them, which the compiler refuses -- correctly, since no
    library can be compiled around a live Python object.
    """
    import numpy as np

    from .node import _table, is_table
    from .recipe import Arg
    if isinstance(v, Arg):
        return v._jl
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(
            f"model {name!r}: an example may be a number, an array of numbers, "
            f"or a table of them -- the C boundary carries 64-bit integers and "
            f"floats. A bool is neither; say 0 or 1 if that is what is meant.")
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, str):
        return v                       # a string reaches argfun, never storage
    if is_table(v):
        jl, _n = _table(v)
        return jl
    a = np.asarray(v)
    if a.ndim != 1:
        raise TypeError(
            f"model {name!r}: an array example must be one-dimensional; got "
            f"shape {a.shape}. Flatten it, or pass a table of rows.")
    if a.dtype.kind in "iu":
        return _b.vec_i64(a)
    if a.dtype.kind == "f":
        return _b.vec_f64(a)
    raise TypeError(
        f"model {name!r}: an example of dtype {a.dtype} cannot cross the C "
        f"boundary, which carries 64-bit integers and floats.")


def _spec(name, value):
    """One `name => core, examples...` model, from what the caller wrote."""
    from .core import Core
    core, args = (value[0], tuple(value[1:])) if isinstance(value, tuple) else (value, ())
    if not isinstance(core, Core):
        raise TypeError(
            f"model {name!r}: expected a Core, or a tuple (core, *examples); "
            f"got {type(core).__name__}")
    return core._core, [_example(name, a) for a in args]


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

    `trim` is passed to the compiler as its trimming mode.

    `argfun` is for a model whose data should NOT cross the C boundary: the
    library carries the function, is handed one string or integer, and calls it
    to obtain the instantiation values. It has to be a named function belonging
    to a Julia package, so it is not reachable from Python -- pass the data as
    example values instead, which is the Python-native route and needs no Julia
    at all. See `_ARGFUN_HELP`.
    """
    if not compiler_available():
        raise RuntimeError(
            "the compiler backend is not installed in this environment; run "
            "`examodels.install_compiler()` once (it needs a network)."
        )
    if isinstance(argfun, str):
        argfun = _named_argfun(argfun)
    elif argfun is not None and not _b.is_julia(argfun):
        raise TypeError(_ARGFUN_HELP.format(what="a Python function"))

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
