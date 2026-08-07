"""The single point at which juliacall is touched — and it is touched lazily.

Importing this module does NOT start Julia; the runtime boots on first use, so
`import examodels` stays instant and users never see a startup stall they cannot
explain. Nothing outside this module imports juliacall.
"""
import json
from pathlib import Path
from types import SimpleNamespace

_S = None

#: Backend symbols this package calls. Several are NOT exported by the backend
#: (`_UNIVARIATES`, `_BIVARIATES`, `fulltype_display!`) or are low-level entry
#: points, so the coupling is listed here rather than left implicit, and checked
#: once at startup. A backend upgrade that removes one fails immediately, with a
#: message naming the symbol, instead of at a user's first model build.
REQUIRED = (
    "ExaCore", "ExaModel", "add_var", "add_par", "add_obj", "add_con",
    "DataSource", "Constant", "obj", "solution", "get_value", "set_value!",
    "_UNIVARIATES", "_BIVARIATES", "fulltype_display!",
)


def _compat():
    """The backend version range this package declares — read from the one place
    it is written down, so the runtime check and the installer cannot disagree."""
    spec = json.loads((Path(__file__).parent / "juliapkg.json").read_text())
    return spec["packages"]["ExaModels"]["version"]


def _configure_runtime():
    """Settings the runtime reads at startup, so they must be set before import.

    The user's personal Julia startup file is skipped: it belongs to their
    interactive sessions, and running it inside a library's backend makes this
    package's behaviour depend on their dotfiles. (On this machine it loads Revise,
    which then fails against a custom system image.)
    """
    import os
    os.environ.setdefault("PYTHON_JULIACALL_STARTUPFILE", "no")


def _satisfies(version, bound):
    """Julia's caret semantics: the leading non-zero component must match."""
    want = tuple(int(p) for p in bound.split("."))
    lead = next((i for i, v in enumerate(want) if v), len(want) - 1)
    return version[: lead + 1] == want[: lead + 1] and version >= want


def _check(jl):
    missing = [n for n in REQUIRED if not bool(jl.seval(f"isdefined(ExaModels, :var\"{n}\")"))]
    version = str(jl.seval("string(pkgversion(ExaModels))"))
    want = _compat()
    got = tuple(int(p) for p in version.split("-")[0].split("."))
    ok = any(_satisfies(got, bound.strip()) for bound in want.split(","))
    if missing or not ok:
        raise RuntimeError(
            f"the ExaModels backend in this environment (version {version}) is not the "
            f"one this package supports (declared: {want})"
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + ". Reinstall the backend, or upgrade `examodels`."
        )
    return version


def _boot():
    global _S
    if _S is not None:
        return _S
    _configure_runtime()
    from juliacall import Main as jl
    jl.seval("using ExaModels, NLPModels")
    version = _check(jl)
    _S = SimpleNamespace(
        jl=jl,
        version=version,
        EM=jl.ExaModels,
        seval=jl.seval,
        ops={op: jl.seval(op) for op in ("+", "-", "*", "/", "^")},
        # Julia lowers `x^2` with a LITERAL exponent to `Base.literal_pow(^, x, Val(2))`;
        # that is how ExaModels gets `Val{n}` exponents and the `^2 -> abs2` rewrite.
        # Python's `**` would otherwise pass a runtime Int64 and lose the specialization.
        literal_pow=jl.seval("(x, n) -> Base.literal_pow(^, x, Val(n))"),
        typestr=jl.seval("x -> string(typeof(x))"),
        same_ty=jl.seval("(a, b) -> typeof(a) === typeof(b)"),
        getidx=jl.seval("(v, i) -> v[i]"),
        getidxn=jl.seval("(v, is...) -> v[is...]"),
        # Dimensions are passed as parallel lo/hi lists and the ranges are built
        # here: a range handed back to Python returns as a StepRange, which the
        # backend's size machinery rejects.
        # A parameter block is declared with an explicit 0-based range for the same
        # reason variables are: an array-shaped dimension would be 1-based.
        add_par_range=jl.seval("(c, lo, hi, vals) -> ExaModels.add_par(c, "
                               "UnitRange(Int(lo), Int(hi)); value = vals)"),
        add_var_dims=jl.seval("(c, los, his; kw...) -> ExaModels.add_var(c, "
                              "(UnitRange(Int(l), Int(h)) for (l, h) in zip(los, his))...; kw...)"),
        # An index set must never round-trip through Python: a Julia range comes
        # back out as a Python `range` and returns as a StepRange, which several of
        # the backend's size and dispatch paths reject (get_lcon/get_ucon among
        # them). So the range is built inside each call that consumes it. Vectors
        # -- what `Records` produces -- do not convert, and pass through fine.
        gen_range=jl.seval("(node, a, b) -> Base.Generator(_ -> node, UnitRange(a, b))"),
        gen_iter=jl.seval("(node, itr) -> Base.Generator(_ -> node, itr)"),
        # A product index set: several ranges, again built here rather than handed
        # across the boundary.
        # Collected, not left lazy: a bare ProductIterator has no `keys`, which the
        # objective path needs. Collecting keeps the rectangular shape, which is
        # what makes the block multi-dimensional rather than merely flattened.
        product=jl.seval("(los, his) -> collect(Iterators.product("
                         "(UnitRange(Int(l), Int(h)) for (l, h) in zip(los, his))...))"),
        obj_range=jl.seval("(c, e, a, b) -> ExaModels.add_obj(c, e, UnitRange(a, b))"),
        obj_iter=jl.seval("(c, e, itr) -> ExaModels.add_obj(c, e, itr)"),
        getfield_=jl.seval("(n, s) -> getproperty(n, Symbol(s))"),
        exa_sum=jl.seval("ns -> ExaModels.SumNode(Tuple(ns))"),
        exa_prod=jl.seval("ns -> ExaModels.ProdNode(Tuple(ns))"),
        # Evaluation buffers must live wherever the model's arrays live, so they
        # are always allocated from the model rather than assumed to be host
        # memory. On CPU this is the identity; on a device it is what makes the
        # same code path work at all.
        like=jl.seval("(m, n) -> similar(m.meta.x0, n)"),
        # `Vector{Float64}(v)` first: a numpy array arrives as a PyArray, and
        # copying one straight into device memory drops into a scalar path that
        # cannot be compiled for a GPU.
        upload=jl.seval("(m, v) -> (h = Vector{Float64}(v); "
                        "y = similar(m.meta.x0, length(h)); copyto!(y, h); y)"),
        tohost=jl.seval("x -> x isa Array ? x : Array(x)"),
        # The pair must be built and consumed without crossing back into Python:
        # a Julia Pair converts to a Python tuple on the way out, and returns as a
        # Tuple, which is not what the augmentation entry point accepts.
        aug_range=jl.seval("(i, e, a, b) -> Base.Generator(_ -> (i => e), UnitRange(a, b))"),
        aug_iter=jl.seval("(i, e, itr) -> Base.Generator(_ -> (i => e), itr)"),
        mkrecords=jl.seval("""(names, cols) -> begin
            nt = Tuple(Symbol.(names))
            [NamedTuple{nt}(Tuple(c[k] for c in cols)) for k in eachindex(first(cols))]
        end"""),
        valtrue=jl.seval("Val(true)"),
    )
    return _S


def __getattr__(name):                     # PEP 562: lazy module attributes
    # Dunder probes (__path__, __all__, __wrapped__, ...) come from the import
    # machinery and from tooling; booting the backend for one of those would make
    # `import examodels` pay the startup cost it is designed to defer.
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_boot(), name)


def started():
    """True once the backend has actually been started."""
    return _S is not None


def unwrap(o):
    """A package handle -> the object it wraps; anything else passes through."""
    return o._jl if hasattr(o, "_jl") else o


class ModelError(Exception):
    """An error raised while building or solving a model."""


def translate(exc):
    """Turn a backend error into a plain Python exception with a readable message."""
    msg = str(exc)
    head = msg.split("\n", 1)[0].strip()
    for noise in ("JuliaError: ", "MethodError: "):
        head = head.replace(noise, "")
    if "no method matching" in head or "not defined for this combination" in head:
        return TypeError(
            f"unsupported operation in an expression: {head}. Only the operators "
            f"registered with ExaModels can appear in a traced function."
        )
    # Map the backend's error kinds onto Python's, once, rather than re-validating
    # every argument on the way in.
    for marker, cls in (("DimensionMismatch", ValueError), ("BoundsError", IndexError),
                        ("ArgumentError", ValueError)):
        if head.startswith(marker):
            return cls(head[len(marker):].lstrip(": "))
    return ModelError(head)


def guard(fn, *args, **kwargs):
    """Run `fn`, re-raising backend errors as Python ones (no foreign traceback)."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:                                  # noqa: BLE001
        if type(e).__name__ == "JuliaError":
            raise translate(e) from None
        raise
