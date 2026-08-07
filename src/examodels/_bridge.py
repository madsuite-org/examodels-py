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
    if os.environ.get("PYTHON_JULIACALL_SYSIMAGE"):
        return
    try:
        from .sysimage import path
        img = path()
    except Exception:                                        # noqa: BLE001
        return
    if img.is_file():
        os.environ["PYTHON_JULIACALL_SYSIMAGE"] = str(img)


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
        mkrange=jl.seval("(a, b) -> a:b"),
        mkgen=jl.seval("(node, iter) -> Base.Generator(_ -> node, iter)"),
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
