"""The single point at which juliacall is touched — and it is touched lazily.

Importing this module does NOT start Julia; the runtime boots on first use, so
`import examodels` stays instant and users never see a startup stall they cannot
explain. Nothing outside this module imports juliacall.
"""
import sys
from types import SimpleNamespace

_S = None


def _boot():
    global _S
    if _S is not None:
        return _S
    from juliacall import Main as jl
    jl.seval("using ExaModels, NLPModels")
    _S = SimpleNamespace(
        jl=jl,
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
    return ModelError(head)


def guard(fn, *args, **kwargs):
    """Run `fn`, re-raising backend errors as Python ones (no foreign traceback)."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:                                  # noqa: BLE001
        if type(e).__name__ == "JuliaError":
            raise translate(e) from None
        raise
