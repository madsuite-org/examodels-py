"""Recorded model construction — `Tape`, the recording sibling of `Core`.

A `Tape` records the same `add_var` / `add_con` / `add_obj` calls a `Core`
executes, against a *data template* whose sizes are symbolic. The recorded
tape is replayed against actual data — of any size matching the template's
schema — producing an ordinary `Core` ready for `Model`/`solve`.

Conventions are Core's: indices are 0-based numbers, index sets are half-open
and `range`-shaped. A static set is a `range`; a data-derived one is
`srange(lo, hi)`, which means exactly what `range(lo, hi)` means except that
its bounds may involve `tape.data` values.

PROTOTYPE (rune/tape): single-stage models, lambda-traced expressions,
scalar template fields. Two-stage and generator-expression sugar follow the
design review.
"""
from . import _bridge as _b
from .core import Core
from .node import Node

__all__ = ["Tape", "srange"]

_H = None


def _helpers():
    global _H
    if _H is None:
        _H = {
            "colon": _b.seval("(a, b) -> (:)(a, b)"),
            "mk_nt": _b.seval(
                "(ks, vs) -> (; (Symbol(pyconvert(String, k)) => pyconvert(Any, v)"
                " for (k, v) in zip(ks, vs))...)"
            ),
        }
    return _H


class _SRange:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi


def srange(lo, hi):
    """Half-open 0-based index set, like `range(lo, hi)`, whose bounds may be
    data-derived: `srange(0, tape.data.N - 2)`."""
    return _SRange(lo, hi)


def _jl(v):
    return v._jl if isinstance(v, Node) else v


def _minus_one(v):
    return (v - 1) if isinstance(v, Node) else v - 1


class _Data:
    """`tape.data.N` — symbolic handles onto the data template's fields."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer):
        self._tracer = tracer

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return Node(_b.getfield_(self._tracer, name))


class Tape:
    """Records a model against symbolic data; `replay(**data)` makes a `Core`.

        tape = Tape(N=4)                      # template: schema only
        x = tape.add_var(tape.data.N, start=-0.5)
        tape.add_con(lambda i: x[i] + x[i+1], over=srange(0, tape.data.N - 1))
        core = tape.replay(N=1000)            # any size, real model
    """

    nscen = 0

    def __init__(self, **template):
        h = _helpers()
        self._tape = _b.guard(_b.EM.ExaTape)
        self._template = dict(template)
        self.data = _Data(
            _b.guard(_b.EM.DataTracer, h["mk_nt"](list(template), list(template.values())))
        )

    def _add(self, fn, *args, **kw):
        self._tape, out = _b.guard(fn, self._tape, *args, **kw)
        return out

    def _over(self, over):
        h = _helpers()
        if isinstance(over, _SRange):
            return h["colon"](_jl(over.lo), _jl(_minus_one(over.hi)))
        if isinstance(over, range):
            if over.step != 1:
                raise ValueError("index sets are contiguous; got a stepped range")
            return h["colon"](over.start, over.stop - 1)
        raise TypeError(
            "an index set is a range (static) or srange(lo, hi) (data-derived); "
            f"got {type(over).__name__}"
        )

    def _trace(self, f):
        return _jl(f(Node(_b.EM.DataSource())))

    def add_var(self, n, *, start=None, lvar=None, uvar=None):
        """A block of `n` variables indexed 0-based: `x[0]` … `x[n-1]`.
        `n` may be data-derived (`tape.add_var(tape.data.N)`)."""
        h = _helpers()
        kw = {k: _jl(v) for k, v in
              (("start", start), ("lvar", lvar), ("uvar", uvar)) if v is not None}
        dims = h["colon"](0, _jl(_minus_one(n)))
        return Node(self._add(_b.EM.add_var, dims, **kw))

    def add_con(self, f, over, *, lcon=None, ucon=None):
        kw = {k: _jl(v) for k, v in (("lcon", lcon), ("ucon", ucon)) if v is not None}
        return self._add(_b.EM.add_con, self._trace(f), self._over(over), **kw)

    def add_obj(self, f, over):
        return self._add(_b.EM.add_obj, self._trace(f), self._over(over))

    def compile(self, *, prefix="rec", out="lib_out", julia_project=None, verbose=False):
        """Compile the recorded tape into a self-contained shared library
        exposing the model through a C ABI (consumable with `cnlpmodels`,
        no Julia needed on the consumer side).

        The build runs in a Julia project providing ExaModels and JuliaC
        (`julia_project`, or `$EXAMODELS_COMPILE_PROJECT`); the tape itself
        crosses over serialized, so no Julia source is written. Currently
        limited to single-integer-field templates. Returns the library path.
        """
        import os
        import subprocess
        import tempfile

        proj = julia_project or os.environ.get("EXAMODELS_COMPILE_PROJECT")
        if not proj:
            raise RuntimeError(
                "tape.compile() needs a Julia project with ExaModels + JuliaC: "
                "pass julia_project= or set EXAMODELS_COMPILE_PROJECT"
            )
        if len(self._template) != 1 or not isinstance(next(iter(self._template.values())), int):
            raise ValueError("tape.compile() currently needs a single integer-field template")
        (fname, fval), = self._template.items()

        fd, jls = tempfile.mkstemp(suffix=".jls")
        os.close(fd)
        serialize = _b.seval(
            "(t, p) -> ExaModels.Serialization.serialize(pyconvert(String, p), t)"
        )
        serialize(self._tape, jls)
        outdir = os.path.abspath(out)
        code = (
            "using ExaModels, JuliaC, Serialization; "
            f'tape = deserialize("{jls}"); '
            f"r = compile_library(tape; template = (; {fname} = {int(fval)}), "
            f'prefix = "{prefix}", out = "{outdir}", template_n = {int(fval)}, '
            f"verbose = {str(bool(verbose)).lower()}); "
            "println(r.libpath)"
        )
        res = subprocess.run(["julia", "--project=" + proj, "-e", code],
                             capture_output=True, text=True)
        os.unlink(jls)
        if res.returncode != 0:
            raise RuntimeError(
                "shared-library build failed:\n" + res.stdout[-2000:] + res.stderr[-2000:]
            )
        return res.stdout.strip().splitlines()[-1]

    def replay(self, **data):
        h = _helpers()
        merged = {**self._template, **data}
        core = Core.__new__(Core)
        core._core = _b.guard(
            _b.EM.replay, self._tape, h["mk_nt"](list(merged), list(merged.values()))
        )
        core._named = {}
        return core
