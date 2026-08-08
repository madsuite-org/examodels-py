"""Recorded model construction — `Tape`, the recording sibling of `Core`.

A `Tape` records the same `add_var` / `add_con` / `add_obj` calls a `Core`
executes, against a *data template* whose sizes are symbolic. The recorded
tape is instantiated against actual data — of any size matching the template's
schema — producing an ordinary `Core` ready for `Model`/`solve`.

Conventions are Core's: indices are 0-based numbers, index sets are half-open
and `range`-shaped. A static set is a `range`; a data-derived one is
`srange(lo, hi)`, which means exactly what `range(lo, hi)` means except that
its bounds may involve `tape.data` values.

PROTOTYPE (rune/tape): single-stage models, lambda-traced expressions,
scalar template fields. Two-stage and generator-expression sugar follow the
design review. Solution read-back (`sol[x]`) resolves against the tape's
most recent `instantiate()`; Block-style handles (shapes, bounds-checked indexing)
follow the review as well.
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


class _TapeHandle(Node):
    """Tape variable handle: a `Node` whose solution read-back is flat.
    (`Solution._block` probes `.shape`; a class attribute answers before
    `Node.__getattr__` would forward the probe to Julia. Block-style shaped
    handles follow the design review.)"""
    __slots__ = ()
    shape = None


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
    """Records a model against symbolic data; `instantiate(**data)` makes a `Core`.

        tape = Tape(N=4)                      # template: schema only
        x = tape.add_var(tape.data.N, start=-0.5)
        tape.add_con(lambda i: x[i] + x[i+1], over=srange(0, tape.data.N - 1))
        core = tape.instantiate(N=1000)            # any size, real model
    """

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
        return _TapeHandle(self._add(_b.EM.add_var, dims, **kw))

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
        import re
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
        ident = r"[A-Za-z_][A-Za-z0-9_]*"
        if not re.fullmatch(ident, prefix) or not re.fullmatch(ident, fname):
            raise ValueError(
                f"prefix and template field must be C identifiers, got {prefix!r}/{fname!r}"
            )

        import juliapkg
        fd, jls = tempfile.mkstemp(suffix=".jls")
        os.close(fd)
        try:
            serialize = _b.seval(
                "(t, p) -> ExaModels.Serialization.serialize(pyconvert(String, p), t)"
            )
            serialize(self._tape, jls)
            outdir = os.path.abspath(out)
            # User-controlled values travel as ARGS, never interpolated into code.
            code = (
                "using ExaModels, JuliaC, Serialization; "
                "tape = deserialize(ARGS[1]); "
                f"r = compile_library(tape; template = (; {fname} = {int(fval)}), "
                f"prefix = ARGS[3], out = ARGS[2], template_n = {int(fval)}, "
                f"verbose = {str(bool(verbose)).lower()}); "
                "println(r.libpath)"
            )
            res = subprocess.run(
                [juliapkg.executable(), "--project=" + proj, "-e", code,
                 "--", jls, outdir, prefix],
                capture_output=True, text=True,
            )
        finally:
            os.unlink(jls)
        if res.returncode != 0:
            raise RuntimeError(
                "shared-library build failed:\n" + res.stdout[-2000:] + res.stderr[-2000:]
            )
        return res.stdout.strip().splitlines()[-1]

    def instantiate(self, *args, **fields):
        """Instantiate the tape: by name (`tape.instantiate(N=1000)`), as one
        bare value for a single-field schema (`tape.instantiate(1000)`), or
        with nothing at all when the tape never touched the data template."""
        if args and fields:
            raise TypeError("pass either one positional value or keyword fields, not both")
        core = Core.__new__(Core)
        if fields:
            if set(fields) != set(self._template):
                raise TypeError(
                    "instantiate() takes exactly the template's fields "
                    f"{sorted(self._template)}; got {sorted(fields) or '{}'} "
                    "(template values are schema placeholders, never defaults)"
                )
            h = _helpers()
            core._core = _b.guard(
                _b.EM.instantiate, self._tape,
                h["mk_nt"](list(fields), list(fields.values())),
            )
        elif args:
            if len(args) != 1 or len(self._template) != 1:
                raise TypeError(
                    "the positional form takes exactly one bare value, for a "
                    f"single-field schema; this template has fields {sorted(self._template)}"
                )
            core._core = _b.guard(_b.EM.instantiate, self._tape, args[0])
        else:
            if self._template:
                raise TypeError(
                    f"this tape's schema has fields {sorted(self._template)}; "
                    "instantiate with them"
                )
            core._core = _b.guard(_b.EM.instantiate, self._tape)
        core._named = {}
        return core
