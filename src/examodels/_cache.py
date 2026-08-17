"""The compiled-library cache — the payoff half of `Core(cache=...)`.

A recorded core resolves here at `Model(core)`:

  HIT  — the sidecar beside a previously compiled library matches this
         record's (fingerprint, data digest): the library is loaded through
         cnlpmodels and wrapped as a `CachedModel`.  No Julia enters the
         process; solves go to Ipopt through cyipopt.
  MISS — the record is replayed through the ordinary eager path (Julia
         boots), and the model is compiled and stored SYNCHRONOUSLY before
         the eager model is returned — predictability over a faster first
         run.  The next identical run hits.

Where an entry lives is the `cache=` argument's business:

  cache=True     — a content-addressed directory under `$EXAMODELS_CACHE`
                   (default `~/.cache/examodels`), named by the digests.
  cache="@name"  — installed on `CNLPMODELS_PATH`, where both consumers
                   find it by that name.
  cache="path"   — that directory (or that library file's), exactly.

Parameter values live outside the digests — the ABI setter keeps them live —
so a hit pushes this record's values into the loaded instance.  Everything
else is baked: any other data change is a different entry.

The sidecar also pins the ExaModels.jl version the library was compiled
against (read julia-free from the packaged `juliapkg.json`): a library from a
different pin is a miss, since the generated derivative code is that
package's output.
"""
import hashlib
import json
import os

import numpy as np

__all__ = ["CachedModel", "CachedSolution"]

SIDECAR = "examodels-cache.json"
FORMAT = "examodels-cache-v0"


def _root():
    return os.environ.get("EXAMODELS_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "examodels")


def _search_dirs():
    """The CNLPMODELS_PATH directories, as cnlpmodels itself would read them."""
    env = os.environ.get("CNLPMODELS_PATH", "")
    return [d for d in env.split(":") if d]


def _pinned_backend():
    """The ExaModels.jl version this package pins, read without Julia."""
    from importlib.resources import files
    meta = json.loads((files("examodels") / "juliapkg.json").read_text())
    return meta["packages"]["ExaModels"]["version"]


_EXTS = (".so", ".dylib", ".dll")


def _argsig(args):
    """The instantiation values' TYPE signature — structure, not data.

    The compiler bakes each example's type into the library while the value
    stays a per-instance input, so two runs whose arguments differ only in
    VALUE share an entry, and two whose types differ do not.  The judgments
    mirror `compile._example`'s: 64-bit numbers, 1-D arrays of them, tables
    of named columns, and strings (which reach an argfun, never storage)."""
    from .node import _columns, is_table
    sig = []
    for v in args:
        if isinstance(v, (bool, np.bool_)):
            raise TypeError("a bool is not a model argument; say 0 or 1")
        if isinstance(v, (int, np.integer)):
            sig.append("i64")
        elif isinstance(v, (float, np.floating)):
            sig.append("f64")
        elif isinstance(v, str):
            sig.append("str")
        elif is_table(v):
            fields, cols, _n = _columns(v)
            sig.append("table:" + ",".join(
                f"{f}:{c.dtype}" for f, c in zip(fields, cols)))
        else:
            a = np.asarray(v)
            if a.ndim != 1 or a.dtype.kind not in "iuf":
                raise TypeError(
                    f"an argument may be a number, a 1-D array of numbers, a "
                    f"table of named rows, or a string; got {v!r}")
            sig.append("vec:i64" if a.dtype.kind in "iu" else "vec:f64")
    return ";".join(sig)


def _entries(spec, fp, dd, sig=""):
    """Candidate entry directories for a `cache=` spec, most specific first.

    An entry directory is wherever a sidecar may sit; for `cache=True` it is
    also unique per (fingerprint, digest, argument signature), which is what
    makes that form content-addressed."""
    if spec is True:
        tag = "-" + hashlib.sha256(sig.encode()).hexdigest()[:8] if sig else ""
        return [os.path.join(_root(), fp[:16] + dd[:16] + tag)]
    s = os.fspath(spec)
    if s.startswith("@"):
        name = s[1:]
        return [os.path.join(d, name) for d in _search_dirs()] + _search_dirs()
    if s.endswith(_EXTS):
        return [os.path.dirname(os.path.abspath(s))]
    return [os.path.abspath(s)]


def _read_sidecar(entry):
    try:
        with open(os.path.join(entry, SIDECAR)) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("format") == FORMAT else None


def _write_sidecar(entry, meta):
    """Written last and renamed into place: a crashed compile leaves no
    sidecar, so a partial entry can never match."""
    tmp = os.path.join(entry, SIDECAR + ".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(entry, SIDECAR))


# -- the record's layout ------------------------------------------------------

def _over_len(desc):
    kind = desc[0]
    if kind == "range":
        return desc[2] - desc[1] + 1
    if kind == "steprange":
        return len(range(desc[1], desc[2], desc[3]))
    if kind == "product":
        n = 1
        for lo, hi in desc[1]:
            n *= hi - lo + 1
        return n
    return len(desc[3][0])                                  # table: any column


def _axes_len(axes):
    n = 1
    for lo, hi in axes:
        n *= hi - lo + 1
    return n


def _layout(core):
    """key -> (offset, length, dims) for variables and constraints, plus the
    parameter records in declaration order.

    The compiled library lays blocks out in the order they were declared —
    the same order the replay adds them — so offsets are a pure function of
    the record."""
    var, con, par = {}, {}, []
    voff = coff = 0
    for r in core._records:
        kind = r["kind"]
        if kind == "var":
            n = _axes_len(r["axes"])
            dims = tuple(hi - lo + 1 for lo, hi in r["axes"])
            var[("var", r["ordinal"])] = (voff, n, dims)
            voff += n
        elif kind == "con":
            n = _over_len(r["over"])
            con[("con", r["ordinal"])] = (coff, n, (n,))
            coff += n
        elif kind == "con_dims":
            n = _axes_len(r["axes"])
            dims = tuple(hi - lo + 1 for lo, hi in r["axes"])
            con[("con", r["ordinal"])] = (coff, n, dims)
            coff += n
        elif kind == "par":
            par.append(r)
    return var, con, par, voff, coff


# -- lookup (the julia-free path) ---------------------------------------------

def _match(core, fp, dd, sig=""):
    """The first sidecar matching this record — the pure lookup, no policy."""
    for entry in _entries(core.cache, fp, dd, sig):
        meta = _read_sidecar(entry)
        if (meta is not None and meta["fingerprint"] == fp
                and meta["data_digest"] == dd
                and meta.get("argsig", "") == sig
                and meta.get("backend_pin") == _pinned_backend()):
            return meta
    return None


def attach(core, args=(), fpdd=None):
    """The cached model for this record (instantiated with `args`, for a
    recipe), or None.

    None sends the caller down the eager path, for either of two reasons: a
    genuine miss (no sidecar, or digests or backend pin differ), or a
    matching entry in a process Julia already owns.  A compiled library
    stands up its own Julia runtime on the first call into it, and a thread
    that already carries the host Julia's TLS aborts the whole process in
    `jl_adopt_thread` (measured: the same library loads fine julia-free and
    SIGABRTs beside juliacall).  Julia being up also means the JIT is already
    paid for, and `materialize` sees the matching sidecar and skips the
    recompile — so the fallback costs a replay, not a build.

    A matching sidecar whose library will not load is an error, not a miss:
    silently recompiling over a broken entry would hide it forever."""
    from ._bridge import loaded
    fp, dd = core.fingerprint() if fpdd is None else fpdd
    meta = _match(core, fp, dd, _argsig(args))
    if meta is None or loaded():
        return None
    try:
        import cnlpmodels
    except ImportError:
        from ._bridge import ModelError
        raise ModelError(
            "a compiled cache entry matches this model, but loading it needs "
            "the [cache] extra (cnlpmodels + cyipopt): pip install "
            "\"examodels[cache]\". Or build without cache= to skip the "
            "cache entirely.") from None
    cm = cnlpmodels.CModel(meta["libpath"], *args, prefix=meta["prefix"])
    return CachedModel._load(cm, core)


# -- store (the miss path; Julia is already up) -------------------------------

def materialize(core, args=(), fpdd=None):
    """Replay the record, compile it, write the entry, return the eager core.

    Synchronous by design: the first run grows by the compile time, and in
    exchange what happened is never in doubt.  A compile failure propagates —
    the caller asked for a cache and did not get one.  For a recipe, `args`
    are this run's instantiation values, which double as the compiler's
    examples: their types are baked, their values stay per-instance."""
    from ._bridge import ModelError
    from .compile import compile_library, compiler_available
    fp, dd = core.fingerprint() if fpdd is None else fpdd
    sig = _argsig(args)
    # The compiler publishes only NAMED blocks, and everything the wrapper
    # addresses on a loaded library goes through that layout — parameters
    # always (the ABI setter), and for a recipe every block, since sizes and
    # offsets exist only per instance.  So the compile replay names what the
    # user did not: parameters on any core, all nameable blocks on a recipe.
    # Set-and-restore rather than keep: the name is part of the structural
    # fingerprint, and the record must stay identical to what the sidecar
    # was keyed on.  (A dims-only constraint block has no name in the eager
    # surface, so it stays unpublished; slicing its handle on a hit refuses.)
    prefixes = {"var": "_v", "con": "_c", "par": "_p"}
    kinds = ("var", "con", "par") if core._nargs else ("par",)
    unnamed = [r for r in core._records
               if r["kind"] in kinds and r["name"] is None]
    for r in unnamed:
        r["name"] = f"{prefixes[r['kind']]}{r['ordinal']}"
    try:
        eager = core.replay()
    finally:
        for r in unnamed:
            r["name"] = None
    if _match(core, fp, dd, sig) is not None:
        # The entry already exists; we are on the eager path only because
        # this process already runs Julia (or raced another build).  Nothing
        # to store, and no compiler needed.
        return eager
    if not compiler_available():
        raise ModelError(
            "Core(cache=...) needs the compiler backend to store the model, "
            "and it is not installed in this environment; run "
            "examodels.install_compiler() once. (It needs Julia 1.12, which "
            "juliapkg only installs when Python links OpenSSL >= 3.5 — see "
            "the install manual.)")
    spec = core.cache
    if spec is True:
        entry = _entries(True, fp, dd, sig)[0]
        os.makedirs(entry, exist_ok=True)
        lib = compile_library(os.path.join(entry, "m"), eager, *args)
    elif os.fspath(spec).startswith("@"):
        lib = compile_library(os.fspath(spec), eager, *args)
        entry = lib.outdir
    else:
        s = os.fspath(spec)
        entry = os.path.dirname(os.path.abspath(s)) if s.endswith(_EXTS) \
            else os.path.abspath(s)
        os.makedirs(entry, exist_ok=True)
        lib = compile_library(os.path.join(entry, os.path.basename(entry)),
                              eager, *args)
    _write_sidecar(entry, {
        "format": FORMAT, "fingerprint": fp, "data_digest": dd, "argsig": sig,
        "libpath": os.path.abspath(lib.path), "prefix": lib.prefixes[0],
        "backend_pin": _pinned_backend(),
    })
    return eager


def _published_layout(cm, core):
    """key -> (offset, length, dims), read from the library's own layout.

    Every nameable block was named into the compile (`materialize`'s
    synthetic names), so the lookup is by those same names.  A dims-only
    constraint block has no name in the eager surface and so no published
    entry; its keys come back separately, for slicing to refuse."""
    var, con, unpublished = {}, {}, set()
    for r in core._records:
        kind = r["kind"]
        if kind == "con_dims":
            unpublished.add(("con", r["ordinal"]))
            continue
        if kind not in ("var", "con"):
            continue
        name = r["name"] or ("_v" if kind == "var" else "_c") + str(r["ordinal"])
        b = (cm._vars if kind == "var" else cm._cons).get(name)
        if b is None:
            raise RuntimeError(
                f"cache entry does not fit its own record: the library "
                f"publishes no block named {name!r} — delete the entry")
        dims = tuple(b.dims) if len(b.dims) > 1 else (b.length,)
        (var if kind == "var" else con)[(kind, r["ordinal"])] = \
            (b.offset, b.length, dims)
    return var, con, unpublished


# -- the wrapper --------------------------------------------------------------

from .model import Model  # noqa: E402  (module import is julia-free; only _b attributes boot)


class CachedModel(Model):
    """The `Model` surface served from a compiled library — no Julia.

    Evaluation and metadata come from the library through cnlpmodels; named
    blocks and solution slicing come from the record, whose declaration order
    fixes every block's offset.  Starts and bounds are baked into the library
    (they are part of the data digest), so their setters refuse; parameter
    values stay live through the ABI setter."""

    def __init__(self, *args, **kwargs):
        # Python re-invokes __init__ after `Model.__new__` returned the
        # already-built instance of a cache hit; there is nothing left to do.
        if "_cm" not in self.__dict__:
            raise TypeError(
                "CachedModel is built by the cache lookup — write "
                "Model(core) on a Core(cache=...)")

    @classmethod
    def _load(cls, cm, core):
        self = object.__new__(cls)
        self._cm = cm
        self._named = dict(core._named)
        self._unpublished = set()
        if core._nargs:
            # a recipe's sizes exist only per instance, so the loaded
            # instance's own layout is the authority
            self._var, self._con, self._unpublished = _published_layout(cm, core)
            nvar = sum(v[1] for v in self._var.values())
            if nvar != cm.nvar:
                raise RuntimeError(
                    f"cache entry does not fit its own record: the instance "
                    f"has {cm.nvar} variables where the published blocks "
                    f"cover {nvar} — delete the entry")
        else:
            self._var, self._con, _, nvar, ncon = _layout(core)
            if (cm.nvar, cm.ncon) != (nvar, ncon):
                raise RuntimeError(
                    f"cache entry does not fit its own record: the library has "
                    f"{cm.nvar} variables / {cm.ncon} constraints where the "
                    f"record laid out {nvar} / {ncon} — delete the entry")
        pars = [r for r in core._records if r["kind"] == "par"]
        self._parref = self._match_pars(cm, pars)
        for r, ref in zip(pars, self._parref):
            cm.set_value(ref, r["values"])                 # live, per instance
        return self

    @staticmethod
    def _match_pars(cm, pars):
        """The library's parameter BlockRefs, in the record's declaration
        order (the library numbers them in the same order it met them)."""
        refs = sorted(cm._pars.values(), key=lambda b: b.index)
        if len(refs) != len(pars):
            raise RuntimeError(
                f"cache entry does not fit its own record: the library "
                f"publishes {len(refs)} parameter blocks where the record "
                f"has {len(pars)} — delete the entry")
        return refs

    # -- metadata and named blocks -------------------------------------------
    _META = frozenset({"nvar", "ncon", "nnzj", "nnzh",
                       "x0", "lvar", "uvar", "lcon", "ucon"})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        named = self.__dict__.get("_named")
        if named and name in named:
            return named[name]
        if name in self._META:
            return getattr(self._cm, name)
        raise AttributeError(f"'CachedModel' has no attribute {name!r}")

    def __dir__(self):
        return sorted({*object.__dir__(self), *self._META, *self._named})

    # -- evaluation ------------------------------------------------------------
    def _hx(self, x):
        return np.ascontiguousarray(x, dtype=np.float64)

    def objective(self, x):
        return float(self._cm.obj(self._hx(x)))

    def gradient(self, x):
        return self._cm.grad(self._hx(x))

    def constraints(self, x):
        return self._cm.cons(self._hx(x))

    # violation() is inherited: it only uses constraints/lcon/ucon.

    # -- parameters ------------------------------------------------------------
    def _par(self, block):
        key = getattr(block, "_key", None)
        if not key or key[0] != "par":
            raise TypeError(f"{block!r} is not a parameter block of this model")
        return self._parref[key[1]]

    def parameters(self, block):
        return np.asarray(self._cm.get_value(self._par(block)), dtype=np.float64)

    def set_parameters(self, block, values):
        self._cm.set_value(self._par(block), np.asarray(values, dtype=np.float64).ravel())
        return self

    get_value, set_value = parameters, set_parameters

    # -- baked data: readable, not writable ------------------------------------
    def _slice(self, table, vec, handle):
        key = getattr(handle, "_key", None)
        if key in self._unpublished:
            raise TypeError(
                "a dims-only constraint block cannot be addressed on a recipe "
                "cache hit — the eager surface cannot name it into the "
                "library's layout. Read the whole vector and slice it, or "
                "build without cache=.")
        if key not in table:
            raise TypeError(f"{handle!r} is not a block of this model")
        off, n, dims = table[key]
        v = vec[off:off + n]
        return v.reshape(dims) if len(dims) > 1 else v

    def get_start(self, h):
        return self._slice(self._var, self._cm.x0, h)

    def get_lvar(self, h):
        return self._slice(self._var, self._cm.lvar, h)

    def get_uvar(self, h):
        return self._slice(self._var, self._cm.uvar, h)

    def get_lcon(self, h):
        return self._slice(self._con, self._cm.lcon, h)

    def get_ucon(self, h):
        return self._slice(self._con, self._cm.ucon, h)

    def _baked(self, *_a, **_k):
        from ._bridge import ModelError
        raise ModelError(
            "starts and bounds are baked into a compiled cache entry — they "
            "are data, so a different value is a different entry. Change the "
            "recording core and rebuild (one recompile), or build without "
            "cache= to keep them live.")

    set_start = set_lvar = set_uvar = set_lcon = set_ucon = _baked

    # -- solving ---------------------------------------------------------------
    def solve(self, solver=None, **options):
        if solver not in (None, "ipopt"):
            from ._bridge import ModelError
            raise ModelError(
                f"a cache-hit model solves with Ipopt through cyipopt; "
                f"{solver!r} would need Julia, which a hit never boots. "
                f"Build without cache= to use it.")
        from cnlpmodels import solve_ipopt
        import time
        t0 = time.perf_counter()
        x, info = solve_ipopt(self._cm, **options)
        return CachedSolution(x, info, self._var, self._con,
                              unpublished=self._unpublished,
                              elapsed=time.perf_counter() - t0)

    def __repr__(self):
        return (f"<CachedModel nvar={self.nvar} ncon={self.ncon} "
                f"nnzj={self.nnzj} nnzh={self.nnzh}>")


class CachedSolution:
    """The `Solution` surface, from cyipopt's `(x, info)` pair.

    Ipopt's C return codes are spelled with the names `Solution` already
    treats as success, so `.success` means the same thing on both paths."""

    _STATUS = {0: "SOLVE_SUCCEEDED", 1: "SOLVED_TO_ACCEPTABLE_LEVEL",
               2: "INFEASIBLE_PROBLEM_DETECTED", 4: "DIVERGING_ITERATES",
               5: "USER_REQUESTED_STOP", -1: "MAXIMUM_ITERATIONS_EXCEEDED",
               -2: "RESTORATION_FAILED", -4: "MAXIMUM_CPUTIME_EXCEEDED"}

    __slots__ = ("_x", "_info", "_var", "_con", "_unpublished", "elapsed")

    def __init__(self, x, info, var, con, unpublished=(), elapsed=float("nan")):
        self._x, self._info = np.asarray(x, dtype=np.float64), info
        self._var, self._con = var, con
        self._unpublished = unpublished
        self.elapsed = elapsed

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return np.asarray(self._info["mult_g"], dtype=np.float64)

    @property
    def objective(self):
        return float(self._info["obj_val"])

    @property
    def status(self):
        code = int(self._info["status"])
        return self._STATUS.get(code, f"IPOPT_{code}")

    @property
    def success(self):
        from .model import Solution
        return self.status in Solution._SUCCESS

    @property
    def iterations(self):
        raise AttributeError(
            "cyipopt does not report an iteration count; read the Ipopt log "
            "(print_level) if you need one")

    def _slice(self, table, vec, handle, what):
        key = getattr(handle, "_key", None)
        if key in self._unpublished:
            raise TypeError(
                "a dims-only constraint block cannot be addressed on a recipe "
                "cache hit — the eager surface cannot name it into the "
                "library's layout. Read the whole vector and slice it, or "
                "build without cache=.")
        if key not in table:
            raise TypeError(f"{handle!r} is not a {what} block of this solution's model")
        off, n, dims = table[key]
        v = np.asarray(vec, dtype=np.float64)[off:off + n]
        return v.reshape(dims) if len(dims) > 1 else v

    def __getitem__(self, block):
        return self._slice(self._var, self._x, block, "variable")

    def multipliers(self, constraint):
        return self._slice(self._con, self._info["mult_g"], constraint, "constraint")

    def multipliers_L(self, block):
        return self._slice(self._var, self._info["mult_x_L"], block, "variable")

    def multipliers_U(self, block):
        return self._slice(self._var, self._info["mult_x_U"], block, "variable")

    def __repr__(self):
        return (f"<CachedSolution status={self.status!r} "
                f"objective={self.objective:.6g}>")
