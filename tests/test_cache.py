"""The cache's julia-free half: entry resolution, sidecar discrimination, the
record's layout arithmetic, and the wrapper's surface — everything short of a
really-compiled library (which `test_cache_e2e.py` covers).

Every test here must run with Julia never entering the process; the last test
asserts that for the whole module, so a regression that boots Julia anywhere
in the lookup path fails loudly rather than slowly.
"""
import os
import sys

import numpy as np
import pytest

import madsuite as exa
from madsuite import _cache
from madsuite._bridge import ModelError
from madsuite._cache import CachedModel


def _core():
    """A record with every layout-bearing form: two variable blocks (one
    two-dimensional), a parameter, a traced constraint block, and a dims-only
    constraint block."""
    core = exa.Core(cache=True)
    x = core.add_var(4, start=0.5)
    y = core.add_var(2, 3)
    p = core.add_par([1.0, 2.0])
    core.add_obj(lambda i: (x[i] - 1.0) ** 2, over=range(4))
    core.add_con(lambda i: x[i] * 2.0, over=range(4), lcon=0.0, ucon=1.0)
    core.add_con(2, lcon=-1.0, ucon=1.0)
    return core, x, y, p


# -- entry resolution ---------------------------------------------------------

def test_entry_resolution_by_spec(monkeypatch, tmp_path):
    monkeypatch.setenv("MADSUITE_CACHE", str(tmp_path / "root"))
    monkeypatch.setenv("CNLPMODELS_PATH", f"{tmp_path}/a:{tmp_path}/b")
    fp, dd = "f" * 64, "d" * 64
    assert _cache._entries(True, fp, dd) == [
        str(tmp_path / "root" / ("f" * 16 + "d" * 16))]
    assert _cache._entries("@opf", fp, dd) == [
        f"{tmp_path}/a/opf", f"{tmp_path}/b/opf", f"{tmp_path}/a", f"{tmp_path}/b"]
    assert _cache._entries(str(tmp_path / "dir"), fp, dd) == [str(tmp_path / "dir")]
    assert _cache._entries(str(tmp_path / "dir" / "libm.so"), fp, dd) == [
        str(tmp_path / "dir")]


# -- sidecar discrimination ---------------------------------------------------

def _sidecar(entry, fp, dd, **override):
    os.makedirs(entry, exist_ok=True)
    meta = {"format": _cache.FORMAT, "fingerprint": fp, "data_digest": dd,
            "libpath": os.path.join(entry, "libm.so"), "prefix": "m",
            "backend_pin": _cache._pinned_backend()}
    meta.update(override)
    _cache._write_sidecar(entry, meta)
    return meta


def test_sidecar_discrimination(tmp_path):
    """Two-sided, against `_match` (the policy-free lookup — `attach` layers
    the julia-in-process guard on top, and conftest has already booted Julia
    in this process): the true sidecar matches, and each single-field
    corruption unmatches."""
    core, *_ = _core()
    core.cache = str(tmp_path / "e")
    fp, dd = core.fingerprint()
    assert _cache._match(core, fp, dd) is None             # no sidecar at all
    _sidecar(str(tmp_path / "e"), fp, dd)
    assert _cache._match(core, fp, dd) is not None         # the honest one matches
    for wrong in ({"fingerprint": "0" * 64}, {"data_digest": "0" * 64},
                  {"backend_pin": "=0.0.1"}, {"format": "madsuite-cache-v9"}):
        _sidecar(str(tmp_path / "e"), fp, dd, **wrong)
        assert _cache._match(core, fp, dd) is None, f"matched despite {wrong}"


def test_attach_declines_matching_entry_when_julia_owns_the_process(tmp_path):
    """Regression for the SIGABRT of 2026-08-17: a compiled library cannot
    stand up its runtime beside juliacall (thread-TLS adoption aborts), so a
    matching entry in this Julia-running process must yield None — the eager
    path — rather than a load attempt.  conftest guarantees juliacall is
    loaded here; assert that premise so this cannot pass vacuously."""
    import sys
    assert "juliacall" in sys.modules
    core, *_ = _core()
    core.cache = str(tmp_path / "e")
    fp, dd = core.fingerprint()
    _sidecar(str(tmp_path / "e"), fp, dd)                  # a perfect match...
    assert _cache._match(core, fp, dd) is not None
    assert _cache.attach(core) is None                     # ...still declined


# -- layout arithmetic --------------------------------------------------------

def test_record_layout_offsets():
    core, x, y, p = _core()
    var, con, pars, nvar, ncon = _cache._layout(core)
    assert (nvar, ncon) == (10, 6)                         # pipeline produced output
    assert var == {("var", 0): (0, 4, (4,)), ("var", 1): (4, 6, (2, 3))}
    assert con == {("con", 0): (0, 4, (4,)), ("con", 1): (4, 2, (2,))}
    assert len(pars) == 1 and pars[0]["values"].tolist() == [1.0, 2.0]
    assert x._key == ("var", 0) and y._key == ("var", 1) and p._key == ("par", 0)


# -- the wrapper against a stub library ---------------------------------------

class _Ref:
    def __init__(self, index):
        self.index = index


class _StubCM:
    """The slice of the cnlpmodels surface `CachedModel` consumes."""
    nvar, ncon, nnzj, nnzh = 10, 6, 3, 4
    x0 = np.arange(10.0)
    lvar, uvar = np.full(10, -1.0), np.full(10, 9.0)
    lcon, ucon = np.zeros(6), np.ones(6)

    def __init__(self):
        self._pars = {"p0": _Ref(0)}
        self.values = {}

    def set_value(self, ref, v):
        self.values[ref.index] = np.asarray(v, dtype=np.float64).copy()

    def get_value(self, ref):
        return self.values[ref.index]

    def obj(self, x):
        return float(np.sum(x))

    def grad(self, x):
        return np.ones(10)

    def cons(self, x):
        return np.full(6, 0.5)


def test_cached_model_surface():
    core, x, y, p = _core()
    m = CachedModel._load(_StubCM(), core)
    # loading pushed the recorded parameter values — they are outside the digests
    assert m.parameters(p).tolist() == [1.0, 2.0]
    m.set_parameters(p, [3.0, 4.0])
    assert m.parameters(p).tolist() == [3.0, 4.0]
    # metadata and baked data, sliced by the record's own offsets
    assert (m.nvar, m.ncon, m.nnzj, m.nnzh) == (10, 6, 3, 4)
    assert m.get_start(x).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert m.get_start(y).shape == (2, 3)
    assert m.get_ucon(_key_holder(("con", 1))).tolist() == [1.0, 1.0]
    # evaluation goes to the library
    assert m.objective(m.x0) == 45.0
    assert m.constraints(m.x0).tolist() == [0.5] * 6
    assert m.violation(m.x0) == 0.0
    # baked setters refuse — with the reason, not an AttributeError
    with pytest.raises(ModelError, match="baked"):
        m.set_start(x, np.zeros(4))
    with pytest.raises(ModelError, match="baked"):
        m.set_uvar(x, np.ones(4))
    # a solver that would need Julia refuses on a hit
    with pytest.raises(ModelError, match="madnlp"):
        m.solve(solver="madnlp")


def _key_holder(key):
    class H:
        _key = key
    return H()


def test_load_refuses_a_library_that_does_not_fit_its_record():
    core, *_ = _core()
    small = _StubCM()
    small.nvar = 7                                          # record lays out 10
    with pytest.raises(RuntimeError, match="does not fit"):
        CachedModel._load(small, core)


def test_lookup_path_is_julia_free():
    """In a fresh interpreter (conftest boots Julia in THIS one, probing for
    solvers): record, fingerprint, miss, and every sidecar rejection — with
    juliacall never imported."""
    import subprocess
    import textwrap
    script = textwrap.dedent("""
        import json, os, sys, tempfile
        import madsuite as exa
        from madsuite import _cache
        core = exa.Core(cache=True)
        x = core.add_var(4, start=0.5)
        core.add_obj(lambda i: (x[i] - 1.0) ** 2, over=range(4))
        with tempfile.TemporaryDirectory() as d:
            core.cache = d
            assert _cache.attach(core) is None
            fp, dd = core.fingerprint()
            meta = {"format": _cache.FORMAT, "fingerprint": "0" * 64,
                    "data_digest": dd, "libpath": os.path.join(d, "libm.so"),
                    "prefix": "m", "backend_pin": _cache._pinned_backend()}
            _cache._write_sidecar(d, meta)
            assert _cache.attach(core) is None
            # a MATCHING sidecar must raise here either way (julia-free, so
            # the guard does not intercept): a missing library because
            # recompiling over a broken entry would hide it forever, and a
            # missing [cache] extra because a raw ModuleNotFoundError names
            # the wrong problem.  Which error depends on the environment —
            # CI has no extra, dev machines do — so expect its own.
            meta["fingerprint"] = fp
            _cache._write_sidecar(d, meta)
            from importlib.util import find_spec
            want = ("libm.so", "shared library") if find_spec("cnlpmodels") \
                else ("madsuite[cache]",)
            try:
                _cache.attach(core)
            except Exception as e:
                assert any(w in str(e) for w in want), (want, e)
            else:
                raise AssertionError("a broken matching entry was not an error")
        assert "juliacall" not in sys.modules, "lookup path booted Julia"
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# -- recipes (nargs=) ---------------------------------------------------------

def _recipe():
    core = exa.Core(nargs=1, cache=True)
    (n,) = core.args
    x = core.add_var(n, start=1.5)
    p = core.add_par([2.0])
    core.add_obj(lambda i: (x[i] - 2.0) ** 2 + p[0] * x[i], over=exa.srange(0, n))
    core.add_con(lambda i: x[i] + x[i + 1], over=exa.srange(0, n - 1),
                 lcon=0.0, ucon=10.0)
    return core, x, p


def test_argsig_is_types_not_values():
    sig = _cache._argsig
    assert sig((10,)) == sig((50,)) == "i64"                # value-free
    assert sig((10.0,)) == "f64" and sig((10,)) != sig((10.0,))
    assert sig((np.arange(3),)) == "vec:i64"
    assert sig((np.arange(3.0),)) == "vec:f64"
    assert sig(("case.m",)) == "str"
    assert sig((10, np.zeros(2))) == "i64;vec:f64"
    with pytest.raises(TypeError, match="bool"):
        sig((True,))


def test_recipe_entry_key_carries_the_argsig(monkeypatch, tmp_path):
    monkeypatch.setenv("MADSUITE_CACHE", str(tmp_path))
    fp, dd = "f" * 64, "d" * 64
    plain = _cache._entries(True, fp, dd)[0]
    intkey = _cache._entries(True, fp, dd, "i64")[0]
    veckey = _cache._entries(True, fp, dd, "vec:f64")[0]
    assert plain == str(tmp_path / ("f" * 16 + "d" * 16))   # fixed form unchanged
    assert len({plain, intkey, veckey}) == 3


def test_recipe_fingerprint_semantics():
    """Sizes and bounds written against a placeholder are STRUCTURE; the
    argument's value never enters the record at all."""
    base, _, _ = _recipe()
    fp, dd = base.fingerprint()
    assert (fp, dd) == base.fingerprint()                   # stable
    # a different size EXPRESSION is different code
    other = exa.Core(nargs=1, cache=True)
    (n,) = other.args
    x = other.add_var(2 * n, start=1.5)
    p = other.add_par([2.0])
    other.add_obj(lambda i: (x[i] - 2.0) ** 2 + p[0] * x[i], over=exa.srange(0, n))
    other.add_con(lambda i: x[i] + x[i + 1], over=exa.srange(0, n - 1),
                  lcon=0.0, ucon=10.0)
    assert other.fingerprint()[0] != fp
    # a changed literal in a recipe still moves only the data digest
    third, _, _ = _recipe()
    third._records[0]["start"] = 9.9
    fp3, dd3 = third.fingerprint()
    assert fp3 == fp and dd3 != dd


def test_recipe_wrapper_reads_layout_from_the_instance():
    core, x, p = _recipe()
    core.add_con(2, lcon=-1.0, ucon=1.0)                    # dims-only: unpublishable
    stub = _StubCM()
    stub.nvar, stub.ncon = 8, 9
    stub._vars = {"_v0": _Ref(0)}
    stub._cons = {"_c0": _Ref(1)}
    v, c = _Ref(0), _Ref(1)
    v.kind, v.offset, v.length, v.dims = "var", 0, 8, (8,)
    c.kind, c.offset, c.length, c.dims = "con", 0, 7, (7,)
    stub._vars["_v0"], stub._cons["_c0"] = v, c
    m = CachedModel._load(stub, core)
    assert m.get_start(x).shape == (8,)                     # from the instance
    assert m.parameters(p).tolist() == [2.0]
    dims_handle = _key_holder(("con", 1))
    with pytest.raises(TypeError, match="dims-only"):
        m.get_ucon(dims_handle)


def test_recording_a_recipe_is_julia_free():
    import subprocess
    import textwrap
    script = textwrap.dedent("""
        import sys
        import madsuite as exa
        core = exa.Core(nargs=2, cache=True)
        n, w = core.args
        x = core.add_var(n, start=0.5)
        core.add_obj(lambda i: w * (x[i] - 1.0) ** 2, over=exa.srange(0, n))
        fp, dd = core.fingerprint()
        assert (fp, dd) == core.fingerprint()
        assert "juliacall" not in sys.modules, "recipe recording booted Julia"
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
