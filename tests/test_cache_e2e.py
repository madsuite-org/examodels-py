"""The cache round-trip against a really-compiled library.

Slow — one ahead-of-time compile — and environment-gated: it needs the
compiler backend (Julia 1.12; see the install manual's OpenSSL note) and
cyipopt.

Process discipline is the whole subject here.  A compiled library and an
in-process Julia cannot share a process (the library's runtime aborts
adopting a thread that already carries Julia TLS), and conftest boots Julia
in the pytest process to probe for solvers — so the compile happens in one
subprocess, every load happens in other, julia-free subprocesses, and the
one thing tested IN this Julia-running process is the fallback that makes
the constraint invisible to users.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import madsuite as exa
from madsuite import _cache


def _ready():
    from importlib.util import find_spec
    if find_spec("cyipopt") is None or find_spec("cnlpmodels") is None:
        return False
    try:
        return bool(exa.compiler_available())
    except Exception:                                        # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _ready(), reason="compiler backend or cyipopt not installed")

#: One model, built identically everywhere: every layout-bearing kind the
#: compiler meets (vars, a parameter, a traced constraint, a dims-only
#: constraint), with a nonlinearity.
MODEL = textwrap.dedent("""
    def build(exa):
        core = exa.Core(cache=True)
        x = core.add_var(6, start=0.5, lvar=0.0, uvar=2.0)
        p = core.add_par([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        core.add_obj(lambda i: (x[i] - p[i]) ** 2 + exa.exp(x[i]), over=range(6))
        core.add_con(lambda i: x[i] + x[i + 1], over=range(5), lcon=0.0, ucon=3.0)
        core.add_con(2, lcon=-1.0, ucon=1.0)
        return core, x, p
""")
_ns = {}
exec(MODEL, _ns)
_build = _ns["build"]

Z = [0.1, 0.46, 0.82, 1.18, 1.54, 1.9]

import contextlib


@contextlib.contextmanager
def _cache_env(root):
    """Point the cache at `root`, restoring whatever was there before — the
    suite-wide isolation fixture owns the variable otherwise."""
    old = os.environ.get("MADSUITE_CACHE")
    os.environ["MADSUITE_CACHE"] = root
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MADSUITE_CACHE", None)
        else:
            os.environ["MADSUITE_CACHE"] = old



def _sub(body, root, timeout):
    """Run `MODEL + body` in a fresh interpreter; its last stdout line is JSON."""
    env = dict(os.environ, MADSUITE_CACHE=root)
    out = subprocess.run([sys.executable, "-c", MODEL + textwrap.dedent(body)],
                         env=env, capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, (out.stdout[-1000:], out.stderr[-3000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def primed(tmp_path_factory):
    """A compiled cache entry — built in a fresh subprocess, since THIS
    process runs Julia — plus the eager model's own numbers at `Z`."""
    root = str(tmp_path_factory.mktemp("cache"))
    eager = _sub(f"""
        import json, sys
        import madsuite as exa
        core, x, p = build(exa)
        m = exa.Model(core)                # miss: replay + compile + store
        assert type(m).__name__ == "Model", type(m).__name__
        # the compile named the parameter internally; the record is restored,
        # so this same core still fingerprints to the entry just written
        assert core._records[1]["name"] is None
        m2 = exa.Model(core)               # same process: eager fallback, no rebuild
        assert type(m2).__name__ == "Model", type(m2).__name__
        # the MISS run must be fully usable through the recorded handles
        # (replay grafts the eager handles onto them): parameters, a solve,
        # and solution slicing — the exact surface a hit run offers
        assert list(m.parameters(p)) == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        sol = m.solve(print_level=0)
        assert len(sol[x]) == 6 and len(sol.multipliers_L(x)) == 6
        z = {Z}
        print(json.dumps({{"obj": m.objective(z), "grad": list(m.gradient(z)),
                           "cons": list(m.constraints(z))}}))
    """, root, timeout=900)
    return {"root": root, "eager": eager}


def test_miss_compiled_and_wrote_the_entry(primed):
    core, x, p = _build(exa)
    fp, dd = core.fingerprint()
    entry = os.path.join(primed["root"], fp[:16] + dd[:16])
    meta = _cache._read_sidecar(entry)
    assert meta is not None, "no sidecar written"
    assert (meta["fingerprint"], meta["data_digest"]) == (fp, dd)
    assert os.path.isfile(meta["libpath"]), meta["libpath"]


def test_hit_matches_the_eager_model_numerically(primed):
    """rtol 1e-12: the library is the same backend's generated code,
    compiled — identical arithmetic up to codegen reassociation."""
    got = _sub(f"""
        import json, sys
        import madsuite as exa
        core, x, p = build(exa)
        m = exa.Model(core)
        assert type(m).__name__ == "CachedModel", type(m).__name__
        z = {Z}
        print(json.dumps({{"obj": m.objective(z), "grad": list(m.gradient(z)),
                           "cons": list(m.constraints(z)),
                           "julia_free": "juliacall" not in sys.modules}}))
    """, primed["root"], timeout=120)
    assert got["julia_free"] is True
    for key in ("obj", "grad", "cons"):
        np.testing.assert_allclose(got[key], primed["eager"][key], rtol=1e-12)


def test_parameter_change_still_hits_and_is_live(primed):
    """Parameter values are outside both digests: a record with different
    values hits the same entry, and the loaded instance carries the new
    ones — visibly, in the objective."""
    got = _sub(f"""
        import json, sys
        import numpy as np
        import madsuite as exa
        core, x, p = build(exa)
        core._records[1]["values"] = np.full(6, 9.0)
        m = exa.Model(core)
        print(json.dumps({{"kind": type(m).__name__,
                           "pars": list(m.parameters(p)),
                           "obj": m.objective({Z}),
                           "julia_free": "juliacall" not in sys.modules}}))
    """, primed["root"], timeout=120)
    assert got["julia_free"] is True
    assert got["kind"] == "CachedModel", "parameter values leaked into a digest"
    assert got["pars"] == [9.0] * 6
    assert got["obj"] > primed["eager"]["obj"]             # (x-9)^2 territory


def test_any_other_data_change_misses(primed):
    core, x, p = _build(exa)
    core._records[0]["start"] = 0.75                       # baked -> different entry
    fp, dd = core.fingerprint()
    with _cache_env(primed["root"]):
        assert _cache._match(core, fp, dd) is None


def test_julia_process_falls_back_to_eager_without_recompiling(primed):
    """Regression for the SIGABRT of 2026-08-17.  In THIS process — Julia is
    up (assert the premise) — a matching entry must produce the ordinary
    eager model: no library load (which would abort the interpreter, so this
    test going green at all is the point), and no recompile (mtimes prove
    the entry was left alone)."""
    assert "juliacall" in sys.modules
    core, x, p = _build(exa)
    fp, dd = core.fingerprint()
    with _cache_env(primed["root"]):
        meta = _cache._match(core, fp, dd)
        assert meta is not None
        before = os.path.getmtime(meta["libpath"])
        m = exa.Model(core)
        assert type(m).__name__ == "Model", type(m).__name__
        np.testing.assert_allclose(m.objective(Z), primed["eager"]["obj"], rtol=1e-12)
        assert os.path.getmtime(meta["libpath"]) == before, "entry was rebuilt"


def test_fresh_process_hits_solves_and_never_boots_julia(primed):
    got = _sub("""
        import json, sys
        import madsuite as exa
        core, x, p = build(exa)
        m = exa.Model(core)
        assert "juliacall" not in sys.modules, "the hit path booted Julia"
        sol = m.solve(print_level=0, sb="yes")
        print(json.dumps({
            "success": bool(sol.success), "objective": sol.objective,
            "x_first": float(sol[x][0]), "y_len": len(sol.y),
            "viol": float(m.violation(sol.x)),
            "julia_free": "juliacall" not in sys.modules}))
    """, primed["root"], timeout=300)
    assert got["julia_free"] is True
    assert got["success"] is True
    assert got["viol"] < 1e-7
    assert got["y_len"] == 7


# -- recipes: one compiled entry, any instantiation of the same types --------

RECIPE = textwrap.dedent("""
    def build_recipe(exa):
        core = exa.Core(nargs=1, cache=True)
        (n,) = core.args
        x = core.add_var(n, start=1.5)
        p = core.add_par([2.0])
        core.add_obj(lambda i: (x[i] - 2.0) ** 2 + p[0] * x[i], over=exa.srange(0, n))
        core.add_con(lambda i: x[i] + x[i + 1], over=exa.srange(0, n - 1),
                     lcon=0.0, ucon=10.0)
        return core, x, p
""")


def _rsub(body, root, timeout):
    env = dict(os.environ, MADSUITE_CACHE=root)
    out = subprocess.run([sys.executable, "-c", RECIPE + textwrap.dedent(body)],
                         env=env, capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, (out.stdout[-1000:], out.stderr[-3000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def primed_recipe(tmp_path_factory):
    """One compiled recipe entry (built at n=8), plus the eager instance's
    numbers at that size."""
    root = str(tmp_path_factory.mktemp("rcache"))
    eager = _rsub("""
        import json, sys
        import madsuite as exa
        core, x, p = build_recipe(exa)
        m = exa.Model(core, 8)             # miss: replay + compile + store
        assert type(m).__name__ == "Model", type(m).__name__
        # the recipe MISS run, too, is fully usable through recorded handles
        assert list(m.parameters(p)) == [2.0]
        sol = m.solve(print_level=0)
        assert len(sol[x]) == 8
        z = [float(k) / 4 for k in range(8)]
        print(json.dumps({"obj": m.objective(z), "cons": list(m.constraints(z))}))
    """, root, timeout=900)
    return {"root": root, "eager": eager}


def test_recipe_hit_at_the_compiled_size(primed_recipe):
    got = _rsub("""
        import json, sys
        import madsuite as exa
        core, x, p = build_recipe(exa)
        m = exa.Model(core, 8)
        assert type(m).__name__ == "CachedModel", type(m).__name__
        z = [float(k) / 4 for k in range(8)]
        print(json.dumps({"obj": m.objective(z), "cons": list(m.constraints(z)),
                          "julia_free": "juliacall" not in sys.modules}))
    """, primed_recipe["root"], timeout=120)
    assert got["julia_free"] is True
    np.testing.assert_allclose(got["obj"], primed_recipe["eager"]["obj"], rtol=1e-12)
    np.testing.assert_allclose(got["cons"], primed_recipe["eager"]["cons"], rtol=1e-12)


def test_recipe_hit_at_a_different_size(primed_recipe):
    """The property that makes recipe caching the stronger form: the argument's
    VALUE is per-instance, so one compiled entry serves every size."""
    got = _rsub("""
        import json, sys
        import madsuite as exa
        core, x, p = build_recipe(exa)
        m = exa.Model(core, 20)            # never compiled at 20
        assert type(m).__name__ == "CachedModel", type(m).__name__
        sol = m.solve(print_level=0, sb="yes")
        print(json.dumps({"nvar": m.nvar, "ncon": m.ncon,
                          "success": bool(sol.success),
                          "x_len": len(sol[x]), "pars": list(m.parameters(p)),
                          "viol": float(m.violation(sol.x)),
                          "julia_free": "juliacall" not in sys.modules}))
    """, primed_recipe["root"], timeout=300)
    assert got["julia_free"] is True
    assert (got["nvar"], got["ncon"]) == (20, 19)
    assert got["success"] is True and got["viol"] < 1e-7
    assert got["x_len"] == 20
    assert got["pars"] == [2.0]


def test_recipe_argument_of_a_different_type_misses(primed_recipe):
    core = None
    ns = {}
    exec(RECIPE, ns)
    core, x, p = ns["build_recipe"](exa)
    fp, dd = core.fingerprint()
    with _cache_env(primed_recipe["root"]):
        assert _cache._match(core, fp, dd, _cache._argsig((8,))) is not None
        assert _cache._match(core, fp, dd, _cache._argsig((8.0,))) is None
