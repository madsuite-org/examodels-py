"""COPS benchmark models, checked against COPSBenchmark.jl.

The oracle is the Julia model itself: same sizes, same objective at the same point,
same constraint values. That catches a mis-transcribed index or a dropped term,
which a pinned objective would not.
"""
import pathlib
import sys

import numpy as np
import pytest

import madsuite as exa
from madsuite import _bridge as _b

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))

try:
    _b.seval("using COPSBenchmark")
    HAVE_COPS = True
except Exception:                                            # noqa: BLE001
    HAVE_COPS = False

cops = pytest.mark.skipif(not HAVE_COPS, reason="COPSBenchmark.jl not installed")

CASES = [
    ("goddard", lambda n: __import__("cops").goddard(n)[0],
     "COPSBenchmark.rocket_model(COPSBenchmark.ExaModelsBackend(), {n})", 50),
    ("minsurf", lambda n: __import__("cops").minsurf(n)[0],
     "COPSBenchmark.minsurf_model(COPSBenchmark.ExaModelsBackend(), {n}, {n})", 20),
    ("catmix", lambda n: __import__("cops").catmix(n)[0],
     "COPSBenchmark.catmix_model(COPSBenchmark.ExaModelsBackend(), {n})", 20),
]


@cops
@pytest.mark.parametrize("name, build, jl, n", CASES, ids=[c[0] for c in CASES])
def test_matches_the_julia_model(name, build, jl, n):
    jm = _b.seval(jl.format(n=n))
    pm = exa.Model(build(n))

    assert (pm.nvar, pm.ncon) == (int(_b.seval("m -> m.meta.nvar")(jm)),
                                  int(_b.seval("m -> m.meta.ncon")(jm)))

    x0 = np.array(_b.seval("m -> Array(m.meta.x0)")(jm), dtype=np.float64)
    assert pm.objective(x0) == pytest.approx(
        float(_b.seval("(m, x) -> ExaModels.obj(m, x)")(jm, x0)), rel=1e-12)

    jc = np.array(_b.seval(
        "(m, x) -> (c = similar(x, m.meta.ncon); ExaModels.cons!(m, x, c); Array(c))")(jm, x0))
    # sorted: the two may order rows differently, which is not a difference in the model
    np.testing.assert_allclose(np.sort(pm.constraints(x0)), np.sort(jc), atol=1e-10)

    for field in ("lcon", "ucon", "lvar", "uvar"):
        jv = np.array(_b.seval(f"m -> Array(m.meta.{field})")(jm))
        np.testing.assert_allclose(np.sort(getattr(pm, field)), np.sort(jv), atol=1e-10)


@cops
def test_elec_matches_from_the_same_starting_point():
    """elec starts from a random scatter, so take Julia's to compare like with like."""
    from cops import elec
    jm = _b.seval("COPSBenchmark.elec_model(COPSBenchmark.ExaModelsBackend(), 20)")
    x0 = np.array(_b.seval("m -> Array(m.meta.x0)")(jm), dtype=np.float64)
    core, _ = elec(20, start=(x0[:20], x0[20:40], x0[40:60]))
    pm = exa.Model(core)
    assert (pm.nvar, pm.ncon) == (60, 20)
    assert pm.objective(x0) == pytest.approx(
        float(_b.seval("(m, x) -> ExaModels.obj(m, x)")(jm, x0)), rel=1e-12)


@pytest.mark.parametrize("name, n, expect", [
    ("goddard", 200, 1.0), ("minsurf", 20, 2.0), ("elec", 20, 100.0), ("catmix", 20, 0.0),
])
def test_each_model_solves(name, n, expect):
    import cops
    core, _ = getattr(cops, name)(n)
    model = exa.Model(core)
    sol = model.solve()
    assert sol.success, f"{name}: {sol.status}"
    assert model.violation(sol.x) < 1e-6
    assert np.isfinite(sol.objective)
