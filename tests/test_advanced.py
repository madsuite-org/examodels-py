"""Oracles, two-stage models, tags and model wrappers."""
from collections import namedtuple

import numpy as np
import pytest

import examodels as exa


# -------------------------------------------------------------- oracles ------
def unit_circle_oracle():
    def f(c, x):
        c[0] = x[0]**2 + x[1]**2 - 1.0

    def jac(v, x):
        v[0], v[1] = 2 * x[0], 2 * x[1]

    def hess(v, x, y):
        v[0] = v[1] = 2 * y[0]

    return exa.VectorNonlinearOracle(
        nvar=2, ncon=1, f=f, jac=jac, hess=hess,
        jac_rows=[1, 1], jac_cols=[1, 2], hess_rows=[1, 2], hess_cols=[1, 2],
        lcon=[0.0], ucon=[0.0])


def test_vector_oracle_constrains_the_solution():
    """A constraint block evaluated entirely by Python callbacks."""
    core = exa.Core()
    x = exa.add_var(core, 2, start=0.5)
    exa.add_obj(core, lambda i: -x[0] - x[1], over=range(1))
    exa.add_con(core, unit_circle_oracle())
    model = exa.Model(core)
    assert model.ncon == 1 and model.nnzj == 2
    sol = model.solve()
    assert sol.success
    np.testing.assert_allclose(sol[x], 1 / np.sqrt(2), atol=1e-6)


def test_matrix_free_oracle():
    """The same block via products instead of matrices."""
    def f(c, x):
        c[0] = x[0]**2 + x[1]**2 - 1.0

    def jvp(Jv, x, v):
        Jv[0] = 2 * x[0] * v[0] + 2 * x[1] * v[1]

    def vjp(Jtv, x, w):
        Jtv[0], Jtv[1] = 2 * x[0] * w[0], 2 * x[1] * w[0]

    def hvp(Hv, x, w, v):
        Hv[0], Hv[1] = 2 * w[0] * v[0], 2 * w[0] * v[1]

    o = exa.VectorNonlinearOracle(nvar=2, ncon=1, f=f, jvp=jvp, vjp=vjp, hvp=hvp,
                                  lcon=[0.0], ucon=[0.0])
    assert exa.has_matfree_jac(o) and exa.has_matfree_hess(o)
    assert not exa.has_matfree_jac(unit_circle_oracle())


def test_scalar_oracle_contributes_to_the_objective():
    def f(x):
        return float((x[0] - 3.0)**2)

    def grad(g, x):
        g[0] = 2 * (x[0] - 3.0)
        g[1] = 0.0

    def hvp(Hv, x, v):
        Hv[0], Hv[1] = 2 * v[0], 0.0

    o = exa.ScalarNonlinearOracle(nvar=2, f=f, grad=grad, hvp=hvp)
    core = exa.Core()
    x = exa.add_var(core, 2, start=0.0)
    exa.add_obj(core, o)
    exa.add_obj(core, lambda i: x[1]**2, over=range(1))
    sol = exa.Model(core).solve()
    assert sol.success
    assert sol[x][0] == pytest.approx(3.0, abs=1e-5)
    assert sol[x][1] == pytest.approx(0.0, abs=1e-5)


def test_an_oracle_model_still_evaluates_and_reports():
    core = exa.Core()
    x = exa.add_var(core, 2, start=0.5)
    exa.add_obj(core, lambda i: -x[0] - x[1], over=range(1))
    exa.add_con(core, unit_circle_oracle())
    model = exa.Model(core)                       # an ExaModelWithOracle
    z = np.array([0.6, 0.8])
    assert model.constraints(z)[0] == pytest.approx(0.0, abs=1e-12)
    assert model.gradient(z).shape == (2,)
    assert model.violation(z) < 1e-9


# ------------------------------------------------------------ two-stage ------
def two_stage(nscen=3, nper=2):
    Row = namedtuple("Row", "i t")
    target = np.arange(1.0, nscen * nper + 1.0)
    core = exa.TwoStageCore(nscen)
    d = exa.add_var(core, 1, start=0.0)
    v = exa.add_var(core, exa.EachScenario(), nper, start=0.0)
    rows = [Row(k, target[k]) for k in range(nscen * nper)]
    exa.add_obj(core, lambda r: (v[r.i] - r.t)**2, over=rows)
    exa.add_con(core, exa.EachScenario(), lambda i: v[i] - d[0],
                over=range(nscen * nper))
    return core, d, v, target


def test_two_stage_shapes_and_scenario_tags():
    core, d, v, target = two_stage()
    assert len(d) == 1 and len(v) == 6
    model = exa.Model(core)
    assert exa.get_nscen(model) == 3
    np.testing.assert_array_equal(exa.get_var_scen(model), [0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(exa.get_con_scen(model), [1, 1, 2, 2, 3, 3])


def test_two_stage_solves_to_the_expected_design():
    """Every recourse variable is tied to the design, so it lands on the mean."""
    core, d, v, target = two_stage()
    sol = exa.Model(core).solve()
    assert sol.success
    assert sol[d][0] == pytest.approx(target.mean(), abs=1e-6)
    np.testing.assert_allclose(sol[v], target.mean(), atol=1e-6)


# ----------------------------------------------------------------- tags ------
def test_ready_made_stage_tags():
    for maker in (exa.FirstStageTag, exa.SecondStageTag,
                  exa.FirstStageConstraintTag, exa.SecondStageConstraintTag):
        assert maker() is not None


def test_a_user_defined_tag_can_be_attached_to_a_block():
    tag = exa.new_tag("WillowTestTag", "variable")
    core = exa.Core()
    x = exa.add_var(core, 3, start=1.0, tag=tag)
    exa.add_obj(core, lambda i: x[i]**2, over=range(3))
    assert exa.Model(core).nvar == 3


def test_new_tag_is_idempotent():
    a = exa.new_tag("WillowRepeatTag", "variable")
    b = exa.new_tag("WillowRepeatTag", "variable")
    assert str(a) == str(b)


# ------------------------------------------------------------- wrappers ------
def base_model():
    core = exa.Core()
    x = exa.add_var(core, 3, start=1.0)
    exa.add_obj(core, lambda i: x[i]**2, over=range(3))
    exa.add_con(core, lambda i: x[i] + x[i + 1], over=range(2), lcon=-1.0, ucon=1.0)
    return exa.Model(core), x


@pytest.mark.parametrize("wrap", [exa.WrapperNLPModel, exa.TimedNLPModel,
                                  exa.CompressedNLPModel])
def test_wrappers_preserve_the_problem(wrap):
    model, x = base_model()
    w = wrap(model)
    assert w.nvar == model.nvar and w.ncon == model.ncon
    z = np.array([0.3, 0.4, 0.5])
    assert w.objective(z) == pytest.approx(model.objective(z))
    np.testing.assert_allclose(w.constraints(z), model.constraints(z))


def test_a_wrapped_model_still_solves():
    model, x = base_model()
    assert exa.WrapperNLPModel(model).solve().success


def test_timings_are_reported():
    model, x = base_model()
    timed = exa.TimedNLPModel(model)
    timed.solve()
    report = exa.timings(timed)
    assert isinstance(report, str) and len(report) > 0


@pytest.mark.parametrize("bad", [
    "X; run(`echo hi`)", "X end; y=1; struct", "1abc", "", "aé", 42,
])
def test_a_tag_name_that_is_not_an_identifier_is_refused(bad):
    """The one caller-supplied string reaching the backend as source, so check it."""
    with pytest.raises(ValueError, match="plain identifier"):
        exa.new_tag(bad)


def test_an_unknown_tag_kind_is_refused():
    with pytest.raises(ValueError, match="variable.*constraint"):
        exa.new_tag("Fine", "elephant")
