"""Parameters: fixed values usable in expressions and changeable without a rebuild."""
import numpy as np
import pytest

import madsuite as exa
from madsuite.testing import reference_trace, same_structure

N = 10


def parametric_model():
    """The parametric Luksan-Vlcek model from the ExaModels parameter docs."""
    core = exa.Core()
    th = core.add_par([100.0, 1.0])              # [penalty, offset]
    x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])
    core.add_obj(lambda i: th[0] * (x[i-1]**2 - x[i])**2 + (x[i-1] - th[1])**2,
               over=range(1, N))
    core.add_con(lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
                + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
                + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
                over=range(0, N - 2), lcon=0.0, ucon=0.0)
    return core, x, th


def test_parametric_expression_matches_julia():
    core = exa.Core()
    th = core.add_par([100.0, 1.0])
    x = core.add_var(N)
    got = exa.trace(lambda i: th[0] * (x[i-1]**2 - x[i])**2 + (x[i-1] - th[1])**2)
    want = reference_trace(
        x, "th[0] * (x[i-1]^2 - x[i])^2 + (x[i-1] - th[1])^2 for i = 1:9", th=th)
    assert same_structure(got, want), f"\n got: {got.julia_type}\nwant: {want.julia_type}"


def test_changing_a_parameter_changes_the_solution_without_rebuilding():
    core, x, th = parametric_model()
    p = exa.Model(core)

    base = p.solve()
    p.set_value(th, [200.0, 1.0])               # double the penalty
    heavier = p.solve()
    p.set_value(th, [200.0, 0.5])               # move the offset
    shifted = p.solve()

    for s in (base, heavier, shifted):
        assert s.success, s.status
    # a no-op setter would make these identical -- they must not be
    assert heavier.objective != pytest.approx(base.objective, rel=1e-6)
    assert shifted.objective != pytest.approx(heavier.objective, rel=1e-6)


def test_matches_the_julia_parametric_model():
    from madsuite import _bridge as _b
    core, x, th = parametric_model()
    p = exa.Model(core)

    jl = _b.seval(f"""
        begin
            c = ExaCore(concrete = Val(true))
            N = {N}
            c, t = add_par(c, [100.0, 1.0])
            c, x = add_var(c, N; start = [i % 2 == 1 ? -1.2 : 1.0 for i = 1:N])
            c, _ = add_obj(c, (t[1]*(x[i-1]^2 - x[i])^2 + (x[i-1] - t[2])^2 for i = 2:N))
            c, _ = add_con(c, (3x[i+1]^3 + 2*x[i+2] - 5 +
                   sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
                   x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2)))
            mm = ExaModel(c)
            using NLPModelsIpopt
            o1 = ipopt(mm; print_level = 0).objective
            set_value!(mm, t, [200.0, 1.0])
            o2 = ipopt(mm; print_level = 0).objective
            [o1, o2]
        end""")
    o1, o2 = float(jl[0]), float(jl[1])

    assert p.solve().objective == pytest.approx(o1, rel=1e-8)
    p.set_value(th, [200.0, 1.0])
    assert p.solve().objective == pytest.approx(o2, rel=1e-8)


def test_get_parameters_round_trips():
    core, x, th = parametric_model()
    p = exa.Model(core)
    np.testing.assert_allclose(p.get_value(th), [100.0, 1.0])
    p.set_value(th, [7.0, -3.0])
    got = p.get_value(th)
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, [7.0, -3.0])


def test_wrong_number_of_values_is_rejected():
    """The backend's own size check must surface as a plain Python ValueError."""
    core, x, th = parametric_model()
    with pytest.raises(ValueError, match="expected 2 elements, got 3"):
        exa.Model(core).set_value(th, [1.0, 2.0, 3.0])


def test_parameter_index_out_of_range():
    core = exa.Core()
    th = core.add_par([1.0, 2.0])
    with pytest.raises(IndexError):
        th[2]
