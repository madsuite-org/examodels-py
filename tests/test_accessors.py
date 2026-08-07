"""Reading and changing a built model: every getter and setter, on every handle."""
import numpy as np
import pytest

import examodels as exa

ACCESSORS = ("value", "start", "lvar", "uvar", "lcon", "ucon")


def model():
    core = exa.Core()
    th = core.add_par([2.0, 3.0])
    x = core.add_var(4, start=0.5, lvar=0.0, uvar=2.0)
    y = core.add_var(2, 3, start=1.0,
                     lvar=np.zeros((2, 3)), uvar=np.full((2, 3), 4.0))
    core.add_obj(th[0] * (x[i] - 1.0)**2 for i in range(4))
    core.add_obj(y[a, b]**2 for a, b in exa.product(range(2), range(3)))
    con = core.add_con((x[i] + x[i + 1] for i in range(3)), lcon=-1.0, ucon=1.0)
    return exa.Model(core), th, x, y, con


@pytest.mark.parametrize("name, handle_index, expected", [
    ("value", 0, [2.0, 3.0]),
    ("start", 1, [0.5] * 4),
    ("lvar", 1, [0.0] * 4),
    ("uvar", 1, [2.0] * 4),
    ("lcon", 3, [-1.0] * 3),
    ("ucon", 3, [1.0] * 3),
])
def test_getters(name, handle_index, expected):
    m, *handles = model()
    handle = handles[handle_index] if handle_index < 3 else handles[3]
    np.testing.assert_allclose(getattr(m, f"get_{name}")(handle), expected)


@pytest.mark.parametrize("name, new", [
    ("value", [7.0, 8.0]), ("start", [0.1, 0.2, 0.3, 0.4]),
    ("lvar", [-1.0] * 4), ("uvar", [9.0] * 4),
    ("lcon", [-5.0] * 3), ("ucon", [5.0] * 3),
])
def test_setters_round_trip(name, new):
    m, th, x, y, con = model()
    handle = {"value": th, "start": x, "lvar": x, "uvar": x,
              "lcon": con, "ucon": con}[name]
    getattr(m, f"set_{name}")(handle, new)
    np.testing.assert_allclose(getattr(m, f"get_{name}")(handle), new)


def test_multi_dimensional_accessors_keep_the_callers_shape():
    m, th, x, y, con = model()
    for name in ("start", "lvar", "uvar"):
        got = getattr(m, f"get_{name}")(y)
        assert got.shape == (2, 3), name
    new = np.arange(6.0).reshape(2, 3)
    m.set_start(y, new)
    np.testing.assert_array_equal(m.get_start(y), new)
    assert m.get_start(y)[0, 1] == 1.0, "transposed"


def test_setters_take_effect_on_the_next_solve():
    m, th, x, y, con = model()
    first = m.solve()
    m.set_uvar(x, [0.25] * 4)              # force the solution against the bound
    second = m.solve()
    assert second[x].max() <= 0.25 + 1e-6
    assert second.objective > first.objective


def test_a_wrong_length_is_rejected_as_a_python_error():
    m, th, x, y, con = model()
    with pytest.raises(ValueError, match="expected 2 elements, got 3"):
        m.set_value(th, [1.0, 2.0, 3.0])


def test_metadata_comes_from_the_backend():
    m, *_ = model()
    assert (m.nvar, m.ncon) == (10, 3)
    assert m.nnzj > 0 and m.nnzh > 0
    assert m.x0.shape == (10,)
    for v in (m.nvar, m.ncon, m.nnzj, m.nnzh):
        assert type(v) is int


def test_evaluation_helpers():
    m, th, x, y, con = model()
    z = m.x0
    assert isinstance(m.objective(z), float)
    assert m.gradient(z).shape == (m.nvar,)
    assert m.constraints(z).shape == (m.ncon,)
    assert m.violation(z) >= 0.0


def test_result_accessors():
    m, th, x, y, con = model()
    sol = m.solve()
    assert sol[x].shape == (4,) and sol[y].shape == (2, 3)
    assert sol.multipliers(con).shape == (3,)
    assert sol.multipliers_L(x).shape == (4,)
    assert sol.multipliers_U(y).shape == (2, 3)
    assert sol.x.shape == (m.nvar,) and sol.y.shape == (m.ncon,)
    assert isinstance(sol.objective, float) and isinstance(sol.iterations, int)
    assert isinstance(sol.status, str) and sol.elapsed >= 0.0


def test_function_style_accessors():
    m, th, x, y, con = model()
    for name in ACCESSORS:
        handle = {"value": th, "lcon": con, "ucon": con}.get(name, x)
        np.testing.assert_allclose(getattr(exa, f"get_{name}")(m, handle),
                                   getattr(m, f"get_{name}")(handle))
    sol = exa.solve(m)
    np.testing.assert_allclose(exa.solution(sol, x), sol[x])
    np.testing.assert_allclose(exa.multipliers(sol, con), sol.multipliers(con))
