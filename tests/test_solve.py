"""End-to-end: build the Luksan-Vlcek problem in Python and solve it."""
import numpy as np
import pytest

import examodels as exa

N = 10


def luksan_vlcek(n=N):
    m = exa.Model()
    x = m.add_variables(n, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
    m.minimize(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2, over=range(1, n))
    m.constrain(
        lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
        + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
        + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
        over=range(0, n - 2), lower=0.0, upper=0.0)
    return m, x


def test_solves_to_first_order_point():
    m, x = luksan_vlcek()
    sol = m.solve(solver="ipopt")
    assert sol.success, sol.status
    assert sol.status == "first_order"
    assert np.isfinite(sol.objective)
    assert sol.iterations > 0


def test_solution_indexing_returns_numpy():
    m, x = luksan_vlcek()
    sol = m.solve()
    xs = sol[x]
    assert isinstance(xs, np.ndarray) and xs.shape == (N,)
    np.testing.assert_allclose(xs, sol.x)


def test_constraints_are_satisfied_at_the_solution():
    m, x = luksan_vlcek()
    p = m.build()
    sol = p.solve()
    c = p.constraints(sol.x)
    assert np.max(np.abs(c)) < 1e-6, f"max |c(x)| = {np.max(np.abs(c))}"


def test_matches_the_julia_formulation():
    """The Python model must be the same problem Julia's own model is."""
    from examodels import _bridge as _b
    m, _ = luksan_vlcek()
    p = m.build()
    _b.seval("using ExaModels")
    jl_obj = float(_b.seval(f"""
        begin
            c = ExaCore(concrete = Val(true))
            N = {N}
            c, x = add_var(c, N; start = [i % 2 == 1 ? -1.2 : 1.0 for i = 1:N])
            c, _ = add_obj(c, (100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))
            c, _ = add_con(c, (3x[i+1]^3 + 2 * x[i+2] - 5 +
                   sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
                   x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2)))
            m = ExaModel(c)
            using NLPModelsIpopt
            ipopt(m; print_level = 0).objective
        end"""))
    sol = p.solve()
    assert sol.objective == pytest.approx(jl_obj, rel=1e-8)
