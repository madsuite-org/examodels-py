"""Subexpressions: reusable, and inlined rather than turned into auxiliary variables."""
import pytest

import examodels as exa
from examodels.testing import reference_trace, same_structure

N = 10


def test_subexpression_is_inlined_not_referenced():
    """`s[i]` must expand to the expression itself, identically to writing it out."""
    core = exa.Core()
    y = core.add_variables(N)
    s = core.add_expression(lambda i: y[i]**2, over=range(N))

    got = exa.trace(lambda i: (s[i] - 1)**2)
    want = reference_trace(y, "(x[i]^2 - 1)^2 for i = 1:10")
    assert same_structure(got, want), f"\n got: {got.julia_type}\nwant: {want.julia_type}"


def test_matches_julias_own_add_expr():
    """The same model written with Julia's @add_expr must produce the same expression."""
    core = exa.Core()
    y = core.add_variables(N)
    s = core.add_expression(lambda i: y[i]**2, over=range(N))
    got = exa.trace(lambda i: (s[i] - 1)**2)

    from examodels import _bridge as _b
    want_jl = _b.seval(f"""
        begin
            c = ExaCore(concrete = Val(true))
            c, y = add_var(c, {N})
            c, s = add_expr(c, (y[i]^2 for i in 1:{N}))
            gen = ((s[i] - 1)^2 for i in 1:{N})
            gen.f(ExaModels.DataSource())
        end""")
    assert same_structure(got, want_jl)


def test_reuse_across_objective_and_constraint():
    core = exa.Core()
    y = core.add_variables(N, start=0.5)
    s = core.add_expression(lambda i: y[i]**2, over=range(N))
    core.minimize(lambda i: (s[i] - 1)**2, over=range(N))
    core.constrain(lambda i: s[i] + s[i+1], over=range(N - 1), lower=0.0, upper=10.0)

    p = exa.Model(core)
    assert p.nvar == N          # inlining adds no variables
    assert p.ncon == N - 1      # and no extra constraint rows
    sol = p.solve()
    assert sol.success, sol.status


def test_multidimensional_subexpression():
    T, K = 4, 3
    core = exa.Core()
    z = core.add_variables(T * K, start=1.0)
    # dx[t, i] = z[t*K + i] - z[(t-1)*K + i]
    dx = core.add_expression(lambda t, i: z[t * K + i] - z[(t - 1) * K + i],
                          over=(range(1, T), range(K)))
    core.minimize(lambda t: dx[t, 0]**2, over=range(1, T))
    p = exa.Model(core)
    assert p.nvar == T * K
    assert len(dx) == (T - 1) * K
    assert p.solve().success


def test_wrong_index_count_is_rejected():
    core = exa.Core()
    y = core.add_variables(N)
    s = core.add_expression(lambda i: y[i]**2, over=range(N))
    with pytest.raises(IndexError, match="indexed by 1, got 2"):
        s[1, 2]


def test_concrete_index_outside_the_set_is_rejected():
    core = exa.Core()
    y = core.add_variables(N)
    s = core.add_expression(lambda i: y[i]**2, over=range(3, 7))
    with pytest.raises(IndexError, match="outside"):
        s[9]
    s[5]        # inside the set: fine
