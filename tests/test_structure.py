"""A Python function must produce exactly the expression ExaModels' own tracing does."""
import pytest

import examodels as exa
from examodels.testing import full_types, reference_trace, same_structure

N = 10


@pytest.fixture(scope="module")
def xvar():
    full_types(True)
    core = exa.Core()
    return core.add_var(N)


def test_luksan_vlcek_objective(xvar):
    x = xvar
    got = exa.trace(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2)
    want = reference_trace(x, "100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:10")
    assert same_structure(got, want), f"\n got: {got.julia_type}\nwant: {want.julia_type}"
    assert len(got.julia_type) > 300, "expression is too small to be a meaningful check"


def test_luksan_vlcek_constraint(xvar):
    x = xvar
    sin, exp = exa.sin, exa.exp
    got = exa.trace(lambda i:
        3 * x[i+1]**3 + 2 * x[i+2] - 5
        + sin(x[i+1] - x[i+2]) * sin(x[i+1] + x[i+2])
        + 4 * x[i+1] - x[i] * exp(x[i] - x[i+1]) - 3)
    want = reference_trace(
        x, "3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + "
           "4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:8")
    assert same_structure(got, want), f"\n got: {got.julia_type}\nwant: {want.julia_type}"
    assert len(got.julia_type) > 900


@pytest.mark.parametrize("py, src", [
    (lambda x, i: x[i]**2,                "x[i]^2 for i=1:1"),
    (lambda x, i: x[i]**3,                "x[i]^3 for i=1:1"),
    (lambda x, i: x[i]**-1,               "x[i]^-1 for i=1:1"),
    (lambda x, i: x[i] * exa.Constant(1), "x[i]*ExaModels.Constant(1) for i=1:1"),
    (lambda x, i: x[i] + exa.Constant(0), "x[i]+ExaModels.Constant(0) for i=1:1"),
    (lambda x, i: exa.Constant(0) * x[i], "ExaModels.Constant(0)*x[i] for i=1:1"),
    (lambda x, i: -x[i],                  "-x[i] for i=1:1"),
    (lambda x, i: 1 / x[i],               "1/x[i] for i=1:1"),
    (lambda x, i: 2 - x[i],               "2-x[i] for i=1:1"),
])
def test_rewrites_and_operators(xvar, py, src):
    x = xvar
    got = exa.trace(lambda i: py(x, i))
    assert same_structure(got, reference_trace(x, src)), \
        f"\n got: {got.julia_type}\nwant: {reference_trace(x, src).julia_type}"


def test_literal_exponent_is_specialized(xvar):
    """`x**2` must reach the abs2 rewrite, not a runtime-Int64 power."""
    got = exa.trace(lambda i: xvar[i]**2)
    assert "abs2" in got.julia_type
    assert "Int64}" not in got.julia_type.split("Var{")[0]


def test_comparison_can_fail(xvar):
    """Negative control: the structural check must not pass for a different expression."""
    x = xvar
    assert not same_structure(exa.trace(lambda i: x[i]**2),
                              reference_trace(x, "x[i]^3 for i=1:1"))


def test_branching_on_index_is_rejected(xvar):
    x = xvar
    with pytest.raises(TypeError):
        exa.trace(lambda i: x[i] if i else x[i-1])


def test_generator_expression_traces_like_a_lambda(xvar):
    """`expr for i in over` must produce exactly what the lambda form does."""
    x = xvar
    core = exa.Core()
    y = core.add_var(N)
    got_gen = exa.trace(lambda i: 100 * (y[i-1]**2 - y[i])**2 + (y[i-1] - 1)**2)
    # build the same thing through the generator path and compare structurally
    from examodels.core import _as_function
    f, over = _as_function((100 * (y[i-1]**2 - y[i])**2 + (y[i-1] - 1)**2
                            for i in range(1, N)), None)
    assert over == range(1, N), over
    assert same_structure(exa.trace(f), got_gen)


def test_generator_expression_does_not_consume_the_generator():
    g = (i * 2 for i in range(4))
    from examodels.core import _as_function
    _as_function(g, None)
    assert list(g) == [0, 2, 4, 6], "the generator was consumed while tracing"


def test_multiple_for_clauses_are_rejected_clearly():
    core = exa.Core()
    z = core.add_var(9)
    with pytest.raises(TypeError, match="more than one"):
        core.add_obj(z[i] * z[j] for i in range(3) for j in range(3))
