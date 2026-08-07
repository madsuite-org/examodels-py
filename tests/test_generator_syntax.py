"""A generator expression should build the same model a function does."""
import numpy as np
import pytest

from collections import namedtuple

import examodels as exa
from examodels.testing import full_types, reference_trace, same_structure

N = 10


@pytest.fixture(scope="module", autouse=True)
def _display():
    full_types(True)


def test_generator_and_function_agree_with_the_backend():
    core = exa.Core()
    x = core.add_var(N)
    want = reference_trace(x, "100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:10")

    from examodels.core import _as_function
    body, over = _as_function(
        (100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2 for i in range(1, N)), None)
    assert over == range(1, N), "the index set must come from the generator itself"
    assert same_structure(exa.trace(body), want)
    assert same_structure(exa.trace(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2),
                          want)


def _luksan(core, x, generator):
    if generator:
        core.add_obj(100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2 for i in range(1, N))
        core.add_con((x[i] + x[i+1] for i in range(N - 1)), lcon=-5.0, ucon=5.0)
    else:
        core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
                     over=range(1, N))
        core.add_con(lambda i: x[i] + x[i+1], over=range(N - 1), lcon=-5.0, ucon=5.0)
    return exa.Model(core)


def test_both_forms_build_identical_models():
    models = []
    for generator in (True, False):
        core = exa.Core()
        x = core.add_var(N, start=0.5)
        models.append(_luksan(core, x, generator))
    a, b = models
    assert (a.nvar, a.ncon, a.nnzj, a.nnzh) == (b.nvar, b.ncon, b.nnzj, b.nnzh)
    xs = np.linspace(0.1, 1.0, N)
    assert a.objective(xs) == pytest.approx(b.objective(xs))
    np.testing.assert_allclose(a.constraints(xs), b.constraints(xs))
    assert a.solve().objective == pytest.approx(b.solve().objective)


def test_the_generator_is_not_consumed_or_materialised():
    """Tracing must not iterate the index set: a huge range must stay cheap."""
    core = exa.Core()
    x = core.add_var(1_000_000)
    core.add_obj(x[i]**2 for i in range(1_000_000))       # would be slow if iterated
    assert exa.Model(core).nvar == 1_000_000


def test_records_index_set_survives_the_generator():
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    Row = namedtuple("Row", "i c")
    rows = exa.Records([Row(k, 2.0) for k in range(3)], index=["i"])
    core.add_obj(r.c * x[r.i]**2 for r in rows)
    assert exa.Model(core).objective(np.ones(3)) == pytest.approx(6.0)


def test_augmentation_accepts_a_generator():
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    y = core.add_var(3, start=1.0)
    Row = namedtuple("Row", "i c")
    rows = exa.Records([Row(k, 1.0) for k in range(3)], index=["i"])
    con = core.add_con(r.c + x[r.i] for r in rows)
    core.add_con(con, ((r.i, y[r.i]) for r in rows))
    model = exa.Model(core)
    assert model.ncon == 3
    np.testing.assert_allclose(model.constraints(np.ones(6)), 3.0)


def test_nested_for_is_rejected_clearly():
    core = exa.Core()
    x = core.add_var(9)
    with pytest.raises(TypeError, match="more than one"):
        core.add_obj(x[i] * x[j] for i in range(3) for j in range(3))
