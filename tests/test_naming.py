"""`name=` registers a block, retrievable from the core and from the model."""
import numpy as np
import pytest

import examodels as exa


def named_core():
    core = exa.Core()
    th = exa.add_par(core, [2.0], name="theta")
    x = exa.add_var(core, 4, start=0.5, name="x")
    exa.add_obj(core, lambda i: (x[i] - th[0])**2, over=range(4), name="cost")
    exa.add_con(core, lambda i: x[i] + x[i + 1], over=range(3),
                lcon=0.0, ucon=5.0, name="link")
    exa.add_expr(core, lambda i: x[i]**2, over=range(4), name="sq")
    return core


@pytest.mark.parametrize("name", ["theta", "x", "cost", "link", "sq"])
def test_names_are_registered_on_the_core(name):
    assert getattr(named_core(), name) is not None


@pytest.mark.parametrize("name", ["theta", "x", "link"])
def test_names_survive_into_the_model(name):
    model = exa.Model(named_core())
    assert getattr(model, name) is not None


def test_a_named_block_is_the_same_handle():
    core = named_core()
    model = exa.Model(core)
    assert model.x is core.x
    np.testing.assert_allclose(model.get_start(model.x), 0.5)
    sol = model.solve()
    np.testing.assert_allclose(sol[model.x], 2.0, atol=1e-6)
    assert sol.multipliers(model.link).shape == (3,)


def test_an_unknown_name_is_an_attribute_error():
    core = named_core()
    with pytest.raises(AttributeError):
        core.nope
    with pytest.raises(AttributeError):
        exa.Model(core).nope


def test_names_do_not_shadow_model_metadata():
    model = exa.Model(named_core())
    assert isinstance(model.nvar, int) and model.nvar == 4
