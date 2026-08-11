"""Reading and writing a built model, and reading a solve result.

The accessor loop (`get_start`, `set_lvar`, ...) is exercised elsewhere; this
file covers the surrounding surface: attribute routing on `Model` and
`Solution`, the parameter block round trip, and the reprs someone actually
reads at a prompt.
"""
import numpy as np
import pytest
from conftest import requires

import examodels as exa


@pytest.fixture
def solved():
    core = exa.Core()
    x = core.add_var(2, start=0.5, name="x")
    th = core.add_par([1.0, 2.0], name="th")
    core.add_obj(lambda i: (x[i] - th[i]) ** 2, over=range(2))
    model = core.build()
    return model, x, th


def test_model_reprs_read_off_the_backend(solved):
    model, _x, _th = solved
    assert repr(model) == "<Model nvar=2 ncon=0 nnzj=0 nnzh=2>"
    assert repr(exa.Core()) == "<Core>"


def test_underscore_attributes_stay_python_side(solved):
    model, _x, _th = solved
    with pytest.raises(AttributeError):
        model._not_ours
    with pytest.raises(AttributeError, match="'Model' has no attribute 'zzz'"):
        model.zzz


def test_dir_lists_the_backend_meta_fields(solved):
    model, _x, _th = solved
    names = dir(model)
    assert {"nvar", "ncon", "x0"} <= set(names)


def test_named_blocks_resolve_on_the_model(solved):
    model, x, th = solved
    assert model.x is x and model.th is th


def test_parameters_round_trip_without_rebuilding(solved):
    model, _x, th = solved
    np.testing.assert_allclose(model.parameters(th), [1.0, 2.0])
    assert model.set_parameters(th, [3, 4]) is model     # chains
    np.testing.assert_allclose(model.parameters(th), [3.0, 4.0])


@requires("ipopt")
def test_new_parameter_values_move_the_solution(solved):
    model, x, th = solved
    first = model.solve()
    model.set_parameters(th, [5.0, 6.0])
    second = model.solve()
    np.testing.assert_allclose(second[x], [5.0, 6.0], atol=1e-6)
    assert not np.allclose(first[x], second[x])


@requires("ipopt")
def test_solution_attribute_routing(solved):
    model, _x, _th = solved
    sol = model.solve()
    with pytest.raises(AttributeError):
        sol._raw_is_private
    with pytest.raises(AttributeError, match="solutions have no attribute"):
        sol.nothing_called_this
    assert repr(sol).startswith("<Solution status='first_order'")


@requires("ipopt")
def test_unconvertible_result_fields_pass_through_as_is(solved):
    # `solver_specific` is a Julia Dict: there is no numpy value for it, and the
    # fallback hands back the object rather than failing the attribute read.
    model, _x, _th = solved
    sol = model.solve()
    v = sol.solver_specific
    assert v is not None
    assert not isinstance(v, np.ndarray)


@requires("ipopt")
def test_solution_alias_matches_indexing(solved):
    model, x, _th = solved
    sol = model.solve()
    np.testing.assert_allclose(exa.solution(sol, x), sol[x])


def test_the_package_refuses_unknown_names():
    with pytest.raises(AttributeError, match="no attribute 'definitely_not_an_op'"):
        exa.definitely_not_an_op


@requires("ipopt")
def test_a_foreign_handle_fails_as_itself_not_as_a_placeholder(solved):
    # The readable placeholder refusal must not swallow a genuine error on an
    # ordinary block: a handle from a different, larger core stays a BoundsError.
    model, _x, _th = solved
    other = exa.Core()
    big = other.add_var(10)
    sol = model.solve()
    with pytest.raises(IndexError):
        sol[big]
    with pytest.raises(IndexError):
        model.get_start(big)
