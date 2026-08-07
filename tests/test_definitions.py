"""Every supported way of defining variables, parameters, expressions,
objectives and constraints — and, where a backend form has no Python spelling,
the error that says so.
"""
from collections import namedtuple

import numpy as np
import pytest

import examodels as exa

Row = namedtuple("Row", "i c")


# ---------------------------------------------------------------- add_var ----
@pytest.mark.parametrize("dims, shape, size", [
    ((5,), (5,), 5),
    ((range(2, 11),), (9,), 9),
    ((3, 4), (3, 4), 12),
    ((range(1, 4), range(0, 2)), (3, 2), 6),          # mixed offsets
    ((2, 2, 2), (2, 2, 2), 8),
])
def test_add_var_dimension_forms(dims, shape, size):
    core = exa.Core()
    x = core.add_var(*dims)
    assert x.shape == shape and len(x) == size
    assert exa.Model(core).nvar == size


@pytest.mark.parametrize("start, lvar, uvar", [
    (0.0, None, None),                                 # scalars, no bounds
    (1.5, -1.0, 2.0),                                  # scalars
    ([0.1, 0.2, 0.3], [0.0] * 3, [1.0] * 3),           # lists
    (np.linspace(0, 1, 3), np.zeros(3), np.ones(3)),   # arrays
    ([i / 10 for i in range(3)], None, None),          # comprehension
])
def test_add_var_bound_forms(start, lvar, uvar):
    core = exa.Core()
    x = core.add_var(3, start=start, lvar=lvar, uvar=uvar)
    model = exa.Model(core)
    np.testing.assert_allclose(model.get_start(x), start)
    if lvar is not None:
        np.testing.assert_allclose(model.get_lvar(x), lvar)
        np.testing.assert_allclose(model.get_uvar(x), uvar)


def test_add_var_defined_by_expressions():
    """add_var(expr for i in over) ties each new variable to its expression."""
    core = exa.Core()
    x = core.add_var(4, start=2.0, lvar=1.0, uvar=3.0)
    s = core.add_var(x[i]**2 for i in range(4))
    core.add_obj((s[i] - 9.0)**2 for i in range(4))
    model = exa.Model(core)
    assert model.nvar == 8 and model.ncon == 4      # 4 new vars, 4 tying rows
    sol = model.solve()
    np.testing.assert_allclose(sol[s], sol[x]**2, atol=1e-6)


def test_add_var_rejects_a_tuple_of_dimensions():
    core = exa.Core()
    with pytest.raises(TypeError, match="one integer or range per dimension"):
        core.add_var((3, 4))


# ---------------------------------------------------------------- add_par ----
@pytest.mark.parametrize("values", [[1.0], [1.0, 2.0, 3.0], np.arange(4.0)])
def test_add_par_forms(values):
    core = exa.Core()
    th = core.add_par(values)
    x = core.add_var(1, start=0.0)                  # a model needs at least one
    core.add_obj(x[0] * th[0], over=range(1))
    assert len(th) == len(values)
    np.testing.assert_allclose(exa.Model(core).get_value(th), values)


def test_parameters_are_changeable_without_rebuilding():
    core = exa.Core()
    th = core.add_par([1.0])
    x = core.add_var(1, start=0.0)
    core.add_obj((x[0] - th[0])**2, over=range(1))
    model = exa.Model(core)
    assert model.solve()[x][0] == pytest.approx(1.0)
    model.set_value(th, [4.0])
    assert model.solve()[x][0] == pytest.approx(4.0)


# ---------------------------------------------------------------- add_obj ----
def test_add_obj_forms_agree():
    """generator, lambda + over, and a prebuilt expression must all agree."""
    results = []
    for how in ("generator", "lambda", "prebuilt"):
        core = exa.Core()
        x = core.add_var(4, start=0.3)
        if how == "generator":
            core.add_obj((x[i] - 1.0)**2 for i in range(4))
        elif how == "lambda":
            core.add_obj(lambda i: (x[i] - 1.0)**2, over=range(4))
        else:
            core.add_obj(exa.trace(lambda i: (x[i] - 1.0)**2), over=range(4))
        results.append(exa.Model(core).solve().objective)
    assert results[0] == pytest.approx(results[1]) == pytest.approx(results[2])


def test_several_objective_terms_are_summed():
    core = exa.Core()
    x = core.add_var(2, start=1.0)
    core.add_obj((x[i] - 1.0)**2 for i in range(2))
    core.add_obj(x[0] * x[1], over=range(1))
    assert exa.Model(core).objective([1.0, 1.0]) == pytest.approx(1.0)


def test_objective_over_a_table_and_over_a_product():
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    rows = exa.Records([Row(k, float(k)) for k in range(3)], index=["i"])
    core.add_obj(r.c * x[r.i] for r in rows)
    y = core.add_var(2, 2, start=1.0)
    core.add_obj(y[a, b] for a, b in exa.product(range(2), range(2)))
    model = exa.Model(core)
    assert model.objective(np.ones(model.nvar)) == pytest.approx(0 + 1 + 2 + 4)


# ---------------------------------------------------------------- add_con ----
@pytest.mark.parametrize("lcon, ucon", [
    (0.0, 0.0),                                        # equality, scalars
    (-1.0, 1.0),                                       # two-sided
    (-np.inf, 0.0),                                    # one-sided
    ([-1.0, -2.0], [1.0, 2.0]),                        # arrays
])
def test_add_con_bound_forms(lcon, ucon):
    core = exa.Core()
    x = core.add_var(3, start=0.0)
    con = core.add_con((x[i] + x[i + 1] for i in range(2)), lcon=lcon, ucon=ucon)
    model = exa.Model(core)
    assert model.ncon == 2
    np.testing.assert_allclose(model.get_lcon(con), lcon)
    np.testing.assert_allclose(model.get_ucon(con), ucon)


def test_add_con_over_range_table_and_product():
    core = exa.Core()
    x = core.add_var(4, start=1.0)
    y = core.add_var(2, 3, start=1.0)
    rows = exa.Records([Row(k, 1.0) for k in range(3)], index=["i"])
    a = core.add_con(x[i] - x[i + 1] for i in range(3))
    b = core.add_con(r.c * x[r.i] for r in rows)
    c = core.add_con(y[t, i] - y[t - 1, i] for t, i in exa.product(range(1, 2), range(3)))
    assert (len(a), len(b), len(c)) == (3, 3, 3)
    assert exa.Model(core).ncon == 9


def test_empty_constraint_block_filled_afterwards():
    """add_con(dims) makes rows with no terms; add_con(handle, ...) fills them."""
    core = exa.Core()
    y = core.add_var(3, start=1.0)
    blk = core.add_con(3, lcon=2.0, ucon=2.0)
    core.add_con(blk, lambda i: (i, y[i]), over=range(3))
    core.add_obj((y[i] - 5.0)**2 for i in range(3))
    model = exa.Model(core)
    assert model.ncon == 3
    np.testing.assert_allclose(model.solve()[y], 2.0, atol=1e-6)


def test_augmenting_the_same_block_several_times():
    """A balance built from more than one source, as a power balance is."""
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    y = core.add_var(3, start=1.0)
    rows = exa.Records([Row(k, 1.0) for k in range(3)], index=["i"])
    con = core.add_con(r.c + x[r.i] for r in rows)
    core.add_con(con, ((r.i, y[r.i]) for r in rows))
    core.add_con(con, ((r.i, -x[r.i]) for r in rows))          # cancels the x term
    model = exa.Model(core)
    assert model.ncon == 3, "augmentation must not add rows"
    np.testing.assert_allclose(model.constraints(np.ones(6)), 2.0)   # 1 + 1 + 1 - 1


def test_augmenting_a_range_based_block():
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    con = core.add_con(x[i] for i in range(3))
    core.add_con(con, lambda i: (i, x[i]), over=range(3))
    np.testing.assert_allclose(exa.Model(core).constraints(np.ones(3)), 2.0)


def test_augmenting_a_block_whose_index_set_is_offset():
    core = exa.Core()
    x = core.add_var(range(2, 5), start=1.0)
    con = core.add_con(x[i] for i in range(2, 5))
    core.add_con(con, lambda i: (i, x[i]), over=range(2, 5))
    np.testing.assert_allclose(exa.Model(core).constraints(np.ones(3)), 2.0)


# --------------------------------------------------------------- add_expr ----
def test_add_expr_is_inlined_and_reusable():
    core = exa.Core()
    y = core.add_var(5, start=0.5)
    sq = core.add_expr(y[i]**2 for i in range(5))
    core.add_obj((sq[i] - 1.0)**2 for i in range(5))
    con = core.add_con((sq[i] + sq[i + 1] for i in range(4)), lcon=0.0, ucon=10.0)
    model = exa.Model(core)
    assert model.nvar == 5, "a subexpression must not add variables"
    assert model.ncon == 4
    assert model.solve().success


def test_add_expr_multi_dimensional():
    T, N = 3, 2
    core = exa.Core()
    x = core.add_var(T, N, start=1.0)
    d = core.add_expr(lambda t, i: x[t, i] - x[t - 1, i],
                      over=exa.product(range(1, T), range(N)))
    core.add_obj(lambda t, i: d[t, i]**2, over=exa.product(range(1, T), range(N)))
    assert exa.Model(core).nvar == T * N


def test_nested_subexpressions():
    core = exa.Core()
    x = core.add_var(4, start=1.0)
    sq = core.add_expr(x[i]**2 for i in range(4))
    quad = core.add_expr(sq[i] * sq[i] for i in range(4))
    core.add_obj(quad[i] for i in range(4))
    assert exa.Model(core).objective(np.full(4, 2.0)) == pytest.approx(4 * 16)


# ----------------------------------------------------- both call styles ------
def test_function_style_matches_method_style():
    """`add_var(core, ...)` and `core.add_var(...)` are the same call."""
    objs = []
    for style in ("method", "function"):
        core = exa.Core()
        if style == "method":
            th = core.add_par([3.0])
            x = core.add_var(4, start=0.5, lvar=0.0, uvar=2.0)
            core.add_obj((x[i] - th[0])**2 for i in range(4))
            core.add_con((x[i] + x[i + 1] for i in range(3)), lcon=0.0, ucon=3.0)
            model = exa.Model(core)
            objs.append(model.solve().objective)
        else:
            th = exa.add_par(core, [3.0])
            x = exa.add_var(core, 4, start=0.5, lvar=0.0, uvar=2.0)
            exa.add_obj(core, ((x[i] - th[0])**2 for i in range(4)))
            exa.add_con(core, (x[i] + x[i + 1] for i in range(3)), lcon=0.0, ucon=3.0)
            model = exa.Model(core)
            objs.append(exa.solve(model).objective)
    assert objs[0] == pytest.approx(objs[1])


# --------------------------------------------------- stepped index sets ------
def test_stepped_range_is_a_valid_index_set():
    """`range(1, 9, 2)` selects indices 1, 3, 5, 7."""
    core = exa.Core()
    x = core.add_var(9, start=0.0)
    con = core.add_con((x[i] - 1.0 for i in range(1, 9, 2)), lcon=0.0, ucon=0.0)
    core.add_obj(x[i]**2 for i in range(9))
    model = exa.Model(core)
    assert len(con) == 4 and model.ncon == 4
    sol = model.solve()
    np.testing.assert_allclose(sol[x][1::2], 1.0, atol=1e-6)
    np.testing.assert_allclose(sol[x][0::2], 0.0, atol=1e-6)


def test_stepped_range_is_not_a_valid_dimension():
    """The backend defines lengths only for whole ranges, so say so clearly."""
    core = exa.Core()
    with pytest.raises(TypeError, match="cannot have a step"):
        core.add_var(range(1, 9, 2))


# ------------------------------------------------- table index sets ----------
def test_a_numpy_structured_array_is_an_index_set_on_its_own():
    """No wrapper needed: a structured array is already named, typed fields."""
    gen = np.array([(0, 1.0), (1, 2.0), (2, 3.0)], dtype=[("i", "i8"), ("c", "f8")])
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    core.add_obj(lambda g: g.c * x[g.i]**2, over=gen)
    assert exa.Model(core).objective(np.ones(3)) == pytest.approx(6.0)


def test_column_types_are_inferred_from_the_values():
    """Whole numbers stay integers, so a field can be used as an index."""
    R = namedtuple("R", "i c")
    rows = exa.Records([R(0, 1.0), R(1, 2.0), R(2, 3.0)])       # no index= given
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    core.add_obj(lambda g: g.c * x[g.i]**2, over=rows)
    assert exa.Model(core).objective(np.ones(3)) == pytest.approx(6.0)


def test_structured_array_and_named_tuples_build_the_same_model():
    R = namedtuple("R", "i c")
    arr = np.array([(0, 1.0), (1, 2.0), (2, 3.0)], dtype=[("i", "i8"), ("c", "f8")])
    models = []
    for table in (arr, exa.Records([R(0, 1.0), R(1, 2.0), R(2, 3.0)])):
        core = exa.Core()
        x = core.add_var(3, start=1.0)
        core.add_obj(lambda g: g.c * x[g.i]**2, over=table)
        m = exa.Model(core)
        models.append((m.nvar, m.ncon, m.nnzh, m.objective(np.ones(3))))
    assert models[0] == models[1]


def test_index_overrides_the_inferred_type():
    """A whole-numbered column that is really data can be forced to a float."""
    R = namedtuple("R", "i c")
    rows = exa.Records([R(0, 2), R(1, 3)], index=["i"])   # c is whole, wanted as data
    core = exa.Core()
    x = core.add_var(2, start=1.0)
    core.add_obj(lambda g: g.c * x[g.i]**2, over=rows)
    assert exa.Model(core).objective(np.ones(2)) == pytest.approx(5.0)
