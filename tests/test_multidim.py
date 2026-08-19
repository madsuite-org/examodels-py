"""Multi-dimensional variable blocks."""
from collections import namedtuple

import numpy as np
import pytest

import madsuite as exa


def test_shape_and_size():
    core = exa.Core()
    x = core.add_var(2, 3)
    assert x.shape == (2, 3) and len(x) == 6
    assert exa.Model(core).nvar == 6


def test_values_round_trip_in_the_callers_orientation():
    """The backend stores a block column-major; a caller must never see that.

    Deliberately non-square with distinct entries: a square block, or one filled
    with a constant, would pass even if the array came back transposed.
    """
    start = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    core = exa.Core()
    x = core.add_var(2, 3, start=start)
    got = exa.Model(core).get_start(x)
    assert got.shape == (2, 3)
    np.testing.assert_array_equal(got, start)
    assert got[0, 1] == 2.0, "transposed: got the [1, 0] element"


def test_concrete_index_addresses_the_intended_variable():
    """x[0, 1] must be the variable whose start is start[0, 1], not start[1, 0]."""
    core = exa.Core()
    x = core.add_var(2, 3, start=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    core.add_obj((x[0, 1] - 7.0)**2 + (x[1, 2] + 3.0)**2, over=range(1))
    sol = exa.Model(core).solve()
    xs = sol[x]
    assert xs.shape == (2, 3)
    assert xs[0, 1] == pytest.approx(7.0)
    assert xs[1, 2] == pytest.approx(-3.0)
    assert xs[0, 0] == pytest.approx(1.0), "an untouched entry moved"


def test_symbolic_two_dimensional_indexing():
    """x[t, i] traced over a table of (t, i) rows."""
    T, N = 4, 3
    Cell = namedtuple("Cell", "t i")
    grid = [Cell(t, i) for t in range(1, T) for i in range(N)]
    core = exa.Core()
    x = core.add_var(T, N, start=1.0,
                     lvar=np.full((T, N), -5.0), uvar=np.full((T, N), 5.0))
    core.add_obj(sum((x[0, i] - float(i))**2 for i in range(N)), over=range(1))
    core.add_con(x[c.t, c.i] - x[c.t - 1, c.i] for c in grid)
    model = exa.Model(core)
    assert model.nvar == T * N and model.ncon == (T - 1) * N
    sol = model.solve()
    assert sol.success
    # every row equals row 0, which is pinned to (0, 1, 2)
    np.testing.assert_allclose(sol[x], np.tile(np.arange(N, dtype=float), (T, 1)),
                               atol=1e-6)


def test_three_dimensions():
    core = exa.Core()
    x = core.add_var(2, 3, 4)
    assert x.shape == (2, 3, 4) and len(x) == 24
    core.add_obj((x[1, 2, 3] - 5.0)**2, over=range(1))
    sol = exa.Model(core).solve()
    assert sol[x].shape == (2, 3, 4)
    assert sol[x][1, 2, 3] == pytest.approx(5.0)


def test_wrong_number_of_indices():
    core = exa.Core()
    x = core.add_var(2, 3)
    with pytest.raises(IndexError, match="takes 2 indices"):
        x[1]
    with pytest.raises(IndexError, match="takes 2 indices"):
        x[1, 2, 3]


def test_index_out_of_range_names_the_axis():
    core = exa.Core()
    x = core.add_var(2, 3)
    with pytest.raises(IndexError, match="axis 1"):
        x[0, 5]
    with pytest.raises(IndexError, match="axis 0"):
        x[9, 0]


def test_one_dimensional_blocks_are_unchanged():
    core = exa.Core()
    x = core.add_var(5, start=2.0)
    assert x.shape == (5,) and len(x) == 5
    assert exa.Model(core).get_start(x).shape == (5,)
    assert x[0] is not None and x[-1] is not None


def test_range_dimensions():
    """A dimension may be a range, like the backend's `add_var(c, 2:10)`."""
    core = exa.Core()
    x = core.add_var(range(2, 11), start=1.0)
    assert x.shape == (9,)
    core.add_obj((x[2] - 5.0)**2 + (x[10] + 4.0)**2, over=range(1))
    core.add_con(x[i] - x[i - 1] for i in range(3, 11))
    sol = exa.Model(core).solve()
    assert sol.success
    np.testing.assert_allclose(sol[x], 0.5, atol=1e-6)      # all tied, midpoint
    with pytest.raises(IndexError, match=r"2\.\.10"):
        x[1]


def test_product_index_set_for_constraints():
    T, N = 4, 3
    core = exa.Core()
    x = core.add_var(T, N, start=1.0)
    core.add_obj(sum((x[0, i] - float(i))**2 for i in range(N)), over=range(1))
    core.add_con(x[t, i] - x[t - 1, i] for t, i in exa.product(range(1, T), range(N)))
    model = exa.Model(core)
    assert model.ncon == (T - 1) * N
    sol = model.solve()
    np.testing.assert_allclose(sol[x], np.tile(np.arange(N, dtype=float), (T, 1)),
                               atol=1e-6)


def test_product_index_set_for_objectives_both_syntaxes():
    T, N = 3, 4
    want = np.add.outer(np.arange(T), np.arange(N)).astype(float)
    for build in ("generator", "lambda"):
        core = exa.Core()
        x = core.add_var(T, N)
        if build == "generator":
            core.add_obj((x[t, i] - (t + i))**2
                         for t, i in exa.product(range(T), range(N)))
        else:
            core.add_obj(lambda t, i: (x[t, i] - (t + i))**2,
                         over=exa.product(range(T), range(N)))
        np.testing.assert_allclose(exa.Model(core).solve()[x], want, atol=1e-6)


def test_an_index_is_a_number_not_a_backend_position():
    """`x[i] - i` must use the index the caller wrote, on both sides of the minus."""
    core = exa.Core()
    y = core.add_var(5)
    core.add_obj((y[i] - i)**2 for i in range(5))
    np.testing.assert_allclose(exa.Model(core).solve()[y], np.arange(5.0), atol=1e-6)


def test_three_dimensional_product():
    core = exa.Core()
    z = core.add_var(2, 2, 2, start=1.0)
    core.add_obj((z[0, 0, 0] - 9.0)**2, over=range(1))
    core.add_con(z[a, b, c] - z[0, 0, 0]
                 for a, b, c in exa.product(range(2), range(2), range(2)))
    sol = exa.Model(core).solve()
    np.testing.assert_allclose(sol[z], 9.0, atol=1e-6)
