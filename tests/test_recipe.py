"""Recipes: a core written against placeholders, given its data later.

The interesting claims are that the *same* core instantiates more than once,
that placeholders reach every slot that takes data, and that the things a
placeholder cannot do fail where they are written rather than much later.
"""
import numpy as np
import pytest

import madsuite as exa


def rosen(core, n):
    """One source, used with real sizes or with a placeholder."""
    x = core.add_var(n, start=1.0)
    core.add_obj(lambda i: (x[i] - 2.0) ** 2,
                 over=exa.srange(0, n) if exa.is_placeholder(n) else range(n))
    return x


def test_a_recipe_instantiates_at_a_size_it_never_saw():
    core, n = exa.recipe(nargs=1)
    rosen(core, n)

    m = exa.Model(core, 7)
    assert m.nvar == 7
    # Built once, used twice: the first model must survive the second.
    other = exa.Model(core, 3)
    assert (other.nvar, m.nvar) == (3, 7)


def test_a_recipe_matches_the_same_model_built_with_real_sizes():
    """The only check that catches a recipe which builds but means something else."""
    core, n = exa.recipe(nargs=1)
    rosen(core, n)
    recipe_model = exa.Model(core, 6)

    plain = exa.Core()
    rosen(plain, 6)
    reference = exa.Model(plain)

    pt = np.linspace(0.5, 3.0, 6)
    assert recipe_model.nvar == reference.nvar
    assert recipe_model.objective(pt) == pytest.approx(reference.objective(pt))
    assert np.allclose(np.asarray(recipe_model.gradient(pt)),
                       np.asarray(reference.gradient(pt)))
    assert np.allclose(np.asarray(recipe_model.x0), np.asarray(reference.x0))


def test_placeholders_reach_starts_and_bounds():
    core, n, start, lower = exa.recipe(nargs=3)
    core.add_var(n, start=start, lvar=lower)
    m = exa.Model(core, 3, [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0])
    assert np.allclose(np.asarray(m.x0), [4.0, 5.0, 6.0])
    assert np.allclose(np.asarray(m.lvar), [-1.0, -2.0, -3.0])


def test_arithmetic_on_a_placeholder_is_deferred():
    core, n = exa.recipe(nargs=1)
    x = core.add_var(n, start=0.5)
    # `n - 1` rows, not `n` — the size is computed when the model is built.
    core.add_con(lambda i: x[i] + x[i + 1], over=exa.srange(0, n - 1),
                 lcon=0.0, ucon=10.0)
    assert exa.Model(core, 5).ncon == 4
    assert exa.Model(core, 9).ncon == 8


def test_reflected_operators_work():
    """`2 * n` is as natural to write as `n * 2`, so both must build."""
    core, n = exa.recipe(nargs=1)
    core.add_var(2 * n, start=0.0)
    assert exa.Model(core, 3).nvar == 6


def test_a_core_without_placeholders_is_unchanged():
    core = exa.Core()
    assert core.args == ()
    x = core.add_var(4, start=1.0)
    core.add_obj(lambda i: x[i] ** 2, over=range(4))
    assert exa.Model(core).nvar == 4


@pytest.mark.parametrize("doing", [
    lambda n: len(n),
    lambda n: int(n),
    lambda n: [0, 1][n],
    lambda n: list(n),
    lambda n: bool(n),
    lambda n: n > 3,
])
def test_what_a_placeholder_refuses(doing):
    """Each of these has a value only after instantiation.

    They matter more in Python than in Julia: silently satisfying `__index__` or
    `__bool__` would produce a model whose shape depended on a value nobody had
    supplied, and the failure would surface far from its cause.
    """
    _, n = exa.recipe(nargs=1)
    with pytest.raises(TypeError, match="placeholder"):
        doing(n)


def test_srange_says_why_it_has_no_length():
    _, n = exa.recipe(nargs=1)
    with pytest.raises(TypeError, match="until the model is built"):
        len(exa.srange(0, n))


def test_a_core_with_nothing_deferred_is_a_fixed_model():
    """Not an error any more: the compiler takes a fixed core as it stands.

    It used to be refused here for having no placeholders. The backend gained
    fixed models -- compiled as-is, instantiated with no arguments -- so the
    refusal would now reject something the compiler supports. What reaches the
    backend for such a core is checked in test_compile.py.
    """
    core = exa.Core()
    core.add_var(3, start=0.0)
    assert core.args == ()
