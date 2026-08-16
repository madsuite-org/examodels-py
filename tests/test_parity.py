"""Coverage of every name ExaModels exports.

The list is read from the backend at run time, not transcribed, so a name added
or removed upstream shows up here immediately. Every export must be classified;
an unclassified one fails, which forces a decision rather than silent drift.

Classification is only half of it — each name marked SUPPORTED is exercised for
real by `test_every_supported_export_works` or by the suites it points at.
"""
from collections import namedtuple

import numpy as np
import pytest

import examodels as exa
from examodels import _bridge as _b

SUPPORTED = "supported"      # reachable from Python and exercised below
SYNTAX = "syntax"            # a Julia macro; the Python spelling is different
LEGACY = "legacy"            # the backend's own deprecated API
JULIA = "julia-only"         # needs Julia code to use at all
INTERNAL = "internal"        # not part of any user-facing surface
ABSENT = "absent"            # no Python surface yet

#: exported name -> (category, how it is reached from Python / why it is not)
SURFACE = {
    # -- construction -------------------------------------------------------
    "ExaCore": (SUPPORTED, "exa.Core"),
    "ExaModel": (SUPPORTED, "exa.Model"),
    # -- recipes: a core written against placeholders, given its data later --
    "ArgSource": (SUPPORTED, "exa.Core(nargs=...) / exa.recipe"),
    "instantiate": (SUPPORTED, "exa.Model(core, *values)"),
    "add_var": (SUPPORTED, "Core.add_var"),
    "add_par": (SUPPORTED, "Core.add_par"),
    "add_obj": (SUPPORTED, "Core.add_obj"),
    "add_con": (SUPPORTED, "Core.add_con"),
    "add_con!": (SUPPORTED, "Core.add_con(handle, ...)"),
    "add_expr": (SUPPORTED, "Core.add_expr"),
    "Expression": (SUPPORTED, "returned by Core.add_expr"),
    # -- expression nodes ---------------------------------------------------
    "Constant": (SUPPORTED, "exa.Constant"),
    "SumNode": (SUPPORTED, "exa.sum"),
    "ProdNode": (SUPPORTED, "exa.prod"),
    "exa_sum": (SUPPORTED, "exa.sum"),
    "exa_prod": (SUPPORTED, "exa.prod"),
    # -- results ------------------------------------------------------------
    "solution": (SUPPORTED, "sol[block]"),
    "multipliers": (SUPPORTED, "Solution.multipliers"),
    "multipliers_L": (SUPPORTED, "Solution.multipliers_L"),
    "multipliers_U": (SUPPORTED, "Solution.multipliers_U"),
    # -- getters and setters ------------------------------------------------
    **{n: (SUPPORTED, f"Model.{n.rstrip('!').replace('set_', 'set_')}")
       for n in ("get_value", "set_value!", "get_start", "set_start!",
                 "get_lvar", "set_lvar!", "get_uvar", "set_uvar!",
                 "get_lcon", "set_lcon!", "get_ucon", "set_ucon!")},
    # -- named blocks: the backend publishes them; Python keeps the same
    # names reachable as attributes on the core and the built model
    "get_vars": (SUPPORTED, "core.<name> / model.<name>"),
    "get_cons": (SUPPORTED, "core.<name> / model.<name>"),
    "get_pars": (SUPPORTED, "core.<name> / model.<name>"),
    # -- macros: Python spells these as generator expressions ---------------
    **dict.fromkeys(
        ("@add_var", "@add_par", "@add_obj", "@add_con", "@add_con!", "@add_expr"),
        (SYNTAX, "generator expression or lambda + over=")),
    # -- needs Julia ---------------------------------------------------------
    "@register_univariate": (JULIA, "registering an operator needs a Julia function"),
    "@register_bivariate": (JULIA, "registering an operator needs a Julia function"),
    # The backend's deprecated mutating API — LegacyExaCore, variable,
    # parameter, constraint!, subexpr — was removed upstream in
    # exanauts/ExaModels.jl#295, along with deprecated.jl itself, so there is
    # nothing left here to classify. The LEGACY category is kept for whatever
    # is deprecated next.
    "ExaModels": (INTERNAL, "the module itself"),
    # -- nonlinear oracles ---------------------------------------------------
    "VectorNonlinearOracle": (SUPPORTED, "exa.VectorNonlinearOracle"),
    "ScalarNonlinearOracle": (SUPPORTED, "exa.ScalarNonlinearOracle"),
    "OracleEvaluator": (SUPPORTED, "exa.OracleEvaluator"),
    "ExaModelWithOracle": (SUPPORTED, "built by Model(core) when an oracle is present"),
    "embed_oracle": (SUPPORTED, "exa.embed_oracle"),
    "add_eval": (SUPPORTED, "exa.add_eval"),
    "has_matfree_jac": (SUPPORTED, "exa.has_matfree_jac"),
    "has_matfree_hess": (SUPPORTED, "exa.has_matfree_hess"),
    "constraint": (SUPPORTED, "add_con(core, oracle) — the oracle registration point"),
    "objective": (SUPPORTED, "add_obj(core, oracle) — the oracle registration point"),
    # -- two-stage stochastic models -----------------------------------------
    "TwoStageExaCore": (SUPPORTED, "exa.TwoStageCore"),
    "TwoStageExaModel": (SUPPORTED, "Model(TwoStageCore(...))"),
    "EachScenario": (SUPPORTED, "exa.EachScenario"),
    "get_nscen": (SUPPORTED, "exa.get_nscen"),
    "get_var_scen": (SUPPORTED, "exa.get_var_scen"),
    "get_con_scen": (SUPPORTED, "exa.get_con_scen"),
    "SecondStageVariable": (SUPPORTED, "a block declared with EachScenario()"),
    # -- tags -----------------------------------------------------------------
    "AbstractVariableTag": (SUPPORTED, "exa.new_tag(name, 'variable')"),
    "AbstractConstraintTag": (SUPPORTED, "exa.new_tag(name, 'constraint')"),
    **{n: (SUPPORTED, f"exa.{n}()") for n in
       ("FirstStageTag", "SecondStageTag",
        "FirstStageConstraintTag", "SecondStageConstraintTag")},
    # -- NLPModel wrappers ----------------------------------------------------
    **{n: (SUPPORTED, f"exa.{n}") for n in
       ("CompressedNLPModel", "TimedNLPModel", "WrapperNLPModel")},
}


def exported():
    return [str(n) for n in _b.seval("[string(n) for n in names(ExaModels)]")]


def test_every_export_is_classified():
    """A name added upstream must be classified before this suite goes green again."""
    missing = sorted(set(exported()) - set(SURFACE))
    assert not missing, f"unclassified ExaModels exports: {missing}"


def test_the_classification_has_no_stale_entries():
    stale = sorted(set(SURFACE) - set(exported()))
    assert not stale, f"classified names the backend no longer exports: {stale}"


def test_coverage_summary(capsys):
    names = exported()
    counts = {}
    for n in names:
        counts.setdefault(SURFACE[n][0], []).append(n)
    with capsys.disabled():
        print(f"\n  {len(names)} exported names:")
        for cat in (SUPPORTED, SYNTAX, LEGACY, JULIA, INTERNAL, ABSENT):
            got = counts.get(cat, [])
            print(f"    {cat:11} {len(got):3}"
                  + (f"   {', '.join(sorted(got)[:4])}"
                     f"{' ...' if len(got) > 4 else ''}" if got else ""))
    # a floor, not a target: this must never go down without someone noticing
    reachable = len(counts.get(SUPPORTED, [])) + len(counts.get(SYNTAX, []))
    in_scope = len(names) - len(counts.get(LEGACY, [])) - len(counts.get(JULIA, [])) \
        - len(counts.get(INTERNAL, []))
    with capsys.disabled():
        print(f"    -> {reachable} of {in_scope} in-scope exports reachable from Python")
    assert reachable == in_scope, (reachable, in_scope)


def test_every_supported_export_works():
    """One model that exercises every SUPPORTED name end to end."""
    Row = namedtuple("Row", "i c")
    rows = [Row(k, 1.0 + k) for k in range(3)]

    core = exa.Core()                                     # ExaCore
    th = core.add_par([2.0, 3.0])                         # add_par
    x = core.add_var(3, start=0.5, lvar=0.0, uvar=2.0)    # add_var
    y = core.add_var(2, 3, start=0.5)                     # add_var, multi-dim
    sq = core.add_expr(x[i]**2 for i in range(3))         # add_expr -> Expression

    core.add_obj(th[0] * sq[r.i] * r.c for r in rows)     # add_obj, Expression, Records
    core.add_obj(exa.sum([x[k] for k in range(3)]), over=range(1))          # exa_sum
    core.add_obj(exa.prod([x[0], x[1]]) * exa.Constant(1), over=range(1))   # exa_prod, Constant
    con = core.add_con((x[i] + x[i + 1] for i in range(2)),
                       lcon=0.5, ucon=1.5)                # add_con
    two = [Row(k, 1.0) for k in range(2)]
    core.add_con(con, ((r.i, th[1] * x[r.i]) for r in two))                # add_con!
    core.add_con(y[t, i] - y[t - 1, i]
                 for t, i in exa.product(range(1, 2), range(3)))            # product

    model = exa.Model(core)                               # ExaModel

    # getters
    np.testing.assert_allclose(model.get_value(th), [2.0, 3.0])            # get_value
    np.testing.assert_allclose(model.get_start(x), 0.5)                    # get_start
    np.testing.assert_allclose(model.get_lvar(x), 0.0)                     # get_lvar
    np.testing.assert_allclose(model.get_uvar(x), 2.0)                     # get_uvar
    np.testing.assert_allclose(model.get_lcon(con), 0.5)                   # get_lcon
    np.testing.assert_allclose(model.get_ucon(con), 1.5)                   # get_ucon

    # setters
    model.set_value(th, [2.5, 3.5])                                        # set_value!
    model.set_start(x, [0.4, 0.4, 0.4])                                    # set_start!
    model.set_lvar(x, [0.0, 0.0, 0.0])                                     # set_lvar!
    model.set_uvar(x, [3.0, 3.0, 3.0])                                     # set_uvar!
    model.set_lcon(con, [0.4, 0.4])                                        # set_lcon!
    model.set_ucon(con, [1.6, 1.6])                                        # set_ucon!
    np.testing.assert_allclose(model.get_value(th), [2.5, 3.5])
    np.testing.assert_allclose(model.get_uvar(x), 3.0)

    sol = model.solve()
    assert sol.success, sol.status
    assert sol[x].shape == (3,)                                            # solution
    assert sol[y].shape == (2, 3)
    assert sol.multipliers(con).shape == (2,)                              # multipliers
    assert sol.multipliers_L(x).shape == (3,)                              # multipliers_L
    assert sol.multipliers_U(x).shape == (3,)                              # multipliers_U
    assert model.violation(sol.x) < 1e-6


def test_set_value_is_the_parameter_setter():
    """`Model.set_value` is the Python spelling of the backend's set_value!."""
    core = exa.Core()
    th = core.add_par([1.0])
    x = core.add_var(1, start=0.0)
    core.add_obj((x[0] - th[0])**2, over=range(1))
    model = exa.Model(core)
    assert model.solve()[x][0] == pytest.approx(1.0)
    model.set_value(th, [4.0])
    assert model.solve()[x][0] == pytest.approx(4.0)


def test_a_filtered_generator_expression_is_refused():
    """`... for i in r if cond` cannot be traced: the index has no value yet."""
    core = exa.Core()
    x = core.add_var(4)
    with pytest.raises(TypeError):
        core.add_obj(x[i]**2 for i in range(4) if i > 1)
