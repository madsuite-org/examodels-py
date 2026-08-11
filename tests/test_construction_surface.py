"""Construction-time surface not covered by the modelling tests.

Backend selection and installation, placeholder arithmetic in every spelling
Python offers, bounds and tags on placeholder-sized blocks, and the operator
namespace the package generates from the backend's own list.
"""
import sys
import types

import numpy as np
import pytest
from conftest import requires

import examodels as exa
from examodels import _bridge as _b
from examodels import core as core_mod
from examodels import ops
from examodels.recipe import is_placeholder


# ---------------------------------------------------------------- backends ----
def test_backends_lists_every_constructible_kind():
    assert exa.backends() == ["cpu", "cuda", "metal", "oneapi", "rocm", "serial"]


def test_serial_is_the_default_and_needs_no_package():
    core = exa.Core(backend="serial")            # explicitly what None means
    x = core.add_var(2)
    core.add_obj(lambda i: x[i] ** 2, over=range(2))
    assert core.build().nvar == 2


def test_unknown_backends_name_the_choices():
    with pytest.raises(ValueError, match="unknown backend 'tpu'"):
        exa.Core(backend="tpu")


def test_a_loadable_backend_is_constructed_once(monkeypatch):
    sevals = []

    def seval(code):
        sevals.append(code)
        return "KA-BACKEND" if "()" in code else None

    monkeypatch.setattr(_b, "seval", seval, raising=False)
    monkeypatch.setattr(core_mod, "_loaded", set())
    assert core_mod._backend("cpu") == "KA-BACKEND"
    assert sevals == ["using KernelAbstractions", "KernelAbstractions.CPU()"]
    assert core_mod._backend("cpu") == "KA-BACKEND"        # loaded once,
    assert sevals.count("using KernelAbstractions") == 1   # constructed per call


def test_an_unloadable_backend_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(core_mod, "_loaded", set())
    if "KernelAbstractions" not in str(_b.seval('Base.find_package("KernelAbstractions")')):
        with pytest.raises(_b.ModelError, match=r"install_backend\('cpu'\)"):
            exa.Core(backend="cpu")
    else:
        pytest.skip("KernelAbstractions happens to be installed here")


def test_install_backend_refuses_what_needs_no_installing():
    with pytest.raises(ValueError, match="nothing to install"):
        exa.install_backend("serial")
    with pytest.raises(ValueError, match="nothing to install"):
        exa.install_backend("tpu")


def test_install_backend_adds_the_named_package(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(add=lambda *a, **k: calls.append(("add", a, k)),
                                 resolve=lambda: calls.append(("resolve",)))
    monkeypatch.setitem(sys.modules, "juliapkg", fake)
    exa.install_backend("cuda")
    assert calls == [("add", ("CUDA", core_mod._BACKEND_UUIDS["CUDA"]), {}),
                     ("resolve",)]


# ------------------------------------------------------------ placeholders ----
def test_placeholder_arithmetic_in_every_spelling():
    core, n = exa.recipe()
    for expr in (n + 1, 1 + n, n - 1, 1 - n, n * 2, 2 * n,
                 n / 2, 2 / n, -n, n ** 2):
        assert is_placeholder(expr), expr
    assert repr(n).startswith("<placeholder ")
    assert repr(exa.srange(0, n)) == f"srange(0, {n!r})"


@requires("ipopt")
def test_deferred_sizes_compute_at_instantiation():
    # A composite bound (`2 * n`) sizes a *block*; an index set iterated by the
    # backend needs a plain placeholder bound, so the objective runs over `n`.
    core, n = exa.recipe()
    x = core.add_var(2 * n, start=0.0)
    core.add_obj(lambda i: (x[i] - 1) ** 2, over=exa.srange(0, n))
    model = exa.Model(core, 3)
    assert model.nvar == 6                       # 2 * 3
    sol = model.solve()
    np.testing.assert_allclose(sol.x[:3], np.ones(3), atol=1e-6)
    np.testing.assert_allclose(sol.x[3:], np.zeros(3), atol=1e-6)   # no objective there


@requires("ipopt")
def test_placeholder_blocks_refuse_block_reads_readably():
    # Neither this package nor ExaModels.jl itself can index a result by a
    # placeholder-sized block (verified upstream); what this package owes the
    # caller is an error that names the way that works.
    core, n = exa.recipe()
    x = core.add_var(n, start=0.0)
    core.add_obj(lambda i: (x[i] - 1) ** 2, over=exa.srange(0, n))
    model = exa.Model(core, 3)
    assert repr(x) == "<variable block ?>"       # sized at build, printable before
    sol = model.solve()
    with pytest.raises(Exception, match=r"sol\.x"):
        sol[x]
    with pytest.raises(Exception, match=r"model\.x0"):
        model.get_start(x)
    with pytest.raises(Exception, match="placeholder"):
        model.set_start(x, np.zeros(3))          # the Arg refusal fires on .shape


def test_placeholder_blocks_take_bounds_and_tags():
    core, n = exa.recipe()
    t = exa.new_tag("CoverageTag")
    x = core.add_var(n, start=0.5, lvar=0.0, uvar=2.0, tag=t)
    core.add_obj(lambda i: x[i] ** 2, over=exa.srange(0, n))
    model = exa.Model(core, 4)
    np.testing.assert_allclose(model.uvar, np.full(4, 2.0))
    np.testing.assert_allclose(model.lvar, np.zeros(4))


def test_defined_variables_need_a_range_index_set():
    core = exa.Core()
    y = core.add_var(3)
    with pytest.raises(TypeError, match="range index set"):
        core.add_var(y[i] ** 2 for i in [0, 1])


# ------------------------------------------------------------ the operators ---
def test_the_bivariate_operators_trace_like_the_univariate_ones():
    biv = [name for name in ops.__all__ if "bivariate" in getattr(ops, name).__doc__]
    assert biv                                    # the backend registers some
    core = exa.Core()
    x = core.add_var(2, start=[3.0, 4.0])
    f = getattr(ops, biv[0])
    node = f(x[0], x[1])
    assert type(node).__name__ == "Node"
    core.add_obj(lambda i: node, over=range(1))
    assert np.isfinite(core.build().objective([3.0, 4.0]))


def test_the_package_dir_includes_the_generated_operators():
    names = dir(exa)
    assert {"sin", "exp", "hypot"} <= set(names)
