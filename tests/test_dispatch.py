"""Solver dispatch: how `solve` picks a solver, defaults its options, and fails.

The ipopt path runs for real throughout the suite; what is tested here is the
routing logic itself -- madnlp defaults, device selection, the not-installed
message -- with the backend stubbed where a GPU (or an uninstalled solver)
would otherwise be required. The stubs replace `_bridge` attributes, which the
`solve` module reads at call time, so each test exercises the real dispatch
code with a scripted backend.
"""
import importlib
import types

import pytest
from conftest import requires

import examodels as exa
from examodels import _bridge as _b
from examodels.model import Solution
from examodels.solve import available_solvers, solve

# `examodels.solve` the attribute is the function; the module is fetched by name
solve_mod = importlib.import_module("examodels.solve")


@pytest.fixture
def model():
    core = exa.Core()
    x = core.add_var(2, start=0.5)
    core.add_obj(lambda i: (x[i] - 1) ** 2, over=range(2))
    return core.build()


def test_available_solvers_is_sorted_and_complete():
    assert available_solvers() == ["ipopt", "madnlp"]


def test_unknown_solver_names_the_choices(model):
    with pytest.raises(ValueError, match=r"unknown solver 'newton'.*ipopt.*madnlp"):
        solve(model, solver="newton")


def test_missing_solver_says_how_to_install_it(model, monkeypatch):
    # A solver whose backend package cannot load -- the real seval refuses it.
    monkeypatch.setitem(solve_mod.SOLVERS, "fake",
                        ("DefinitelyNotAJuliaPackage", "0000", "fake"))
    with pytest.raises(_b.ModelError, match=r"install_solver\('fake'\)"):
        solve(model, solver="fake")


def test_install_solver_adds_the_named_package(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(add=lambda *a, **k: calls.append(("add", a, k)),
                                 resolve=lambda: calls.append(("resolve",)))
    monkeypatch.setitem(__import__("sys").modules, "juliapkg", fake)
    solve_mod.install_solver("madnlp")
    assert calls == [("add", ("MadNLP", "2621e9c9-9eb4-46b1-8089-e8c72242dfb6"), {}),
                     ("resolve",)]


@requires("ipopt")
def test_a_host_model_defaults_to_ipopt_quietly(model):
    sol = model.solve()          # no solver named: host arrays -> ipopt
    assert sol.success
    assert sol.elapsed >= 0.0


class _Recorder:
    """Scripted stand-ins for the bridge attributes `solve` touches."""

    def __init__(self, monkeypatch, on_device):
        self.sevals, self.solved = [], []
        self.raw = object()
        monkeypatch.setattr(solve_mod, "_on_device", lambda m: on_device)
        monkeypatch.setattr(_b, "seval", self._seval, raising=False)
        monkeypatch.setattr(_b, "guard", self._guard, raising=False)
        monkeypatch.setattr(_b, "jl", types.SimpleNamespace(
            ipopt="IPOPT-ENTRY", madnlp="MADNLP-ENTRY"), raising=False)
        # dispatch must believe the solver is loaded, or it hits the real seval
        monkeypatch.setattr(solve_mod, "_loaded", {"ipopt", "madnlp"})

    def _seval(self, code):
        self.sevals.append(code)
        return f"SEVAL({code})"

    def _guard(self, fn, *args, **kwargs):
        self.solved.append((fn, args, kwargs))
        return self.raw


def test_nothing_is_set_on_the_solver_s_behalf(model, monkeypatch):
    # The solver's own defaults stand: a caller who wants it quiet says so, and
    # a caller who says nothing gets the reporting the solver normally does.
    rec = _Recorder(monkeypatch, on_device=False)
    sol = solve(model, solver="madnlp")
    (fn, args, options), = rec.solved
    assert fn == "MADNLP-ENTRY" and args == (model._jl,)
    assert options == {}                          # nothing invented
    assert not any("MadNLP.ERROR" in s for s in rec.sevals)
    assert isinstance(sol, Solution) and sol._raw is rec.raw

    rec.solved.clear()
    solve(model, solver="ipopt")
    (fn, _args, options), = rec.solved
    assert fn == "IPOPT-ENTRY"
    assert options == {}                          # no print_level, no sb


def test_solver_options_are_passed_through_verbatim(model, monkeypatch):
    rec = _Recorder(monkeypatch, on_device=False)
    solve(model, solver="ipopt", print_level=5, sb="no", tol=1e-9)
    (_fn, _args, options), = rec.solved
    assert options == {"print_level": 5, "sb": "no", "tol": 1e-9}


def test_a_device_model_routes_to_madnlp_with_a_device_solver(model, monkeypatch):
    rec = _Recorder(monkeypatch, on_device=True)
    monkeypatch.setattr(solve_mod, "_device_linear_solver", lambda: "DEVICE-LS")
    solve(model)                                 # no solver named
    (fn, _args, options), = rec.solved
    assert fn == "MADNLP-ENTRY"                  # chosen by where the arrays live
    # the one thing still chosen for the caller -- a host linear solver cannot
    # run on device arrays at all -- and naming one still wins
    assert options["linear_solver"] == "DEVICE-LS"
    rec.solved.clear()
    solve(model, linear_solver="MINE")
    (_fn, _args, options), = rec.solved
    assert options["linear_solver"] == "MINE"


def test_device_solver_prefers_cudss_and_reports_when_none_load(model, monkeypatch):
    # Scripted environment: the full package trio fails to load, the pair loads,
    # CUDSSSolver is unassigned, LapackCUDASolver is assigned.
    def seval(code):
        if code == "using MadNLPGPU, CUDA, CUDSS":
            raise RuntimeError("no CUDSS here")
        if "CUDSSSolver" in code and code.startswith("try"):
            return False
        if "LapackCUDASolver" in code and code.startswith("try"):
            return True
        return f"SEVAL({code})"

    monkeypatch.setattr(_b, "seval", seval, raising=False)
    assert solve_mod._device_linear_solver() == "SEVAL(MadNLPGPU.LapackCUDASolver)"

    # And when nothing is assigned, the error says what to install.
    monkeypatch.setattr(_b, "seval",
                        lambda code: False if code.startswith("try") else None,
                        raising=False)
    with pytest.raises(_b.ModelError, match="install_backend"):
        solve_mod._device_linear_solver()


@requires("ipopt")
def test_on_device_is_false_for_a_host_model(model):
    assert solve_mod._on_device(model) is False
