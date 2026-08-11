"""The advanced module's remaining surface: reprs, oracle plumbing, host guards.

The oracle *mathematics* is covered by the definition tests; here the plumbing
around it -- how handles unwrap on the way into the backend, and what the
wrappers hand back -- is pinned with the backend recorded rather than run.
"""
import pytest

import examodels as exa
from examodels import _bridge as _b
from examodels import advanced


def test_marker_and_oracle_reprs():
    assert repr(advanced.EachScenario()) == "EachScenario()"
    oracle = exa.VectorNonlinearOracle(
        nvar=2, ncon=1, f=lambda c, x: None,
        jvp=lambda *a: None, vjp=lambda *a: None, hvp=lambda *a: None)
    assert repr(oracle) == "<constraint oracle>"


def test_as_cupy_refuses_a_host_array_before_needing_cupy():
    core = exa.Core()
    core.add_var(3)
    host = _b.like(core.build()._jl, 3)          # a backend host vector
    with pytest.raises(TypeError, match="host memory"):
        exa.as_cupy(host)


class _Handle:
    def __init__(self, jl):
        self._jl = jl


def test_add_eval_unwraps_handles_and_rebinds_the_core(monkeypatch):
    recorded = {}

    def guard(fn, *args, **kwargs):
        recorded["args"], recorded["kw"] = args, kwargs
        return "NEWCORE", "EVAL-HANDLE"

    monkeypatch.setattr(_b, "guard", guard, raising=False)
    core = _Handle(None)
    core._core = "OLDCORE"
    out = advanced.add_eval(core, [_Handle("C1"), _Handle("C2")], [_Handle("V")],
                            "CALLBACK", extra=1)
    assert out == "EVAL-HANDLE"
    assert core._core == "NEWCORE"               # the returned core replaces the old
    assert recorded["args"] == ("OLDCORE", ("C1", "C2"), ("V",), "CALLBACK")
    assert recorded["kw"] == {"extra": 1}


def test_embed_oracle_passes_the_block_and_output_arity(monkeypatch):
    recorded = {}

    def guard(fn, *args, **kwargs):
        recorded["args"], recorded["kw"] = args, kwargs
        return "NEWCORE", "ORACLE-OUT"

    monkeypatch.setattr(_b, "guard", guard, raising=False)
    core = _Handle(None)
    core._core = "OLDCORE"
    out = advanced.embed_oracle(core, _Handle("BLOCK"), 3.0, adapt=True)
    assert out == "ORACLE-OUT" and core._core == "NEWCORE"
    assert recorded["args"] == ("OLDCORE", "BLOCK", 3)     # arity arrives as int
    assert recorded["kw"] == {"adapt": True}


def test_oracle_evaluator_forwards_to_the_backend(monkeypatch):
    seen = {}
    monkeypatch.setattr(_b, "guard",
                        lambda fn, *a, **kw: seen.update(a=a, kw=kw) or "EVALUATOR",
                        raising=False)
    assert advanced.OracleEvaluator(1, two=2) == "EVALUATOR"
    assert seen["a"] == (1,) and seen["kw"] == {"two": 2}
