"""The compile front end: argument checking, and what reaches the backend.

Actually producing a shared library needs the compiler backend and a Julia able
to run it, which the test environment does not promise. The contract tested
here is everything up to that point: what is refused, what the error messages
say to do, and -- with the backend stubbed -- exactly what a well-formed call
passes through.
"""
import sys
import types

import pytest

import examodels as exa
from examodels import _bridge as _b
from examodels import compile as compile_mod
from examodels.compile import COMPILER, compile_library, compiler_available, install_compiler


def test_only_a_core_can_be_compiled():
    with pytest.raises(TypeError, match="Core built with nargs="):
        compile_library({"not": "a core"}, "lib", arg=1)


def test_a_core_without_placeholders_is_refused():
    core = exa.Core()
    core.add_var(3)
    with pytest.raises(ValueError, match=r"Core\(nargs=1\)"):
        compile_library(core, "lib", arg=1)


@pytest.mark.skipif(compiler_available(),
                    reason="the compiler backend happens to be installed here")
def test_a_missing_compiler_says_how_to_install_it():
    core, n = exa.recipe()
    x = core.add_var(n)
    core.add_obj(lambda i: x[i] ** 2, over=exa.srange(0, n))
    with pytest.raises(RuntimeError, match=r"install_compiler\(\)"):
        compile_library(core, "lib", arg=4)


def test_a_well_formed_call_reaches_the_backend_as_declared(monkeypatch):
    core, n = exa.recipe()
    x = core.add_var(n)
    core.add_obj(lambda i: x[i] ** 2, over=exa.srange(0, n))

    recorded = {}

    def guard(fn, *args, **kwargs):
        recorded["fn"], recorded["args"], recorded["kw"] = fn, args, kwargs
        return "COMPILED"

    def at_field(got, field):
        assert got == "COMPILED" and field == "libpath"
        return "/somewhere/librosen.so"

    monkeypatch.setattr(compile_mod, "compiler_available", lambda: True)
    monkeypatch.setattr(_b, "guard", guard, raising=False)
    monkeypatch.setattr(_b, "at_field", at_field, raising=False)

    from pathlib import Path
    out = compile_library(core, Path("/tmp/out"), arg=10, prefix=Path("/p"),
                          bundle=False, verbose=True)

    assert out == "/somewhere/librosen.so"
    assert recorded["args"] == (core._core, "/tmp/out")   # out arrives as str
    assert recorded["kw"] == {"arg": 10, "bundle": False, "verbose": True,
                              "prefix": "/p"}             # prefix arrives as str


def test_prefix_is_omitted_rather_than_passed_as_none(monkeypatch):
    core, n = exa.recipe()
    x = core.add_var(n)
    core.add_obj(lambda i: x[i] ** 2, over=exa.srange(0, n))

    kw_seen = {}
    monkeypatch.setattr(compile_mod, "compiler_available", lambda: True)
    monkeypatch.setattr(_b, "guard",
                        lambda fn, *a, **kw: kw_seen.update(kw) or "GOT",
                        raising=False)
    monkeypatch.setattr(_b, "at_field", lambda got, f: "lib.so", raising=False)

    compile_library(core, "name", arg=2)
    assert "prefix" not in kw_seen
    assert kw_seen["bundle"] is True                      # the documented default


def test_install_compiler_names_the_unregistered_source(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(add=lambda *a, **k: calls.append(("add", a, k)),
                                 resolve=lambda: calls.append(("resolve",)))
    monkeypatch.setitem(sys.modules, "juliapkg", fake)
    install_compiler()
    assert calls == [
        ("add", (COMPILER["name"], COMPILER["uuid"]),
         {"url": COMPILER["url"], "subdir": COMPILER["subdir"]}),
        ("resolve",),
    ]
