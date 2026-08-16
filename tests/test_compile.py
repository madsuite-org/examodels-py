"""The compile front end: argument checking, and what reaches the backend.

Actually producing a shared library needs the compiler backend and a Julia able
to run it (1.12), which the test environment does not promise. The contract
tested here is everything up to that point: what is refused, what the error
messages say to do, and -- with the backend stubbed -- exactly what a
well-formed call passes through, in each of the three shapes the compiler
accepts (a recipe with examples, a fixed core, several models in one library).
"""
import os
import sys
import types

import pytest

import examodels as exa
from examodels import _bridge as _b
from examodels import compile as compile_mod
from examodels.compile import (
    COMPILER,
    CompiledLibrary,
    compile_library,
    compiler_available,
    install_compiler,
)


@pytest.fixture
def recipe_core():
    core, n = exa.recipe()
    x = core.add_var(n)
    core.add_obj(lambda i: x[i] ** 2, over=exa.srange(0, n))
    return core


@pytest.fixture
def fixed_core():
    core = exa.Core()
    x = core.add_var(3)
    core.add_obj(lambda i: x[i] ** 2, over=range(3))
    return core


@pytest.fixture
def backend(monkeypatch):
    """The compiler, recorded rather than run."""
    calls = []

    def record(which):
        def fn(*args, **kwargs):
            calls.append((which, args, kwargs))
            return which
        return fn

    monkeypatch.setattr(compile_mod, "compiler_available", lambda: True)
    monkeypatch.setattr(_b, "compile_library", record("one"), raising=False)
    monkeypatch.setattr(_b, "compile_models", record("many"), raising=False)
    monkeypatch.setattr(_b, "guard", lambda fn, *a, **kw: fn(*a, **kw), raising=False)
    monkeypatch.setattr(_b, "at_field",
                        lambda got, f: {"libpath": "/out/libm.so", "outdir": "/out",
                                        "prefix": "libm"}[f],
                        raising=False)
    return calls


# ------------------------------------------------------------- refusals ------
def test_only_a_core_can_be_compiled(backend):
    with pytest.raises(TypeError, match="expected a Core"):
        compile_library("lib", {"not": "a core"})


def test_a_bad_model_in_a_mapping_is_named(backend):
    with pytest.raises(TypeError, match="model 'two': expected a Core"):
        compile_library("lib", {"two": 42})


@pytest.mark.skipif(compiler_available(),
                    reason="the compiler backend happens to be installed here")
def test_a_missing_compiler_says_how_to_install_it(recipe_core):
    with pytest.raises(RuntimeError, match=r"install_compiler\(\)"):
        compile_library("lib", recipe_core, 4)


def test_an_empty_mapping_is_refused(backend):
    with pytest.raises(ValueError, match="at least one model"):
        compile_library("lib", {})


def test_per_model_keywords_are_refused_for_several_models(backend, fixed_core):
    with pytest.raises(TypeError, match="prefix.*no meaning"):
        compile_library("lib", {"a": fixed_core}, prefix="p")
    with pytest.raises(TypeError, match="argfun"):
        compile_library("lib", {"a": fixed_core}, argfun=_b.seval("x -> (x,)"))
    with pytest.raises(TypeError, match="rather than positionally"):
        compile_library("lib", {"a": fixed_core}, 10)


def test_a_python_argfun_is_refused_with_the_route_that_works(backend, recipe_core):
    # Not a Julia-vs-Python nicety: the compiler resolves argfun BY NAME out of
    # a package, so even a Julia function built through the bridge is refused
    # (verified against the compiler itself). A Python caller wants the other
    # route, and the message has to say so rather than send them round a loop.
    with pytest.raises(TypeError) as e:
        compile_library("lib", recipe_core, 4, argfun=lambda n: (n,))
    assert "no Python in it" in str(e.value)
    assert "Pass the data as example values" in str(e.value)
    assert "seval" in str(e.value)              # names the thing that also fails


def test_a_non_callable_argfun_is_refused_too(backend, recipe_core):
    # `callable()` was the old guard, so a non-callable slipped straight past
    # it and reached the backend as a keyword it cannot use. (A string is not
    # in this class: it NAMES a package function, and is resolved below.)
    with pytest.raises(TypeError, match="no Python in it"):
        compile_library("lib", recipe_core, 4, argfun=42)


# --------------------------------------------------------- what is passed ----
def test_a_recipe_with_examples_reaches_the_backend_as_declared(backend, recipe_core):
    from pathlib import Path
    lib = compile_library(Path("/tmp/out"), recipe_core, 10, bundle=True, verbose=True,
                          trim="unsafe", prefix=Path("/p"))
    (which, args, kw), = backend
    assert which == "one"
    assert args == ("/tmp/out", recipe_core._core, [10])   # out first, examples splatted
    assert kw == {"trim": "unsafe", "bundle": True, "verbose": True, "prefix": "/p"}
    assert lib.path == "/out/libm.so" and lib.prefixes == ("libm",)


def test_a_fixed_core_compiles_with_no_examples(backend, fixed_core):
    compile_library("fixed", fixed_core)
    (_which, args, kw), = backend
    assert args == ("fixed", fixed_core._core, [])         # no examples: a fixed model
    assert kw == {"trim": "safe", "bundle": False, "verbose": False}   # the defaults
    assert "prefix" not in kw and "argfun" not in kw       # omitted, not passed as None


def test_several_models_go_through_the_multi_model_entry(backend, recipe_core, fixed_core):
    lib = compile_library("@grid", {"acopf": (recipe_core, 100), "small": fixed_core})
    (which, args, kw), = backend
    assert which == "many"
    names, cores, argss, out = args
    assert names == ["acopf", "small"]                     # insertion order kept
    assert cores == [recipe_core._core, fixed_core._core]
    assert argss == [[100], []]                            # each carries its own
    assert out == "@grid"
    assert kw == {"trim": "safe", "bundle": False, "verbose": False}
    assert lib.prefixes == ("acopf", "small")


def test_a_julia_argfun_is_passed_through(backend, recipe_core):
    argfun = _b.seval("n -> (n,)")
    compile_library("lib", recipe_core, 4, argfun=argfun)
    (_which, _args, kw), = backend
    assert kw["argfun"] is argfun


def test_placeholder_examples_are_unwrapped(backend, recipe_core):
    # An example given as a placeholder expression is data, and reaches the
    # backend as the node it stands for rather than as a Python wrapper.
    from examodels.recipe import Arg
    core, n = exa.recipe()
    assert isinstance(n + 1, Arg)                # what the caller wrote
    compile_library("lib", recipe_core, n + 1)
    (_which, args, _kw), = backend
    assert not isinstance(args[2][0], Arg)       # what the backend received


# ------------------------------------------------------------- the result ----
def test_the_result_is_usable_as_a_path():
    lib = CompiledLibrary("/out/libm.so", "/out", ["a", "b"])
    assert os.fspath(lib) == "/out/libm.so"
    assert str(lib) == "/out/libm.so"
    assert lib == "/out/libm.so"                 # compares equal to its path
    assert lib != "/elsewhere.so"
    assert (lib == 3) is False                   # not a path: no opinion, no crash
    assert os.path.basename(lib) == "libm.so"    # works in the stdlib path calls
    assert repr(lib) == "<CompiledLibrary '/out/libm.so' models=['a', 'b']>"
    assert hash(lib) == hash("/out/libm.so")


# ------------------------------------------------------------ installation ---
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


def test_the_compiler_is_the_renamed_package_at_its_new_home():
    # The backend moved org and the compiler subpackage was renamed; a stale
    # name here would resolve to nothing and only fail at install time.
    assert COMPILER["name"] == "ExaModelsCompiler"
    assert COMPILER["subdir"] == "ExaModelsCompiler"
    assert "madsuite-org" in COMPILER["url"]


def test_compiler_available_asks_without_importing(monkeypatch):
    asked = []
    monkeypatch.setattr(_b, "seval", lambda code: asked.append(code) or False,
                        raising=False)
    assert compiler_available() is False
    assert asked == ['Base.find_package("ExaModelsCompiler") !== nothing']


# ------------------------------------------------- examples as native data ---
def test_examples_are_converted_to_the_types_their_storage_will_have(backend, recipe_core):
    # The compiler emits storage of exactly the example's Julia type, so a
    # numpy array must not arrive as the PyArray juliacall makes of it. The
    # compiler refuses that by name, minutes into a build.
    import numpy as np
    compile_library("lib", recipe_core, 10, np.zeros(3), np.arange(3))
    (_which, args, _kw), = backend
    n, floats, ints = args[2]
    assert isinstance(n, int)
    assert _b.typestr(floats) == "Vector{Float64}"
    assert _b.typestr(ints) == "Vector{Int64}"


def test_a_table_example_crosses_as_named_rows(backend, recipe_core):
    from collections import namedtuple
    Row = namedtuple("Row", "bus load")
    compile_library("lib", recipe_core, [Row(0, 1.5), Row(1, 2.5)])
    (_which, args, _kw), = backend
    assert "NamedTuple" in _b.typestr(args[2][0])


@pytest.mark.parametrize("bad, match", [
    (True, "A bool is neither"),
    ([[1.0, 2.0], [3.0, 4.0]], "one-dimensional"),
    (["a", "b"], "cannot cross the C boundary"),
])
def test_examples_that_cannot_cross_are_refused(backend, recipe_core, bad, match):
    with pytest.raises(TypeError, match=match):
        compile_library("lib", recipe_core, bad)


def test_a_string_example_is_left_alone(backend, recipe_core):
    # A string is what argfun is called WITH; it never becomes storage.
    compile_library("lib", recipe_core, "case14.m", argfun="Pkg.dir")
    (_which, args, _kw), = backend
    assert args[2] == ["case14.m"]


# --------------------------------------------------- argfun named by string --
def test_an_argfun_can_be_named_rather_than_written(backend, recipe_core, monkeypatch):
    seen = {}

    def seval(code):
        seen["code"] = code
        return "JLFUNC"

    monkeypatch.setattr(_b, "seval", seval, raising=False)
    compile_library("lib", recipe_core, "case14.m", argfun="ExaPowerIO.parse_case")
    assert seen["code"] == "import ExaPowerIO; ExaPowerIO.parse_case"
    (_which, _args, kw), = backend
    assert kw["argfun"] == "JLFUNC"


@pytest.mark.parametrize("bad", ["parse_case", "Pkg.f(); rm(\"/\")", "", "A..b"])
def test_an_argfun_name_must_be_a_qualified_identifier(backend, recipe_core, bad):
    # It is evaluated as source, so it is checked before it is evaluated.
    with pytest.raises(ValueError, match="name a function in a package"):
        compile_library("lib", recipe_core, "x", argfun=bad)


def test_a_missing_argfun_package_says_what_is_missing(backend, recipe_core, monkeypatch):
    def boom(code):
        raise RuntimeError("ArgumentError: Package NotHere not found")
    monkeypatch.setattr(_b, "seval", boom, raising=False)
    with pytest.raises(_b.ModelError, match="is 'NotHere' installed"):
        compile_library("lib", recipe_core, "x", argfun="NotHere.load")
