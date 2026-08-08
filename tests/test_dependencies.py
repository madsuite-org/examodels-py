"""Dependency hygiene: the coupling to the backend must be explicit and checked."""
import json
import pathlib
import sys

import examodels
from examodels import _bridge as _b

ROOT = pathlib.Path(examodels.__file__).parent


def test_declared_backend_range_is_the_one_installed():
    """The version the package declares must be the version it is actually running."""
    installed = _b.version
    declared = _b._compat()
    assert installed.startswith(declared + "."), \
        f"declared {declared}, running {installed} -- the compat bound is not being enforced"


def test_every_backend_symbol_we_use_still_exists():
    """Several of these are not part of the backend's public API.

    Listing them here means a backend upgrade that removes one fails this suite,
    rather than failing at a user's first model build.
    """
    missing = [n for n in _b.REQUIRED
               if not bool(_b.seval(f'isdefined(ExaModels, :var"{n}")'))]
    assert not missing, f"backend symbols we depend on have gone: {missing}"


def test_startup_check_is_not_vacuous():
    """The startup check must actually reject a backend that is missing a symbol."""
    import pytest
    saved = _b.REQUIRED
    try:
        _b.REQUIRED = saved + ("a_symbol_that_does_not_exist",)
        with pytest.raises(RuntimeError, match="a_symbol_that_does_not_exist"):
            _b._check(_b.jl)
    finally:
        _b.REQUIRED = saved


def test_python_dependencies_are_only_what_we_declare():
    """Required dependencies stay at two; anything else must be an optional extra."""
    from _toml import load as _load_toml
    declared = _load_toml(ROOT.parents[1] / "pyproject.toml")

    def names_of(entries):
        return {d.split(">=")[0].split("[")[0].split("-cuda")[0].strip()
                for d in entries}

    required = names_of(declared["project"]["dependencies"])
    assert required == {"juliacall", "numpy"}, required
    optional = set().union(*(names_of(v) for v in
                             declared["project"]["optional-dependencies"].values()))

    imported = set()
    for f in ROOT.glob("*.py"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line.startswith(("import ", "from ")) and not line.split()[1].startswith("."):
                imported.add(line.split()[1].split(".")[0])
    # stdlib_module_names is 3.10+, and this package claims 3.9
    stdlib = set(getattr(sys, "stdlib_module_names", ())) or set(sys.builtin_module_names) | {
        "os", "sys", "re", "io", "math", "cmath", "json", "types", "typing", "dis",
        "pathlib", "runpy", "subprocess", "textwrap", "shutil", "time", "importlib",
        "collections", "dataclasses", "tomllib", "warnings", "itertools", "functools",
    }
    allowed = required | optional | {"juliapkg"}
    third_party = imported - stdlib - {"examodels"}
    assert third_party <= allowed, \
        f"undeclared third-party imports: {sorted(third_party - allowed)}"


def test_backend_requirements_are_declared_in_one_place():
    """juliapkg.json is the single source of truth for the backend requirement."""
    spec = json.loads((ROOT / "juliapkg.json").read_text())
    assert set(spec["packages"]) >= {"ExaModels", "NLPModels"}
    for name, entry in spec["packages"].items():
        assert entry.get("version"), f"{name} has no version bound"


def test_only_the_bridge_touches_juliacall():
    """Exactly one module may import the interop layer."""
    offenders = [f.name for f in ROOT.glob("*.py")
                 if "juliacall" in f.read_text() and f.name != "_bridge.py"]
    assert not offenders, f"juliacall imported outside the bridge: {offenders}"


def test_the_backend_source_in_use_is_reported():
    """A dev checkout silently overriding the declared version must be visible."""
    path = str(_b.seval("pathof(ExaModels)"))
    print(f"\nbackend in use: {path} (version {_b.version})")
    assert path.endswith("ExaModels.jl")


def test_user_startup_file_is_not_run_in_the_backend():
    """Our behaviour must not depend on the user's personal Julia dotfiles."""
    import os
    _b._boot()
    assert os.environ.get("PYTHON_JULIACALL_STARTUPFILE") == "no"
