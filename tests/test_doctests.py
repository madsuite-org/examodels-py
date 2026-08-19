"""The docstrings' own examples, verified.

Modules are listed explicitly rather than swept with --doctest-modules, so
the population is named; the per-module count assertion keeps this from
passing vacuously if collection ever comes back empty (a doctest suite that
finds nothing is green for the wrong reason).
"""
import doctest
from importlib import import_module

import pytest

# Resolved through importlib: `madsuite.recipe` and `madsuite.solve` as
# ATTRIBUTES are the exported functions of those names, not the submodules.
MODULES = [import_module(f"madsuite.{n}")
           for n in ("core", "recipe", "solve", "_record")]


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples(mod):
    result = doctest.testmod(mod, verbose=False)
    assert result.attempted > 0, f"no doctest examples collected from {mod.__name__}"
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {mod.__name__}"
