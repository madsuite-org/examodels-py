"""Shared fixtures.

A solver is an optional dependency — the package models happily without one — so
tests that solve skip when none is installed rather than failing.
"""
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_model_cache(tmp_path_factory):
    """Tests must never read or write the user's real model cache: a
    `Core(cache=True)` built anywhere in the suite would otherwise compile
    into (or hit from) `~/.cache/examodels` — found there as residue."""
    if "EXAMODELS_CACHE" in os.environ:
        yield
        return
    os.environ["EXAMODELS_CACHE"] = str(tmp_path_factory.mktemp("model-cache"))
    yield
    del os.environ["EXAMODELS_CACHE"]


def _installed(name):
    from examodels import _bridge as _b
    from examodels.solve import SOLVERS
    pkg = SOLVERS[name][0]
    try:
        _b.seval(f"using {pkg}")
        return True
    except Exception:                                        # noqa: BLE001
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_solver: requires a solver backend")


def requires(name):
    """Skip unless a particular solver backend is installed.

    The suite-wide skip below only fires when there is no solver at all; a test
    naming one specifically needs its own guard, or it fails in an environment
    that happens to have a different one.
    """
    return pytest.mark.skipif(not _installed(name),
                              reason=f"the {name!r} solver is not installed")


def pytest_collection_modifyitems(config, items):
    if _installed("ipopt") or _installed("madnlp"):
        return
    skip = pytest.mark.skip(reason="no solver backend installed")
    for item in items:
        item.add_marker(skip)
