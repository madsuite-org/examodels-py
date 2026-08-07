"""Shared fixtures.

A solver is an optional dependency — the package models happily without one — so
tests that solve skip when none is installed rather than failing.
"""
import pytest


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


@pytest.fixture(scope="session")
def has_ipopt():
    return _installed("ipopt")


def pytest_collection_modifyitems(config, items):
    if _installed("ipopt") or _installed("madnlp"):
        return
    skip = pytest.mark.skip(reason="no solver backend installed")
    for item in items:
        item.add_marker(skip)
