"""Elementwise math, generated from ExaModels' own registered operator list.

Nothing here is transcribed: the names come from `ExaModels._UNIVARIATES` and
`._BIVARIATES` at import time, so an operator added upstream needs no change here.
"""
from . import _bridge as _b
from .node import Node

UNIVARIATES = [str(s) for s in _b.seval("[string(t[1]) for t in ExaModels._UNIVARIATES]")]
BIVARIATES = [str(s) for s in _b.seval("[string(t[1]) for t in ExaModels._BIVARIATES]")]

_OPERATORS = {"+", "-", "*", "/", "^"}   # exposed as Python operators on Node instead


def _univariate(name):
    jf = _b.seval(name)

    def f(a):
        return Node(jf(_b.unwrap(a)))

    f.__name__ = name
    f.__doc__ = f"ExaModels-registered univariate `{name}`."
    return f


def _bivariate(name):
    jf = _b.seval(name)

    def f(a, b):
        return Node(jf(_b.unwrap(a), _b.unwrap(b)))

    f.__name__ = name
    f.__doc__ = f"ExaModels-registered bivariate `{name}`."
    return f


__all__ = []
for _n in UNIVARIATES:
    if _n not in _OPERATORS and _n.isidentifier():
        globals()[_n] = _univariate(_n)
        __all__.append(_n)
for _n in BIVARIATES:
    if _n not in _OPERATORS and _n.isidentifier() and _n not in __all__:
        globals()[_n] = _bivariate(_n)
        __all__.append(_n)
