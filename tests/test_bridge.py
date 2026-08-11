"""The bridge's own logic: version gating, error translation, laziness flags.

Everything here is pure Python -- the point of `_bridge` is that the rest of the
package never reasons about the backend runtime, so its small pieces of logic
(caret semantics, error mapping) deserve direct tests rather than being
exercised only as a side effect of model building.
"""
import pytest

from examodels import _bridge as _b
from examodels._bridge import ModelError, _satisfies, translate


def test_caret_semantics_match_on_leading_nonzero():
    # ^0.11 admits 0.11.x but not 0.12
    assert _satisfies((0, 11, 2), "0.11")
    assert not _satisfies((0, 12, 0), "0.11")
    # ^1.2 admits any 1.x >= 1.2, not 2.x
    assert _satisfies((1, 5, 0), "1.2")
    assert not _satisfies((2, 0, 0), "1.2")
    assert not _satisfies((1, 1, 9), "1.2")
    # all-zero bound: the last component is the lead
    assert _satisfies((0, 0, 3), "0.0.3")
    assert not _satisfies((0, 0, 4), "0.0.3")


def test_method_errors_translate_to_type_errors():
    err = translate(Exception("MethodError: no method matching sin(::Foo)"))
    assert isinstance(err, TypeError)
    assert "registered with ExaModels" in str(err)


@pytest.mark.parametrize("head, cls", [
    ("DimensionMismatch: arrays could not be broadcast", ValueError),
    ("BoundsError: attempt to access 3-element Vector", IndexError),
    ("ArgumentError: range must be non-empty", ValueError),
])
def test_backend_error_kinds_map_onto_pythons(head, cls):
    err = translate(Exception(head))
    assert isinstance(err, cls)
    # the marker is stripped, the message kept
    assert not str(err).startswith(head.split(":")[0])
    assert str(err) in head


def test_unrecognised_backend_errors_become_model_errors():
    err = translate(Exception("SomeNovelError: it went wrong\nlong julia backtrace"))
    assert isinstance(err, ModelError)
    # first line only: the foreign backtrace is the thing translate() exists to drop
    assert "backtrace" not in str(err)


def test_guard_reraises_python_exceptions_untranslated():
    class Boom(Exception):
        pass

    def raises():
        raise Boom("mine")

    with pytest.raises(Boom, match="mine"):
        _b.guard(raises)


def test_guard_translates_by_exception_type_name():
    # guard keys on the type NAME, since juliacall's class is not importable here
    JuliaError = type("JuliaError", (Exception,), {})

    def raises():
        raise JuliaError("ArgumentError: bad argument")

    with pytest.raises(ValueError, match="bad argument"):
        _b.guard(raises)


def test_started_reports_the_booted_backend():
    # By the time any test runs, conftest's solver probe has booted the runtime.
    assert _b.started() is True
