"""Device-memory interchange, tested without a device.

`from_cupy` and `_CudaView` are pointer bookkeeping around the CUDA array
interface -- pure Python up to the single backend call that wraps or publishes
a pointer. That call is stubbed here, so the logic (validation, shape and type
handling, ownership) runs for real on any machine; the true device round trip
is covered by `test_cupy.py` where hardware exists.
"""
import numpy as np
import pytest

import madsuite as exa
from madsuite import _bridge as _b
from madsuite.advanced import _CudaView, from_cupy


class FakeDeviceArray:
    """The shape of thing CuPy hands over: only the interface dict matters."""

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(arr)
        self.__cuda_array_interface__ = {
            "shape": self._arr.shape,
            "typestr": self._arr.dtype.str,
            "data": (self._arr.ctypes.data, False),
            "version": 3,
            "strides": None,
        }


def test_from_cupy_requires_the_interface():
    with pytest.raises(TypeError, match="__cuda_array_interface__"):
        from_cupy(np.zeros(3))


def test_from_cupy_requires_float64():
    with pytest.raises(TypeError, match="expected float64, got <f4"):
        from_cupy(FakeDeviceArray(np.zeros(3, dtype=np.float32)))


def test_from_cupy_hands_the_backend_pointer_and_flat_length(monkeypatch):
    captured = {}
    monkeypatch.setattr(_b, "guard",
                        lambda fn, p, n: captured.update(p=p, n=n) or "WRAPPED",
                        raising=False)
    arr = np.zeros((2, 3), dtype=np.float64)
    assert from_cupy(FakeDeviceArray(arr)) == "WRAPPED"
    assert captured["p"] == arr.ctypes.data
    assert captured["n"] == 6                        # flattened, not per-axis


def test_cuda_view_publishes_what_the_backend_reports(monkeypatch):
    owner = object()
    monkeypatch.setattr(_b, "guard", lambda fn, a: (12345, 7, "<f8"), raising=False)
    view = _CudaView(owner)
    assert view.__cuda_array_interface__ == {
        "shape": (7,), "typestr": "<f8", "data": (12345, False),
        "version": 3, "strides": None,
    }
    assert view._owner is owner            # keeps the backend array alive


def test_evaluation_accepts_a_device_vector(monkeypatch):
    # A model evaluated at a "device" vector: `_x` takes the interface route into
    # `from_cupy`; the stub hands the host values back, so the objective is real.
    core = exa.Core()
    x = core.add_var(2)
    core.add_obj(lambda i: (x[i] - 1) ** 2, over=range(2))
    model = core.build()

    fake = FakeDeviceArray(np.array([3.0, 3.0]))
    monkeypatch.setattr(_b, "wrap_device_ptr",
                        lambda p, n: fake._arr, raising=False)
    assert model.objective(fake) == pytest.approx(model.objective([3.0, 3.0])) == 8.0


def test_setters_accept_a_device_vector(monkeypatch):
    core = exa.Core()
    x = core.add_var(2)
    model = core.build()

    fake = FakeDeviceArray(np.array([4.0, 5.0]))
    monkeypatch.setattr(_b, "wrap_device_ptr",
                        lambda p, n: fake._arr, raising=False)
    assert model.set_start(x, fake) is model
    np.testing.assert_allclose(model.get_start(x), [4.0, 5.0])


def test_as_cupy_wraps_the_view_for_cupy(monkeypatch):
    # The device half of `as_cupy`, with cupy stood in for: the backend array
    # reports as device memory, and what reaches cupy is the published view.
    import sys
    import types

    seen = {}
    monkeypatch.setitem(sys.modules, "cupy",
                        types.SimpleNamespace(asarray=lambda v: seen.update(v=v) or "CUPY"))
    monkeypatch.setattr(_b, "guard", lambda fn, a: (True if fn is _b.is_device
                                                    else (777, 3, "<f8")), raising=False)
    assert exa.as_cupy("JLARRAY") == "CUPY"
    assert isinstance(seen["v"], _CudaView)
    assert seen["v"].__cuda_array_interface__["data"] == (777, False)
    assert seen["v"]._owner == "JLARRAY"
