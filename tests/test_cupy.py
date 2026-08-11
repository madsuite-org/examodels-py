"""Sharing device memory with CuPy, without going through the host."""
import numpy as np
import pytest

import examodels as exa

cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    not __import__("shutil").which("nvidia-smi"), reason="no GPU on this machine")


def device_model(n=8):
    core = exa.Core(backend="cuda")
    x = exa.add_var(core, n, start=1.0)
    exa.add_obj(core, lambda i: (x[i] - 3.0)**2, over=range(n))
    return exa.Model(core), x


def test_a_cupy_view_shares_the_backend_pointer():
    """Not merely equal values — the same address."""
    from examodels import _bridge as _b
    model, x = device_model()
    arr = _b.seval("m -> m.meta.x0")(model._jl)
    view = exa.as_cupy(arr)
    assert isinstance(view, cupy.ndarray) and view.shape == (8,)
    assert view.data.ptr == int(_b.seval("a -> UInt64(UInt(pointer(a)))")(arr)), \
        "the view copied instead of aliasing"


def test_writing_through_the_view_changes_the_model():
    from examodels import _bridge as _b
    model, x = device_model()
    exa.as_cupy(_b.seval("m -> m.meta.x0")(model._jl))[:] = 7.0
    np.testing.assert_allclose(model.get_start(x), 7.0)


def test_a_cupy_array_can_be_given_to_a_setter():
    model, x = device_model()
    model.set_start(x, cupy.full(8, 2.5, dtype=cupy.float64))
    np.testing.assert_allclose(model.get_start(x), 2.5)


def test_a_cupy_array_can_be_evaluated_at():
    model, x = device_model()
    point = cupy.ones(8, dtype=cupy.float64)
    assert model.objective(point) == pytest.approx(8 * 4.0)
    np.testing.assert_allclose(model.gradient(point), -4.0)
    np.testing.assert_allclose(model.constraints(point), np.zeros(0))


def test_evaluating_at_a_cupy_point_matches_the_host_path():
    model, x = device_model()
    host = np.linspace(0.0, 1.0, 8)
    assert model.objective(cupy.asarray(host)) == pytest.approx(model.objective(host))
    np.testing.assert_allclose(model.gradient(cupy.asarray(host)),
                               model.gradient(host))


def test_a_host_array_is_refused_by_as_cupy():
    core = exa.Core()
    x = exa.add_var(core, 3, start=1.0)
    exa.add_obj(core, lambda i: x[i]**2, over=range(3))
    from examodels import _bridge as _b
    arr = _b.seval("m -> m.meta.x0")(exa.Model(core)._jl)
    with pytest.raises(TypeError, match="host memory"):
        exa.as_cupy(arr)


def test_from_cupy_refuses_the_wrong_dtype():
    with pytest.raises(TypeError, match="float64"):
        exa.from_cupy(cupy.ones(4, dtype=cupy.float32))
    with pytest.raises(TypeError, match="CuPy array"):
        exa.from_cupy(np.ones(4))


def test_cupy_and_the_backend_coexist_across_a_solve():
    """A CuPy allocation must survive the backend using the device."""
    keep = cupy.arange(1000, dtype=cupy.float64)
    model, x = device_model()
    assert model.solve().success
    assert float(keep.sum()) == pytest.approx(999 * 1000 / 2)
    np.testing.assert_allclose(cupy.asnumpy(cupy.sin(keep[:4])),
                               np.sin(np.arange(4.0)))
