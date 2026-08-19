"""Accelerator backends: resolved by name, loaded only when asked for."""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import madsuite as exa


def test_gpu_backend_is_not_loaded_for_a_cpu_model():
    """Building a CPU model must not pull in a GPU runtime."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import madsuite as exa
            from madsuite import _bridge as b
            core = exa.Core()
            x = core.add_var(4)
            core.add_obj(lambda i: x[i]**2, over=range(4))
            exa.Model(core)
            print(bool(b.seval("isdefined(Main, :CUDA)")))
        """)], capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith("False"), out.stdout


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        exa.Core(backend="quantum")


def test_backends_are_listed():
    assert {"cuda", "cpu", "serial"} <= set(exa.backends())


def _luksan(n, backend):
    core = exa.Core(backend=backend)
    x = core.add_var(n, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
    core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2, over=range(1, n))
    core.add_con(lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
                   + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
                   + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
                   over=range(0, n - 2), lcon=0.0, ucon=0.0)
    return exa.Model(core), x


def _gpu_hardware_present():
    """Is there a GPU on this machine at all?"""
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _cuda_backend_works():
    try:
        exa.Core(backend="cuda")
        return True, ""
    except Exception as e:                                  # noqa: BLE001
        return False, str(e).splitlines()[0]


_HW = _gpu_hardware_present()
_OK, _WHY = _cuda_backend_works() if _HW else (False, "")

# Skipping only because there is no hardware. If a GPU IS present and the backend
# still will not load, that is a regression and must fail -- a skip here would be a
# silent pass exactly where the coverage matters. (This is how the system-image
# regression was found: it turned these tests from passing into skipped.)
if _HW and not _OK:
    def test_cuda_backend_is_broken_despite_gpu_hardware():
        pytest.fail(f"a GPU is present but the cuda backend will not load: {_WHY}")

cuda_only = pytest.mark.skipif(not _HW, reason="no GPU hardware on this machine")


@cuda_only
def test_model_arrays_live_on_the_device():
    from madsuite import _bridge as b
    model, _ = _luksan(100, "cuda")
    assert "CuArray" in str(b.seval("m -> string(typeof(m.meta.x0))")(model._jl))


@cuda_only
def test_gpu_and_cpu_reach_the_same_solution():
    n = 5000
    gpu, xg = _luksan(n, "cuda")
    cpu, xc = _luksan(n, None)
    sg, sc = gpu.solve(solver="madnlp"), cpu.solve(solver="madnlp")
    assert sg.success and sc.success, (sg.status, sc.status)
    # different linear solvers, so they stop at slightly different points
    assert sg.objective == pytest.approx(sc.objective, rel=1e-3)
    assert abs(sg[xg] - sc[xc]).max() < 1e-3


@cuda_only
def test_a_device_model_picks_a_device_solver():
    """Ipopt cannot take device arrays, so a device model must not be sent to it."""
    core = exa.Core(backend="cuda")
    x = exa.add_var(core, 20, start=1.0)
    exa.add_obj(core, lambda i: (x[i] - 2.0)**2, over=range(20))
    sol = exa.Model(core).solve()                    # no solver= given
    assert sol.success and sol.status == "SOLVE_SUCCEEDED"


@cuda_only
def test_setters_work_on_a_device_model():
    """Values must be placed on the device; a host array hits a scalar path."""
    core = exa.Core(backend="cuda")
    th = exa.add_par(core, [2.0])
    x = exa.add_var(core, 10, start=1.0)
    exa.add_obj(core, lambda i: (x[i] - th[0])**2, over=range(10))
    model = exa.Model(core)
    np.testing.assert_allclose(model.solve()[x], 2.0, atol=1e-6)
    model.set_value(th, [3.0])
    np.testing.assert_allclose(model.solve()[x], 3.0, atol=1e-6)
    model.set_start(x, np.full(10, 9.0))
    np.testing.assert_allclose(model.get_start(x), 9.0)


@cuda_only
def test_wrappers_work_on_a_device_model():
    core = exa.Core(backend="cuda")
    x = exa.add_var(core, 20, start=1.0)
    exa.add_obj(core, lambda i: x[i]**2, over=range(20))
    exa.add_con(core, lambda i: x[i] + x[i + 1], over=range(19), lcon=1.0, ucon=3.0)
    model = exa.Model(core)
    for wrap in (exa.WrapperNLPModel, exa.TimedNLPModel, exa.CompressedNLPModel):
        w = wrap(model)
        assert (w.nvar, w.ncon) == (20, 19)


@cuda_only
def test_compressed_wrapper_needs_constraints_upstream():
    """Known backend limitation, recorded so a fix upstream shows up here.

    `CompressedNLPModel` divides by the constraint count, so a model with none
    raises. Nothing to do about it from this side.
    """
    core = exa.Core(backend="cuda")
    x = exa.add_var(core, 5, start=1.0)
    exa.add_obj(core, lambda i: x[i]**2, over=range(5))
    with pytest.raises(exa.ModelError, match="DivideError"):
        exa.CompressedNLPModel(exa.Model(core))
