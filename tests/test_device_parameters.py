"""`set_parameters` on a device model — needs hardware.

Every other setter places its values where the model's arrays live
(`_b.upload`); `set_parameters` passed the host array straight through, which
on a device model drops into the scalar path GPUArrays forbids and raised
`ModelError: Scalar indexing is disallowed`. The scalar-indexing guard only
exists on a real device array, so this cannot be pinned by the stubbed-device
tests in `test_device_interop.py`.
"""
import numpy as np
import pytest

import madsuite as exa

pytestmark = pytest.mark.skipif(
    not __import__("shutil").which("nvidia-smi"), reason="no GPU on this machine")


def test_set_parameters_takes_effect_on_a_device_model():
    core = exa.Core(backend="cuda")
    x = core.add_var(4, start=0.0)
    p = core.add_par([1.0, 2.0, 3.0, 4.0])
    core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(4))
    m = exa.Model(core)

    assert m.objective(np.zeros(4)) == pytest.approx(1.0 + 4.0 + 9.0 + 16.0)

    m.set_parameters(p, [5.0, 6.0, 7.0, 8.0])
    assert m.objective(np.zeros(4)) == pytest.approx(25.0 + 36.0 + 49.0 + 64.0)
