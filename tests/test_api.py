"""The interface must not expose the fact that a Julia runtime is involved."""
import numpy as np
import pytest

import examodels as exa


def test_import_does_not_start_the_backend():
    """`import examodels` must be instant: no runtime boot until a model is built."""
    import subprocess, sys, textwrap
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import time; t = time.perf_counter()
            import examodels
            from examodels import _bridge
            print(f"{_bridge.started()} {time.perf_counter() - t:.3f}")
        """)], capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "False", "importing the package started the backend"
    assert float(out[1]) < 2.0, f"import took {out[1]}s"


def test_sizes_are_python_ints():
    core = exa.Core()
    x = core.add_var(5)
    core.add_obj(lambda i: x[i]**2, over=range(5))
    p = exa.Model(core)
    for v in (p.nvar, p.ncon, p.nnzj, p.nnzh):
        assert type(v) is int


def test_arrays_are_numpy():
    core = exa.Core()
    x = core.add_var(4, start=1.0)
    core.add_obj(lambda i: x[i]**2, over=range(4))
    p = exa.Model(core)
    assert isinstance(p.x0, np.ndarray) and p.x0.dtype == np.float64
    assert isinstance(p.gradient(np.ones(4)), np.ndarray)
    assert p.objective(np.ones(4)) == pytest.approx(4.0)


def test_unsupported_operation_raises_python_typeerror():
    core = exa.Core()
    x = core.add_var(3)
    with pytest.raises(TypeError):
        exa.trace(lambda i: __import__("math").sin(x[i]))   # math.sin, not exa.sin


def test_data_given_as_a_function_is_rejected_with_a_useful_message():
    core = exa.Core()
    with pytest.raises(TypeError, match="data"):
        core.add_var(3, start=lambda i: float(i))


def test_no_julia_identifiers_in_public_surface():
    leaked = [n for n in dir(exa) if "julia" in n.lower() or n.endswith("_jl")]
    assert not leaked, f"Julia-flavoured names in the public API: {leaked}"
