"""Recording cores — `Core(cache=...)`.

A recording core captures the model in pure Python: no Julia during
construction, a stable (fingerprint, data-digest) pair, and a replay through
the ordinary eager path that reproduces the eager model exactly.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import examodels as exa

N = 10

#: One model exercising every recorded form: variables with data, parameters,
#: math functions, Constant, generator syntax, an augmented constraint block,
#: and a dims-only block. Shared with the subprocess test, so the in-process
#: and out-of-process fingerprints come from the identical construction.
MODEL_SRC = textwrap.dedent("""
    def build(exa, cache):
        core = exa.Core(cache=True) if cache else exa.Core()
        x = core.add_var(6, start=0.5, lvar=0.0, uvar=2.0)
        p = core.add_par([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        core.add_obj(exa.exp(x[i]) - p[i] * x[i] + exa.Constant(2.0) for i in range(6))
        c = core.add_con((x[i] + x[i + 1] for i in range(5)), lcon=0.0, ucon=3.0)
        core.add_con(c, lambda i: (i, exa.sin(x[i])), range(5))
        core.add_con(2, lcon=-1.0, ucon=1.0)
        return core
""")
_ns = {}
exec(MODEL_SRC, _ns)
_build = _ns["build"]


def _rosen(core, n=N, coeff=100):
    x = core.add_var(n, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
    core.add_obj(lambda i: coeff * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,
                 over=range(1, n))
    return core


# -- julia-free construction --------------------------------------------------

def test_recording_never_loads_julia():
    """The point of the whole layer: record + fingerprint in a fresh process
    without juliacall ever entering sys.modules."""
    code = MODEL_SRC + textwrap.dedent("""
        import sys
        import examodels as exa
        core = build(exa, cache=True)
        fp, dd = core.fingerprint()
        assert len(fp) == 64 and len(dd) == 64
        assert "juliacall" not in sys.modules, "recording booted Julia"
        print(fp, dd)
    """)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    fp, dd = out.stdout.split()
    # the same construction in THIS process hashes identically: nothing
    # process-specific (id(), hash(), dict order) leaks into the digest
    assert (fp, dd) == _build(exa, cache=True).fingerprint()


# -- replay parity ------------------------------------------------------------

def test_replay_matches_eager_numerically():
    m1 = exa.Model(_build(exa, cache=False))
    m2 = exa.Model(_build(exa, cache=True))          # replays here
    assert (m1.nvar, m1.ncon) == (m2.nvar, m2.ncon)
    np.testing.assert_array_equal(m1.x0, m2.x0)
    np.testing.assert_array_equal(m1.lvar, m2.lvar)
    np.testing.assert_array_equal(m1.uvar, m2.uvar)
    np.testing.assert_array_equal(m1.lcon, m2.lcon)
    np.testing.assert_array_equal(m1.ucon, m2.ucon)
    assert m1.objective(m1.x0) == m2.objective(m2.x0)
    np.testing.assert_array_equal(m1.gradient(m1.x0), m2.gradient(m2.x0))
    np.testing.assert_array_equal(m1.constraints(m1.x0), m2.constraints(m2.x0))


def test_replay_matches_eager_on_rosenbrock():
    m1 = exa.Model(_rosen(exa.Core()))
    m2 = exa.Model(_rosen(exa.Core(cache=True)))
    assert m1.objective(m1.x0) == m2.objective(m2.x0)
    np.testing.assert_array_equal(m1.gradient(m1.x0), m2.gradient(m2.x0))


def test_replayed_tree_lowers_to_the_identical_backend_type():
    """Structure, not just numbers: the replayed tree must lower to the very
    same backend expression TYPE as tracing the lambda directly — including
    `**2` via literal_pow and reflected operators.

    `same_structure` compares the types themselves; `julia_type` strings are
    abbreviated and would pass for almost any pair.
    """
    from examodels._record import _render
    from examodels.testing import same_structure
    eager = exa.Core()
    x = eager.add_var(N)
    fns = [
        lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,
        lambda i: exa.sin(x[i]) / x[i] + 1.0 / x[i],
        lambda i: -x[i] + (1.0 - x[i]) * exa.Constant(3),
        lambda i: 2 ** x[i] + x[i] ** 0.5,
    ]
    rec = exa.Core(cache=True)
    rx = rec.add_var(N)
    rfns = [
        lambda i: 100 * (rx[i - 1] ** 2 - rx[i]) ** 2 + (rx[i - 1] - 1) ** 2,
        lambda i: exa.sin(rx[i]) / rx[i] + 1.0 / rx[i],
        lambda i: -rx[i] + (1.0 - rx[i]) * exa.Constant(3),
        lambda i: 2 ** rx[i] + rx[i] ** 0.5,
    ]
    for k, (f, rf) in enumerate(zip(fns, rfns)):
        rec.add_obj(rf, over=range(1, N))
        tree = rec._records[-1]["tree"]
        got = exa.trace(_render(tree, {("var", 0): x}))
        want = exa.trace(f)
        assert same_structure(got, want), f"expression {k} lowers differently"


def test_literal_values_are_data_not_structure():
    """A changed coefficient must not change the backend expression type —
    that is what licenses keeping literal VALUES in the data digest while the
    fingerprint keeps only their numeric type."""
    from examodels.testing import same_structure
    core = exa.Core()
    x = core.add_var(N)
    a = exa.trace(lambda i: 100 * (x[i] ** 2 - 1) ** 2)
    b = exa.trace(lambda i: 200 * (x[i] ** 2 - 1) ** 2)
    assert same_structure(a, b)
    # ... while the literal's numeric TYPE is structural, matching the fingerprint
    c = exa.trace(lambda i: 100.0 * (x[i] ** 2 - 1) ** 2)
    assert not same_structure(a, c)


# -- fingerprint semantics ----------------------------------------------------

def test_fingerprint_is_stable():
    assert (_rosen(exa.Core(cache=True)).fingerprint()
            == _rosen(exa.Core(cache=True)).fingerprint())
    assert (_build(exa, cache=True).fingerprint()
            == _build(exa, cache=True).fingerprint())


def _fps(**kw):
    return _rosen(exa.Core(cache=True), **kw).fingerprint()


def test_fingerprint_separates_structure_from_data():
    fp0, dd0 = _fps()
    # a changed literal VALUE: same code, different baked data
    fp1, dd1 = _fps(coeff=200)
    assert fp1 == fp0 and dd1 != dd0
    # a changed exponent (2 -> 3) lives in the type: different code
    core = exa.Core(cache=True)
    x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])
    core.add_obj(lambda i: 100 * (x[i - 1] ** 3 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,
                 over=range(1, N))
    fp2, dd2 = core.fingerprint()
    assert fp2 != fp0
    # int -> float literal changes the operand type: different code
    fp3, dd3 = _fps(coeff=100.0)
    assert fp3 != fp0
    # a changed start is data only
    core = exa.Core(cache=True)
    x = core.add_var(N, start=0.0)
    core.add_obj(lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,
                 over=range(1, N))
    fp4, dd4 = core.fingerprint()
    assert fp4 == fp0 and dd4 != dd0
    # a resized index set is data only (the backend type carries no lengths)
    fp5, dd5 = _fps(n=N - 1)
    assert fp5 == fp0 and dd5 != dd0
    # a block name is structural: the compiled library bakes its layout
    core = exa.Core(cache=True)
    x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)],
                     name="x")
    core.add_obj(lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,
                 over=range(1, N))
    fp6, dd6 = core.fingerprint()
    assert fp6 != fp0


def test_parameter_values_are_in_neither_digest():
    """Parameter values stay live through the ABI setter on a loaded library,
    so changing them must not miss the cache."""
    def with_values(v):
        core = exa.Core(cache=True)
        x = core.add_var(4)
        p = core.add_par(v)
        core.add_obj(lambda i: p[i] * x[i] ** 2, over=range(4))
        return core.fingerprint()
    assert with_values([1.0, 2.0, 3.0, 4.0]) == with_values([5.0, 6.0, 7.0, 8.0])
    # but a resized parameter block is a different model
    core = exa.Core(cache=True)
    x = core.add_var(4)
    p = core.add_par([1.0, 2.0, 3.0])
    core.add_obj(lambda i: p[i] * x[i] ** 2, over=range(3))
    assert core.fingerprint()[0] != with_values([1.0, 2.0, 3.0, 4.0])[0]


# -- refusals -----------------------------------------------------------------

def test_unsupported_features_are_refused_up_front():
    # recipes (nargs=) record since the recipe-cache milestone; the refusals
    # that remain are the ones a compiled CPU library genuinely cannot carry
    with pytest.raises(ValueError, match="cannot be cached"):
        exa.Core(cache=True, backend="cuda")
    with pytest.raises(NotImplementedError, match="tags"):
        core = exa.Core(cache=True)
        core.add_var(3, tag=object())


def test_mixing_cores_raises_a_clear_error():
    eager = exa.Core()
    ex = eager.add_var(N)
    rec = exa.Core(cache=True)
    rx = rec.add_var(N)
    with pytest.raises(TypeError, match="cache"):
        rec.add_obj(lambda i: ex[i] ** 2, over=range(N))
    with pytest.raises(TypeError, match="cache"):
        eager.add_obj(lambda i: rx[i] ** 2, over=range(N))


def test_misspelled_math_function_fails_at_replay():
    core = exa.Core(cache=True)
    x = core.add_var(3)
    core.add_obj(lambda i: exa.sine(x[i]), over=range(3))   # recorded happily
    with pytest.raises((TypeError, AttributeError), match="sine"):
        exa.Model(core)
