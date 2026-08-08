"""Tape parity: a recorded-and-instantiated model must match the Core-built one."""
import examodels as exa

N_TEMPLATE = 4
N_INSTANTIATE = 30  # deliberately different from the template


def _lv_core(n):
    core = exa.Core()
    x = exa.add_var(core, n, start=-0.5)
    core.add_con(lambda i: 3*x[i+1]**3 + 2*x[i+2] - 5
                 + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
                 + 4*x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
                 over=range(0, n - 2))
    core.add_obj(lambda i: 100*(x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
                 over=range(1, n))
    return core


def _lv_tape():
    tape = exa.Tape(N=N_TEMPLATE)
    x = tape.add_var(tape.data.N, start=-0.5)
    tape.add_con(lambda i: 3*x[i+1]**3 + 2*x[i+2] - 5
                 + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
                 + 4*x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
                 over=exa.srange(0, tape.data.N - 2))
    tape.add_obj(lambda i: 100*(x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
                 over=exa.srange(1, tape.data.N))
    return tape


def test_instantiated_tape_matches_core():
    sol_tape = exa.Model(_lv_tape().instantiate(N=N_INSTANTIATE)).solve(solver="ipopt")
    sol_core = exa.Model(_lv_core(N_INSTANTIATE)).solve(solver="ipopt")
    assert sol_tape.success and sol_core.success
    assert sol_tape.status == sol_core.status
    assert abs(sol_tape.objective - sol_core.objective) < 1e-10


def test_one_tape_many_sizes():
    tape = _lv_tape()
    objs = [exa.Model(tape.instantiate(N=n)).solve(solver="ipopt").objective
            for n in (20, 60)]
    # LuksanVlcek's optimum is size-stable; both must be finite and close.
    assert all(abs(o - 6.2324586324) < 1e-6 for o in objs)


def test_static_range_also_works():
    tape = exa.Tape(N=4)
    x = tape.add_var(tape.data.N, start=0.0)
    tape.add_obj(lambda i: (x[i] - 1)**2, over=range(0, 4))  # static set
    sol = exa.Model(tape.instantiate(N=4)).solve(solver="ipopt")
    assert sol.success
    assert abs(sol.objective) < 1e-8          # optimum: every x[i] = 1
    assert abs(sol[x][0] - 1.0) < 1e-6        # read-back through the tape handle


def test_structure_cannot_depend_on_data_python_layer():
    tape = exa.Tape(N=4)
    try:
        bool(tape.data.N > 5)
    except TypeError:
        pass  # comparisons on data handles are refused (Python-side, by Node)
    else:
        raise AssertionError("comparing a data value should not be allowed")


def test_structure_cannot_depend_on_data_backend_guard():
    """The recorder's own guard, reached through the bridge's error translation."""
    import pytest

    from examodels import _bridge as _b
    tape = exa.Tape(N=4)
    compare = _b.seval("d -> d.N > 5")
    with pytest.raises(exa.ModelError, match="[Ss]tructure"):
        _b.guard(compare, tape.data._tracer)


def test_instantiate_requires_exact_template_fields():
    tape = exa.Tape(N=4)
    x = tape.add_var(tape.data.N, start=0.0)
    tape.add_obj(lambda i: (x[i] - 1)**2, over=range(0, 4))
    import pytest
    with pytest.raises(TypeError, match="template"):
        tape.instantiate(M=999)     # wrong key must not silently build N=4
    with pytest.raises(TypeError, match="schema has fields"):
        tape.instantiate()          # missing key must not default to the placeholder


def test_bare_value_instantiates_a_single_field_schema():
    tape = _lv_tape()
    a = exa.Model(tape.instantiate(20)).solve(solver="ipopt")
    b = exa.Model(tape.instantiate(N=20)).solve(solver="ipopt")
    assert abs(a.objective - b.objective) < 1e-10


def test_positional_and_keyword_args_are_exclusive():
    import pytest
    with pytest.raises(TypeError):
        _lv_tape().instantiate(20, N=20)
    with pytest.raises(TypeError):
        _lv_tape().instantiate()          # the schema has a field
