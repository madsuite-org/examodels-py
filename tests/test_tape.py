"""Tape parity: a recorded-and-replayed model must match the Core-built one."""
import examodels as exa

N_TEMPLATE = 4
N_REPLAY = 30  # deliberately different from the template


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


def test_replayed_tape_matches_core():
    sol_tape = exa.Model(_lv_tape().replay(N=N_REPLAY)).solve(solver="madnlp")
    sol_core = exa.Model(_lv_core(N_REPLAY)).solve(solver="madnlp")
    assert sol_tape.success and sol_core.success
    assert sol_tape.status == sol_core.status
    assert abs(sol_tape.objective - sol_core.objective) < 1e-10


def test_one_tape_many_sizes():
    tape = _lv_tape()
    objs = [exa.Model(tape.replay(N=n)).solve(solver="madnlp").objective
            for n in (20, 60)]
    # LuksanVlcek's optimum is size-stable; both must be finite and close.
    assert all(abs(o - 6.2324586324) < 1e-6 for o in objs)


def test_static_range_also_works():
    tape = exa.Tape(N=4)
    x = tape.add_var(tape.data.N, start=0.0)
    tape.add_obj(lambda i: (x[i] - 1)**2, over=range(0, 4))  # static set
    core = tape.replay(N=4)
    assert core is not None


def test_structure_cannot_depend_on_data():
    tape = exa.Tape(N=4)
    try:
        bool(tape.data.N > 5)
    except TypeError:
        pass  # comparisons on data handles are refused (Python-side, by Node)
    else:
        raise AssertionError("comparing a data value should not be allowed")
