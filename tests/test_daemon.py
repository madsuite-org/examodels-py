"""The warm session end to end, on CPU: a real daemon subprocess serves
recorded models, answers match the in-process path exactly, and every
degradation falls back rather than breaking the run.

The module-scoped daemon boots Julia once, on its first solve — later tests
here are the warm case the feature exists for.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
from conftest import requires

import examodels as exa
from examodels import _wire
from examodels._warm import DaemonModel, DaemonSolution, DispatchCore
from examodels.core import Core as EagerCore


def _spawn(path, *extra):
    env = dict(os.environ)
    env.pop("EXAMODELS_DAEMON", None)      # serve() sets its own guard
    proc = subprocess.Popen(
        [sys.executable, "-m", "examodels.daemon", "serve", "--socket", path,
         *extra],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        if os.path.exists(path):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"daemon died at start:\n{proc.stdout.read()}")
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("daemon socket never appeared")


def _sockdir():
    """A SHORT directory for socket files: an AF_UNIX path is limited to
    ~104 bytes on macOS, which pytest's tmp_path comfortably exceeds."""
    return tempfile.mkdtemp(prefix="exad-")


@pytest.fixture
def sockdir():
    d = _sockdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def daemon():
    d = _sockdir()
    path = os.path.join(d, "daemon.sock")
    proc = _spawn(path)
    yield path
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(d, ignore_errors=True)


def _stat(path):
    sock = _wire.connect(path)
    _wire.send(sock, {"op": "STATUS"})
    reply = _wire.recv(sock)
    sock.close()
    return reply


def _solves(path):
    return _stat(path)["solves"]


def _rosenbrock(n=12):
    core = exa.Core()
    x = core.add_var(n, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
    core.add_obj(lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,
                 over=range(1, n))
    return core, x


@requires("ipopt")
def test_the_daemon_serves_and_matches_the_in_process_answer(daemon, monkeypatch):
    monkeypatch.setenv("EXAMODELS_DAEMON", "0")
    core0, x0 = _rosenbrock()
    ref = exa.Model(core0).solve(print_level=0, sb="yes")

    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    core, x = _rosenbrock()
    assert isinstance(core, DispatchCore), "a reachable daemon means recording"
    model = exa.Model(core)
    assert isinstance(model, DaemonModel)
    before = _solves(daemon)
    sol = model.solve(print_level=0, sb="yes")
    assert _solves(daemon) == before + 1, "the daemon did not serve this solve"
    assert isinstance(sol, DaemonSolution)
    assert sol.status == ref.status and sol.success
    assert sol.objective == pytest.approx(ref.objective, rel=1e-9)
    np.testing.assert_allclose(sol[x], ref[x0], rtol=1e-8)
    np.testing.assert_allclose(sol.x, ref.x, rtol=1e-8)


@requires("ipopt")
def test_parameter_overrides_ride_along(daemon, monkeypatch):
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    core = exa.Core()
    x = core.add_var(3, start=0.0)
    p = core.add_par([1.0, 2.0, 3.0])
    core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(3))
    model = exa.Model(core)
    model.set_parameters(p, [5.0, 6.0, 7.0])
    before = _solves(daemon)
    sol = model.solve(print_level=0, sb="yes")
    assert _solves(daemon) == before + 1
    np.testing.assert_allclose(sol[x], [5.0, 6.0, 7.0], atol=1e-6)


@requires("ipopt")
def test_a_dying_daemon_never_loses_the_run(sockdir, monkeypatch):
    path = os.path.join(sockdir, "short-lived.sock")
    proc = _spawn(path)
    monkeypatch.setenv("EXAMODELS_DAEMON", path)
    core, x = _rosenbrock()
    model = exa.Model(core)
    assert isinstance(model, DaemonModel)
    subprocess.run([sys.executable, "-m", "examodels.daemon", "stop",
                    "--socket", path], check=True, timeout=30)
    proc.wait(10)
    sol = model.solve(print_level=0, sb="yes")     # in-process fallback
    assert sol.success
    assert model._eager is not None, "the fallback model should now exist"
    monkeypatch.setenv("EXAMODELS_DAEMON", "0")
    ref_core, ref_x = _rosenbrock()
    ref = exa.Model(ref_core).solve(print_level=0, sb="yes")
    assert sol.objective == pytest.approx(ref.objective, rel=1e-8)
    np.testing.assert_allclose(sol[x], ref[ref_x], rtol=1e-8)


def test_an_oracle_converts_the_core_to_eager(daemon, monkeypatch):
    from test_advanced import unit_circle_oracle
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    core = exa.Core()
    assert isinstance(core, DispatchCore)
    x = exa.add_var(core, 2, start=0.5)
    exa.add_obj(core, lambda i: -x[0] - x[1], over=range(1))
    exa.add_con(core, unit_circle_oracle())        # a record cannot carry this
    assert type(core) is EagerCore, "the core should have become eager"
    model = exa.Model(core)
    assert not isinstance(model, DaemonModel)
    assert model.ncon == 1, "construction continued seamlessly across the switch"


def test_version_skew_disables_dispatch(daemon, monkeypatch):
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    real = _wire.identity()
    monkeypatch.setattr(_wire, "identity",
                        lambda: {**real, "proto": real["proto"] + 1})
    core = exa.Core()
    assert type(core) is EagerCore, "a refused handshake must mean eager"


def test_the_status_cli_reports(daemon):
    out = subprocess.run(
        [sys.executable, "-m", "examodels.daemon", "status", "--socket", daemon],
        capture_output=True, text=True, timeout=30)
    assert out.returncode == 0
    assert "solves" in out.stdout


def test_a_second_daemon_refuses_the_busy_socket(daemon):
    out = subprocess.run(
        [sys.executable, "-m", "examodels.daemon", "serve", "--socket", daemon],
        capture_output=True, text=True, timeout=30)
    assert out.returncode == 1
    assert "already serving" in out.stderr


# ---- phase 2: live instances -------------------------------------------------

@requires("ipopt")
def test_a_repeated_model_reuses_the_live_instance(daemon, monkeypatch):
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    s0 = _stat(daemon)
    core1, x1 = _rosenbrock(9)
    m1 = exa.Model(core1)               # bound: the lease must outlive m2's build
    a = m1.solve(print_level=0, sb="yes")
    core2, x2 = _rosenbrock(9)          # same structure, same data
    m2 = exa.Model(core2)
    b = m2.solve(print_level=0, sb="yes")
    s1 = _stat(daemon)
    assert s1["solves"] == s0["solves"] + 2
    assert s1["replays"] == s0["replays"] + 1, "the second build must hit live"
    assert s1["hits"] == s0["hits"] + 3, "build hit + two solve hits"
    np.testing.assert_allclose(b.x, a.x, rtol=1e-12)


@requires("ipopt")
def test_a_parameter_sweep_replays_once(daemon, monkeypatch):
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    core = exa.Core()
    x = core.add_var(4, start=0.0)
    p = core.add_par([0.0] * 4)
    core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(4))
    s0 = _stat(daemon)
    model = exa.Model(core)              # the build (and the one replay) is here
    for k in range(6):
        target = [float(k + j) for j in range(4)]
        model.set_parameters(p, target)
        sol = model.solve(print_level=0, sb="yes")
        np.testing.assert_allclose(sol[x], target, atol=1e-6)
    s1 = _stat(daemon)
    assert s1["solves"] == s0["solves"] + 6
    assert s1["replays"] == s0["replays"] + 1, "a sweep is one replay, then hits"
    assert s1["hits"] == s0["hits"] + 6
    assert s1["records_received"] == s0["records_received"] + 1, \
        "the record must cross the wire exactly once for a sweep"


@requires("ipopt")
def test_an_instance_hit_takes_the_records_own_parameter_values(daemon, monkeypatch):
    """Parameter values live outside both digests (the cache invariant), so
    two records differing ONLY in add_par values share one instance — and a
    hit that trusted the instance's last values would give the second run
    the first run's answer."""
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)

    def build(vals):
        core = exa.Core()
        x = core.add_var(2, start=0.0)
        p = core.add_par(vals)
        core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(2))
        return exa.Model(core), x

    s0 = _stat(daemon)
    m1, x1 = build([1.0, 1.0])
    a = m1.solve(print_level=0, sb="yes")
    m2, x2 = build([9.0, 9.0])
    b = m2.solve(print_level=0, sb="yes")
    s1 = _stat(daemon)
    assert s1["replays"] == s0["replays"] + 1 and s1["hits"] == s0["hits"] + 3, \
        "these two records must share one instance"
    np.testing.assert_allclose(a[x1], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(b[x2], [9.0, 9.0], atol=1e-6)


@requires("ipopt")
def test_eviction_rebuilds_and_stays_correct(sockdir, monkeypatch):
    path = os.path.join(sockdir, "tiny.sock")
    proc = _spawn(path, "--max-instances", "1")
    try:
        monkeypatch.setenv("EXAMODELS_DAEMON", path)
        core_a, _ = _rosenbrock(7)
        core_b, _ = _rosenbrock(8)
        m_a, m_b = exa.Model(core_a), exa.Model(core_b)
        a1 = m_a.solve(print_level=0, sb="yes")   # A evicted by B's build: rebuild
        m_b.solve(print_level=0, sb="yes")        # B evicted by A's rebuild: rebuild
        a2 = m_a.solve(print_level=0, sb="yes")   # and again
        s = _stat(path)
        assert s["instances"] == 1
        assert s["evictions"] == 4
        assert s["replays"] == 5, "2 builds + 3 post-eviction rebuilds"
        assert a2.objective == pytest.approx(a1.objective, rel=1e-9)
    finally:
        subprocess.run([sys.executable, "-m", "examodels.daemon", "stop",
                        "--socket", path], timeout=30)
        proc.wait(10)


# ---- phase 3: lifetime -------------------------------------------------------

@requires("ipopt")
def test_client_exit_destroys_its_instances(daemon, monkeypatch):
    """The directive this phase implements: a model object must not outlive
    the client process that asked for it. Only compilation caches stay."""
    code = (
        "import os\n"
        f"os.environ['EXAMODELS_DAEMON'] = {daemon!r}\n"
        "import examodels as exa\n"
        "core = exa.Core()\n"
        "x = core.add_var(5, start=0.0)\n"
        "core.add_obj(lambda i: (x[i] - 2.5) ** 2, over=range(5))\n"
        "m = exa.Model(core)\n"
        "s = m.solve(print_level=0, sb='yes')\n"
        "assert s.success\n"
    )
    s0 = _stat(daemon)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    deadline = time.time() + 10          # the hangup reaches the worker async
    while time.time() < deadline:
        s1 = _stat(daemon)
        if (s1["instances"] == s0["instances"]
                and s1["destroyed"] == s0["destroyed"] + 1):
            break
        time.sleep(0.2)
    assert s1["instances"] == s0["instances"], \
        "the exited client's instance is still alive in the daemon"
    assert s1["destroyed"] == s0["destroyed"] + 1


@requires("ipopt")
def test_a_shared_instance_survives_until_its_last_owner_leaves(daemon, monkeypatch):
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    core1, _ = _rosenbrock(11)
    keeper = exa.Model(core1)            # this process holds a lease
    s0 = _stat(daemon)
    code = (
        "import os\n"
        f"os.environ['EXAMODELS_DAEMON'] = {daemon!r}\n"
        "import examodels as exa\n"
        "core = exa.Core()\n"
        "x = core.add_var(11, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(11)])\n"
        "core.add_obj(lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - 1) ** 2,\n"
        "             over=range(1, 11))\n"
        "m = exa.Model(core)\n"
        "assert m.solve(print_level=0, sb='yes').success\n"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    time.sleep(2)                        # give a wrong teardown time to happen
    s1 = _stat(daemon)
    assert s1["destroyed"] == s0["destroyed"], \
        "the other client's exit must not kill an instance this one still leases"
    assert keeper.solve(print_level=0, sb="yes").success


# ---- phase 4: interrupts + lifetime policy ----------------------------------

@requires("ipopt")
def test_a_client_killed_mid_solve_leaves_a_healthy_daemon(daemon, monkeypatch):
    """The design's v1 interrupt policy: an abandoned request is finished by
    the daemon and discarded; the dead client's lease is torn down; nobody
    else notices."""
    code = (
        "import os\n"
        f"os.environ['EXAMODELS_DAEMON'] = {daemon!r}\n"
        "import examodels as exa\n"
        "n = 50_000\n"
        "core = exa.Core()\n"
        "x = core.add_var(n, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)])\n"
        "core.add_obj(lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2"
        " + (x[i - 1] - 1) ** 2, over=range(1, n))\n"
        "m = exa.Model(core)\n"
        "print('BUILT', flush=True)\n"
        "m.solve(print_level=0, sb='yes', max_iter=300)\n"   # bounded orphan
        "print('SOLVED', flush=True)\n"
    )
    s0 = _stat(daemon)
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    assert "BUILT" in line, "client never got its model built"
    time.sleep(1.0)                  # let the solve get airborne
    proc.kill()                      # no goodbye of any kind
    proc.wait(10)
    deadline = time.time() + 120     # the abandoned solve runs to completion
    while time.time() < deadline:
        s1 = _stat(daemon)           # <- the daemon answering IS the health check
        if s1["current"] is None and s1["instances"] == s0["instances"]:
            break
        time.sleep(0.5)
    assert s1["current"] is None, "the daemon is still stuck on the orphan solve"
    assert s1["instances"] == s0["instances"], "the dead client's lease survived"
    # and it still serves: a fresh solve straight through
    monkeypatch.setenv("EXAMODELS_DAEMON", daemon)
    core, x = _rosenbrock(10)
    assert exa.Model(core).solve(print_level=0, sb="yes").success


@requires("ipopt")
def test_idle_exit_stops_a_daemon_nobody_uses(sockdir):
    path = os.path.join(sockdir, "sleepy.sock")
    proc = _spawn(path, "--idle-exit", "0.03")      # 1.8 s
    code = (
        "import os\n"
        f"os.environ['EXAMODELS_DAEMON'] = {path!r}\n"
        "import examodels as exa\n"
        "core = exa.Core()\n"
        "x = core.add_var(4, start=0.0)\n"
        "core.add_obj(lambda i: (x[i] - 1.0) ** 2, over=range(4))\n"
        "assert exa.Model(core).solve(print_level=0, sb='yes').success\n"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    try:
        rc = proc.wait(30)           # client gone; idle clock runs out
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("idle daemon never exited") from None
    assert rc == 0
    assert not os.path.exists(path), "the socket must be cleaned up on exit"
