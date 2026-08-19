"""The warm session: `madsuite` runs it; scripts dispatch to it.

One foreground process owning one Julia runtime; C-c stops it. Clients send
recorded models (never scripts); the daemon replays each through the ordinary
eager path and solves. Julia boot, package load, CUDA initialization and JIT
are paid once here instead of once per run — see `design/daemon.md`.

Threading is dictated by juliacall, which wants Julia driven from one
thread: a first version that booted Julia from a worker thread hung at 0%
CPU on its first solve. So the MAIN thread owns Julia and consumes the
solve queue FIFO; a helper thread accepts connections, and per-connection
threads answer `HELLO`/`STATUS`/`SHUTDOWN` without ever touching Julia.
"""
import contextlib
import os
import queue
import socket
import sys
import threading
import time

import numpy as np

from . import _wire

__all__ = ["main", "serve"]


class _State:
    """What `madsuite status` reports. Lock-protected counters, no Julia."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.solves = 0
        self.errors = 0
        self.queued = 0
        self.current = None
        self.replays = 0        # instance built by replaying a record
        self.hits = 0           # instance served live, replay skipped
        self.instances = 0
        self.evictions = 0
        self.destroyed = 0      # instances torn down when their clients left
        self.records_received = 0   # full records that crossed the wire
        self.conns = 0          # open client connections (leases + probes)
        self.last_activity = time.time()   # last BUILD/SOLVE/HANGUP job
        self.mem = {}           # refreshed by the worker after each job

    def snapshot(self):
        with self._lock:
            return {"pid": os.getpid(), "uptime": time.time() - self.started,
                    "solves": self.solves, "errors": self.errors,
                    "queued": self.queued, "current": self.current,
                    "replays": self.replays, "hits": self.hits,
                    "instances": self.instances, "evictions": self.evictions,
                    "destroyed": self.destroyed,
                    "records_received": self.records_received,
                    "connections": self.conns, "mem": dict(self.mem),
                    "identity": _wire.identity()}

    @contextlib.contextmanager
    def job(self, label):
        with self._lock:
            self.queued -= 1
            self.current = label
        try:
            yield
        finally:
            with self._lock:
                self.current = None


def _instance(payload, instances, cap, state, lease):
    """The live model for this request — served from the instance table when
    the (fingerprint, data digest) pair matches, replayed otherwise.

    Instances belong to client connections: a `lease` (a BUILD, or a solve
    that had to build) adds this connection to the instance's owners, and
    `_release` destroys the instance when its last owner hangs up — a model
    object must never outlive every process that asked for it; only
    compilation caches stay warm.  On a hit the record is never consulted
    (the client did not even send it): parameter VALUES live outside both
    digests, so the client ships every block's values with every request,
    and that is the only per-request state there is.
    """
    from .model import Model

    args = payload.get("args", ())
    key = payload.get("key")
    conn = payload.get("conn")
    built = False
    if key is not None and key in instances:
        instances.move_to_end(key)
        model, pars, owners = instances[key]
        with state._lock:
            state.hits += 1
    else:
        record = payload["record"]
        with state._lock:
            state.records_received += 1
        if getattr(record, "cache", None):
            # A cache-carrying record missed its library client-side; the
            # replay here is materialize, so the entry is compiled and stored
            # as a side effect and the client's NEXT run hits julia-free,
            # daemon or not.
            from ._cache import materialize
            eager = materialize(record, args)
        else:
            eager = record.replay()
        model = Model(eager, *args)
        # Replay grafts the eager handles onto the record's own issued
        # handles, so block keys address this model directly.
        pars = {h._key: h for h in record._issued if h._key[0] == "par"}
        owners = {}
        built = True
        with state._lock:
            state.replays += 1
        if key is not None:
            instances[key] = (model, pars, owners)
            evicted = 0
            while len(instances) > cap:
                instances.popitem(last=False)
                evicted += 1
            if evicted:
                with state._lock:
                    state.evictions += evicted
                _reclaim()
    if lease and conn is not None and key is not None:
        owners[conn] = owners.get(conn, 0) + 1
    with state._lock:
        state.instances = len(instances)
    for pkey, values in (payload.get("params") or {}).items():
        model.set_parameters(pars[pkey], values)
    return model, built


def _release(conn, keys, instances, state):
    """A client hung up: drop its leases; destroy whatever nobody owns."""
    died = 0
    for key in keys:
        entry = instances.get(key)
        if entry is None:
            continue                # evicted earlier; nothing left to drop
        owners = entry[2]
        n = owners.get(conn, 0)
        if n <= 1:
            owners.pop(conn, None)
        else:
            owners[conn] = n - 1
        if not owners:
            del instances[key]
            died += 1
    with state._lock:
        state.destroyed += died
        state.instances = len(instances)
    if died:
        _reclaim()


_UNLOCKED = {}


def _gil_free_solve(model, solver, options):
    """`model.solve(...)`, with the GIL released for the pure-Julia part.

    A solve can run for minutes, and the worker holding the GIL that whole
    time freezes every other thread — including the ones answering `status`,
    which the design promises stays honest during long solves.  Releasing is
    safe here and only here: an oracle model (Julia calling back into
    Python) is unrecordable, so one can never reach the daemon."""
    import time as _time

    from . import _bridge as _b
    from .model import Solution
    from .solve import _prepared

    entry, options, name = _prepared(model, solver, options)
    fn = _UNLOCKED.get(name)
    if fn is None:
        fn = _b.seval(
            "f -> ((m; kw...) -> PythonCall.GIL.@unlock f(m; kw...))")(entry)
        _UNLOCKED[name] = fn
    t0 = _time.perf_counter()
    raw = _b.guard(fn, model._jl, **options)
    return Solution(raw, elapsed=_time.perf_counter() - t0)


def _reclaim():
    """Best-effort device-memory release after an eviction; refined in the
    memory/lifetime phase.  Only touches CUDA when it is already loaded."""
    with contextlib.suppress(Exception):
        from . import _bridge as _b
        _b.seval("GC.gc()")
        if _b.seval("isdefined(Main, :CUDA)"):
            _b.seval("Main.CUDA.reclaim()")


def _solve(payload, instances, cap, state, build_only=False):
    """Serve one request on the Julia-owning thread. Every exception is the
    caller's answer, never the daemon's problem: the reply carries it back
    and the daemon stays up."""
    from . import _bridge as _b

    key = payload.get("key")
    if payload.get("record") is None and not (key is not None and key in instances):
        # Key-first negotiation: the client held the ~MB record back; this
        # key is not live here, so ask for it (one extra round trip, paid
        # only on the path that is about to replay anyway).
        return {"ok": False, "need_record": True}
    # A BUILD always leases; a solve leases only when it carried a record
    # (a post-eviction rebuild — its BUILD-time lease covers the hit case).
    leased = build_only or payload.get("record") is not None
    model, _built = _instance(payload, instances, cap, state, lease=leased)
    if build_only:
        return {"ok": True, "leased": True}
    solver = payload.get("solver")
    options = payload.get("options") or {}
    sol = _gil_free_solve(model, solver, options)
    raw = sol._raw

    def host(name):
        try:
            return np.array(_b.tohost(getattr(raw, name)), dtype=np.float64)
        except Exception:                                    # noqa: BLE001
            return None

    try:
        iterations = int(sol.iterations)
    except Exception:                                        # noqa: BLE001
        iterations = None
    return {"ok": True, "leased": leased,
            "status": sol.status, "objective": float(sol.objective),
            "iterations": iterations, "elapsed": sol.elapsed,
            "x": host("solution"), "y": host("multipliers"),
            "zL": host("multipliers_L"), "zU": host("multipliers_U")}


@contextlib.contextmanager
def _stdout_to(reply):
    """Redirect this process's fd 1 into stream frames on `reply`.

    The solver's iteration log is written by Julia straight to fd 1 — the
    daemon's terminal — while the person who asked is watching a different
    one. During a solve, fd 1 becomes a pipe; a pump thread turns whatever
    arrives into {"stream": True, "out": ...} frames, which the connection
    thread forwards ahead of the final result. One solve at a time and only
    the worker redirects, so the juggling is race-free."""
    import threading as _threading
    r, w = os.pipe()
    saved = os.dup(1)
    os.dup2(w, 1)
    os.close(w)

    def pump():
        while True:
            chunk = os.read(r, 4096)
            if not chunk:
                os.close(r)
                return
            reply.put({"stream": True, "out": chunk.decode(errors="replace")})

    t = _threading.Thread(target=pump, daemon=True)
    t.start()
    try:
        yield
    finally:
        os.dup2(saved, 1)       # closes the pipe's last write end -> pump EOF
        os.close(saved)
        t.join(timeout=5)


def _work(jobs, state, stop, cap, idle_exit=None):
    """The main-thread loop: everything that touches Julia happens here.
    The instance table lives here too — main-thread-confined, like Julia.

    `idle_exit` (seconds) is the restart-on-idle policy from the design:
    Julia never unloads compiled code, so a long-lived daemon only grows;
    with no client connected and no work arriving for that long, exiting
    is the cleanup. The next `madsuite` starts fresh."""
    import collections
    instances = collections.OrderedDict()
    while not stop.is_set():
        try:
            kind, payload, reply = jobs.get(timeout=0.5)
        except queue.Empty:
            with state._lock:
                idle = (state.conns == 0
                        and time.time() - state.last_activity > (idle_exit or 0))
            if idle_exit is not None and idle:
                print(f"madsuite: idle for {int(idle_exit)}s with no "
                      f"clients; exiting", flush=True)
                stop.set()
            continue
        with state._lock:
            state.last_activity = time.time()
        if kind == "HANGUP":
            _release(payload["conn"], payload["keys"], instances, state)
            _refresh_mem(state)
            continue
        with state.job(payload.get("label") or "solve"):
            try:
                if kind == "SOLVE":
                    with _stdout_to(reply):
                        result = _solve(payload, instances, cap, state)
                else:
                    result = _solve(payload, instances, cap, state,
                                    build_only=True)
                if kind == "SOLVE" and result.get("ok"):
                    # neither a build nor a need_record round is a solve
                    with state._lock:
                        state.solves += 1
            except Exception as e:                           # noqa: BLE001
                with state._lock:
                    state.errors += 1
                result = {"ok": False, "error": {
                    "type": type(e).__name__, "message": str(e)}}
        reply.put(result)
        _refresh_mem(state)


def _refresh_mem(state):
    """Host RSS and device memory, refreshed on the worker thread — the only
    thread allowed to ask Julia. Read by `status` from wherever."""
    from . import _bridge as _b
    rss_mb = None
    with contextlib.suppress(OSError, IndexError, ValueError):
        with open("/proc/self/statm") as f:
            rss_mb = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    dev = None
    if _b.loaded():
        with contextlib.suppress(Exception):
            if _b.seval("isdefined(Main, :CUDA)"):
                free = float(_b.seval("Float64(Main.CUDA.free_memory())"))
                total = float(_b.seval("Float64(Main.CUDA.total_memory())"))
                dev = {"used_mb": (total - free) / 1e6, "total_mb": total / 1e6}
    with state._lock:
        state.mem = {"rss_mb": rss_mb, "device": dev}


def _client(conn, jobs, state, stop):
    me = _wire.identity()
    conn_id = id(conn)
    owned = []                      # keys this connection holds leases on
    with state._lock:
        state.conns += 1
    try:
        while True:
            req = _wire.recv(conn)
            if req is None:
                return
            op = req.get("op")
            if op == "HELLO":
                ok = _wire.agree(me, req.get("identity"))
                _wire.send(conn, {"ok": ok, "identity": me})
            elif op == "STATUS":
                _wire.send(conn, {"ok": True, **state.snapshot()})
            elif op == "SHUTDOWN":
                _wire.send(conn, {"ok": True})
                stop.set()
                return
            elif op in ("SOLVE", "BUILD"):
                req["conn"] = conn_id
                with state._lock:
                    state.queued += 1
                reply = queue.Queue()
                jobs.put((op, req, reply))
                while True:
                    result = reply.get()
                    if result.get("stream"):
                        # solver output for the client's terminal; if the
                        # client vanished, keep draining to the final frame
                        with contextlib.suppress(OSError, ConnectionError):
                            _wire.send(conn, result)
                        continue
                    break
                if result.get("leased") and req.get("key") is not None:
                    owned.append(req["key"])
                _wire.send(conn, result)
            else:
                _wire.send(conn, {"ok": False, "error": {
                    "type": "ValueError", "message": f"unknown op {op!r}"}})
    except (OSError, ConnectionError):
        return                      # client went away; teardown still runs
    finally:
        conn.close()
        with state._lock:
            state.conns -= 1
            state.last_activity = time.time()
        if owned:
            # The client is gone: its models must not pile up in here.  The
            # release runs on the worker thread, where the table lives.
            jobs.put(("HANGUP", {"conn": conn_id, "keys": owned}, None))


def serve(path=None, max_instances=32, idle_exit=None):
    path = path or _wire.default_socket_path()
    # The daemon must never dispatch to a daemon: replay constructs Core()s,
    # and dispatch here would mean connecting to ourselves.
    os.environ["MADSUITE_DAEMON"] = "0"
    # A raw connect, not a handshake: a live daemon of any version keeps its
    # socket; only a socket nothing is listening on is stale.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(path)
        probe.close()
        print(f"madsuite: a daemon is already serving {path}", file=sys.stderr)
        return 1
    except OSError:
        probe.close()
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)             # stale socket from an unclean exit
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen()

    state = _State()
    jobs = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_accept, args=(listener, jobs, state, stop),
                     daemon=True).start()
    print(f"madsuite: warm session on {path} (C-c to close)", flush=True)
    try:
        _work(jobs, state, stop, max_instances, idle_exit)
    except KeyboardInterrupt:
        print("\nmadsuite: closing")
    finally:
        listener.close()
        with contextlib.suppress(OSError):
            os.unlink(path)
    return 0


def _accept(listener, jobs, state, stop):
    listener.settimeout(0.5)
    while not stop.is_set():
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            return                  # listener closed under us at shutdown
        threading.Thread(target=_client, args=(conn, jobs, state, stop),
                         daemon=True).start()


def _ask(path, op):
    sock = _wire.connect(path or _wire.default_socket_path(), timeout=2.0)
    if sock is None:
        print("madsuite: no daemon running", file=sys.stderr)
        return 1
    try:
        _wire.send(sock, {"op": op})
        reply = _wire.recv(sock)
    finally:
        sock.close()
    if op == "STATUS" and reply and reply.get("ok"):
        up = int(reply["uptime"])
        mem = reply.get("mem") or {}
        dev = mem.get("device")
        line = (f"pid {reply['pid']}, up {up // 3600}h{(up % 3600) // 60:02d}m, "
                f"{reply['solves']} solves ({reply['errors']} errors), "
                f"{reply['queued']} queued, {reply['connections']} clients, "
                f"{reply['instances']} live instances "
                f"({reply['replays']} replays, {reply['hits']} hits, "
                f"{reply['evictions']} evictions, {reply['destroyed']} destroyed)")
        if mem.get("rss_mb"):
            line += f", rss {mem['rss_mb'] / 1000:.1f} GB"
        if dev:
            line += (f", device {dev['used_mb'] / 1000:.1f}"
                     f"/{dev['total_mb'] / 1000:.1f} GB")
        if reply["current"]:
            line += f", solving: {reply['current']}"
        print(line)
    elif op == "SHUTDOWN":
        print("madsuite: daemon stopped")
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="madsuite",
        description="Warm session for madsuite: keeps Julia, the solver "
                    "stack and compiled kernels alive between runs. Scripts "
                    "dispatch to it automatically while it runs.")
    parser.add_argument("command", nargs="?", choices=["serve", "status", "stop"],
                        default="serve")
    parser.add_argument("--socket", help="listen/connect here instead of the default")
    parser.add_argument("--max-instances", type=int, default=32,
                        help="live models kept before the least recent is "
                             "dropped; every drop rebuilds transparently on "
                             "next use")
    parser.add_argument("--idle-exit", type=float, metavar="MINUTES",
                        help="exit after this long with no clients and no "
                             "work — Julia never unloads compiled code, so "
                             "an occasional fresh start is the only reset")
    ns = parser.parse_args(argv)
    if ns.command == "serve":
        return serve(ns.socket, ns.max_instances,
                     None if ns.idle_exit is None else ns.idle_exit * 60)
    return _ask(ns.socket, "STATUS" if ns.command == "status" else "SHUTDOWN")


if __name__ == "__main__":
    sys.exit(main())
