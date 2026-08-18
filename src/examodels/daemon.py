"""The warm session: `examodels` runs it; scripts dispatch to it.

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
    """What `examodels status` reports. Lock-protected counters, no Julia."""

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

    def snapshot(self):
        with self._lock:
            return {"pid": os.getpid(), "uptime": time.time() - self.started,
                    "solves": self.solves, "errors": self.errors,
                    "queued": self.queued, "current": self.current,
                    "replays": self.replays, "hits": self.hits,
                    "instances": self.instances, "evictions": self.evictions,
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


def _instance(payload, instances, cap, state):
    """The live model for this request — served from the instance table when
    the (fingerprint, data digest) pair matches, replayed otherwise.

    Parameter VALUES live outside both digests (the cache invariant this
    reuses), so a hit must not trust the values the instance last solved
    with: the incoming record's own values are pushed first, then the
    request's overrides.  A fresh replay bakes the record's values already.
    """
    from .model import Model

    record = payload["record"]
    args = payload.get("args", ())
    key = payload.get("key")
    hit = key is not None and key in instances
    if hit:
        instances.move_to_end(key)
        model, pars = instances[key]
        with state._lock:
            state.hits += 1
        for r in record._records:
            if r["kind"] == "par":
                model.set_parameters(pars[("par", r["ordinal"])], r["values"])
    else:
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
        with state._lock:
            state.replays += 1
        if key is not None:
            instances[key] = (model, pars)
            evicted = 0
            while len(instances) > cap:
                instances.popitem(last=False)
                evicted += 1
            if evicted:
                with state._lock:
                    state.evictions += evicted
                _reclaim()
        with state._lock:
            state.instances = len(instances)
    for pkey, values in (payload.get("params") or {}).items():
        model.set_parameters(pars[pkey], values)
    return model


def _reclaim():
    """Best-effort device-memory release after an eviction; refined in the
    memory/lifetime phase.  Only touches CUDA when it is already loaded."""
    with contextlib.suppress(Exception):
        from . import _bridge as _b
        _b.seval("GC.gc()")
        if _b.seval("isdefined(Main, :CUDA)"):
            _b.seval("Main.CUDA.reclaim()")


def _solve(payload, instances, cap, state):
    """Serve one request on the Julia-owning thread. Every exception is the
    caller's answer, never the daemon's problem: the reply carries it back
    and the daemon stays up."""
    from . import _bridge as _b

    model = _instance(payload, instances, cap, state)
    solver = payload.get("solver")
    options = payload.get("options") or {}
    sol = model.solve(solver=solver, **options)
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
    return {"ok": True, "status": sol.status, "objective": float(sol.objective),
            "iterations": iterations, "elapsed": sol.elapsed,
            "x": host("solution"), "y": host("multipliers"),
            "zL": host("multipliers_L"), "zU": host("multipliers_U")}


def _work(jobs, state, stop, cap):
    """The main-thread loop: everything that touches Julia happens here.
    The instance table lives here too — main-thread-confined, like Julia."""
    import collections
    instances = collections.OrderedDict()
    while not stop.is_set():
        try:
            payload, reply = jobs.get(timeout=0.5)
        except queue.Empty:
            continue
        with state.job(payload.get("label") or "solve"):
            try:
                result = _solve(payload, instances, cap, state)
                with state._lock:
                    state.solves += 1
            except Exception as e:                           # noqa: BLE001
                with state._lock:
                    state.errors += 1
                result = {"ok": False, "error": {
                    "type": type(e).__name__, "message": str(e)}}
        reply.put(result)


def _client(conn, jobs, state, stop):
    me = _wire.identity()
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
            elif op == "SOLVE":
                with state._lock:
                    state.queued += 1
                reply = queue.Queue()
                jobs.put((req, reply))
                _wire.send(conn, reply.get())
            else:
                _wire.send(conn, {"ok": False, "error": {
                    "type": "ValueError", "message": f"unknown op {op!r}"}})
    except (OSError, ConnectionError):
        return                      # client went away; nothing to answer
    finally:
        conn.close()


def serve(path=None, max_instances=32):
    path = path or _wire.default_socket_path()
    # The daemon must never dispatch to a daemon: replay constructs Core()s,
    # and dispatch here would mean connecting to ourselves.
    os.environ["EXAMODELS_DAEMON"] = "0"
    # A raw connect, not a handshake: a live daemon of any version keeps its
    # socket; only a socket nothing is listening on is stale.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(path)
        probe.close()
        print(f"examodels: a daemon is already serving {path}", file=sys.stderr)
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
    print(f"examodels: warm session on {path} (C-c to close)", flush=True)
    try:
        _work(jobs, state, stop, max_instances)
    except KeyboardInterrupt:
        print("\nexamodels: closing")
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
        print("examodels: no daemon running", file=sys.stderr)
        return 1
    try:
        _wire.send(sock, {"op": op})
        reply = _wire.recv(sock)
    finally:
        sock.close()
    if op == "STATUS" and reply and reply.get("ok"):
        up = int(reply["uptime"])
        print(f"pid {reply['pid']}, up {up // 3600}h{(up % 3600) // 60:02d}m, "
              f"{reply['solves']} solves ({reply['errors']} errors), "
              f"{reply['queued']} queued"
              + (f", solving: {reply['current']}" if reply["current"] else ""))
    elif op == "SHUTDOWN":
        print("examodels: daemon stopped")
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="examodels",
        description="Warm session for examodels: keeps Julia, the solver "
                    "stack and compiled kernels alive between runs. Scripts "
                    "dispatch to it automatically while it runs.")
    parser.add_argument("command", nargs="?", choices=["serve", "status", "stop"],
                        default="serve")
    parser.add_argument("--socket", help="listen/connect here instead of the default")
    parser.add_argument("--max-instances", type=int, default=32,
                        help="live models kept before the least recent is "
                             "dropped (a count for now; a device-byte budget "
                             "comes with the memory phase)")
    ns = parser.parse_args(argv)
    if ns.command == "serve":
        return serve(ns.socket, ns.max_instances)
    return _ask(ns.socket, "STATUS" if ns.command == "status" else "SHUTDOWN")


if __name__ == "__main__":
    sys.exit(main())
