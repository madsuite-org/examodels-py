"""Client side of the warm session — dispatch, and the model it returns.

While a daemon is reachable, `Core(...)` records instead of building eagerly,
and `Model(core)` ships the record; the user's script runs unchanged in the
user's own process. Every failure — no daemon, version skew, the daemon dying
mid-run — falls back to the in-process path, so the presence of a daemon can
change timing but never behavior. The one thing a record cannot carry is a
Python-callback oracle; hitting one converts the core to eager on the spot
(`DispatchCore._go_eager`) and the run simply does not use the daemon.
"""
import contextlib
import contextvars

import numpy as np

from . import _wire
from ._record import RecordingCore
from .model import Model

__all__ = ["dispatching", "DispatchCore", "DaemonModel", "DaemonSolution"]

_suspended = contextvars.ContextVar("madsuite-no-dispatch", default=False)


@contextlib.contextmanager
def suspended():
    """No dispatch while this is open. `replay()` wraps its eager `Core(...)`
    in it: a replay IS the fall-back (or the daemon's own build), so letting
    the construction probe the still-reachable daemon would hand back another
    recording core — infinite recursion, observed as such."""
    token = _suspended.set(True)
    try:
        yield
    finally:
        _suspended.reset(token)


def dispatching():
    """True when a compatible daemon answers right now.

    Probed per `Core(...)`: one connect + handshake against a local socket,
    microseconds when absent (ENOENT) and milliseconds when present — and
    never cached, so a daemon started or stopped between two cores in one
    process is seen. The generous timeout only ever costs anything when a
    daemon exists but is busy — and then eagerly giving up is the expensive
    choice, not the wait."""
    if _suspended.get():
        return False
    sock = _wire.connect(_wire.socket_path(), timeout=10.0)
    if sock is None:
        return False
    sock.close()
    return True


class DispatchCore(RecordingCore):
    """Records for the daemon; becomes an eager `Core` if it must.

    The conversion: replay the partial record (which grafts every handle the
    caller already holds), then *become* the eager core, so construction
    continues seamlessly on whatever call recording could not take. The
    recorder refuses with TypeError and NotImplementedError (oracles, tags,
    expression shapes it cannot carry); both are caught broadly on the add
    methods: over-catching costs a Julia boot before re-raising a genuine
    user error from the eager path, while under-catching would make a script
    fail only-when-a-daemon-runs — the invariant violation this class exists
    to prevent."""

    def __init__(self, backend=None, minimize=True, nargs=0, cache=None):
        super().__init__(backend=backend, minimize=minimize, nargs=nargs,
                         cache=None)

    def _go_eager(self):
        eager = self.replay()
        self.__dict__ = eager.__dict__
        self.__class__ = type(eager)

    def _or_eager(self, name, *args, **kwargs):
        try:
            return getattr(super(), name)(*args, **kwargs)
        except (TypeError, NotImplementedError):
            self._go_eager()
            return getattr(self, name)(*args, **kwargs)

    def add_var(self, *args, **kwargs):
        return self._or_eager("add_var", *args, **kwargs)

    def add_par(self, *args, **kwargs):
        return self._or_eager("add_par", *args, **kwargs)

    def add_obj(self, *args, **kwargs):
        return self._or_eager("add_obj", *args, **kwargs)

    def add_con(self, *args, **kwargs):
        return self._or_eager("add_con", *args, **kwargs)


def try_daemon(core, args):
    """A `DaemonModel` for this record, or None to fall through to MISS.

    Model creation is a daemon-side BUILD, separate from any solve: the
    instance exists (and its replay cost is paid) when `Model(core)`
    returns, mirroring the eager path — including errors, which are raised
    here, where eager construction would raise them.  The connection opened
    for the BUILD is the model's LEASE: it stays open for the model's
    lifetime, and the daemon destroys the instance when the last lease
    closes — a client exiting takes its models with it.

    Falling through to None is load-bearing for cache= records: with no
    daemon, a cache miss must materialize at `Model(core)` — synchronously,
    erroring where the eager path errors — exactly as it does today."""
    if core._nargs:
        return None       # a recipe's layout exists only per instance
    from ._cache import _layout
    fp, dd = core.fingerprint()
    key = (fp, dd)
    sock = _build_lease(core, key, tuple(args), fp[:12], raise_errors=True)
    if sock is None:
        return None
    self = object.__new__(DaemonModel)
    self.__dict__["_record"] = core
    self.__dict__["_args"] = tuple(args)
    self.__dict__["_overrides"] = {}
    self.__dict__["_key"] = key
    self.__dict__["_label"] = fp[:12]
    self.__dict__["_sock"] = sock
    var, con, _par, nvar, ncon = _layout(core)
    self.__dict__["_var"], self.__dict__["_con"] = var, con
    self.__dict__["_sizes"] = (nvar, ncon)
    self.__dict__["_named"] = dict(core._named)
    self.__dict__["_eager"] = None
    return self


def _build_lease(record, key, args, label, raise_errors=False):
    """A handshaken connection on which `key` is built and leased, or None.

    Key first: the record crosses only if the daemon does not already hold
    the instance.  A genuine model error during the daemon-side build is
    raised when `raise_errors` (eager parity at `Model(core)`); transport
    failures are always just None — the caller falls back in-process."""
    # Patient on purpose: a daemon busy replaying someone's model answers
    # slowly, and giving up here costs the client a minutes-long in-process
    # boot — seconds of waiting is the far cheaper side of that bet.
    sock = _wire.connect(_wire.socket_path(), timeout=15.0)
    if sock is None:
        return None
    base = {"op": "BUILD", "label": label, "key": key, "args": args}
    try:
        _wire.send(sock, {**base, "record": None})
        reply = _wire.recv(sock)
        if reply is not None and reply.get("need_record"):
            _wire.send(sock, {**base, "record": record})
            reply = _wire.recv(sock)
    except (OSError, ConnectionError):
        reply = None
    if reply is None or not reply.get("ok"):
        sock.close()
        if reply is not None and reply.get("error") and raise_errors:
            raise _mapped(reply["error"])
        return None
    return sock


def _recv_streaming(sock):
    """The final reply, printing solver-output frames as they arrive — the
    iteration log lands in this terminal, as it would for an eager solve."""
    import sys
    while True:
        reply = _wire.recv(sock)
        if reply is None or not reply.get("stream"):
            return reply
        sys.stdout.write(reply.get("out", ""))
        sys.stdout.flush()


def _mapped(err):
    """The daemon's error as the exception the eager path would have raised."""
    from ._bridge import ModelError
    kinds = {"ModelError": ModelError, "ValueError": ValueError,
             "TypeError": TypeError, "KeyError": KeyError}
    kind = kinds.get(err.get("type"))
    msg = err.get("message", "")
    return kind(msg) if kind else RuntimeError(f"daemon: {err.get('type')}: {msg}")


class DaemonModel(Model):
    """The `Model` surface while a daemon serves the solves.

    Holds the record, not a Julia model. `solve` ships (record, parameter
    overrides, options). Everything else `Model` can do — evaluation,
    setters, metadata — works through `_jl`, which here is a property that
    quietly builds the in-process model once and uses it from then on; that
    same lazily-built model is the fallback when the daemon disappears."""

    def __init__(self, *args, **kwargs):
        # Python re-invokes __init__ after Model.__new__ returned this
        # instance from the daemon lookup; it is already built.
        if "_record" not in self.__dict__:
            raise TypeError("DaemonModel is built by Model(core) while a "
                            "daemon is running")

    @property
    def _jl(self):
        return self._fallback()._jl

    @property
    def nvar(self):
        return self._sizes[0]      # from the record: no Julia for a size

    @property
    def ncon(self):
        return self._sizes[1]

    def set_parameters(self, block, values):
        """Change a parameter block's values; sent with every solve."""
        key = getattr(block, "_key", None)
        if key is None or key[0] != "par":
            raise TypeError(f"{block!r} is not a parameter block of this model")
        if self._eager is not None:
            self._eager.set_parameters(block, values)
        self._overrides[key] = np.asarray(values, dtype=np.float64).ravel()
        return self

    def solve(self, solver=None, **options):
        if self._eager is not None:
            return self._eager.solve(solver=solver, **options)
        # The solve rides the lease connection from Model-creation time.
        # Key first, record only on demand (an eviction under the daemon's
        # cap); parameter values always cross in full — they live outside
        # both digests, so the instance's last values say nothing about ours.
        base = {"op": "SOLVE", "label": self._label, "key": self._key,
                "args": self._args, "params": self._all_params(),
                "solver": solver, "options": options}
        for _attempt in (1, 2):
            sock = self._sock
            if sock is None:
                sock = _build_lease(self._record, self._key, self._args,
                                    self._label)
                if sock is None:
                    break           # no daemon anymore: in-process it is
                self.__dict__["_sock"] = sock
            try:
                _wire.send(sock, {**base, "record": None})
                reply = _recv_streaming(sock)
                if reply is not None and reply.get("need_record"):
                    _wire.send(sock, {**base, "record": self._record})
                    reply = _recv_streaming(sock)
            except (OSError, ConnectionError):
                reply = None
            except KeyboardInterrupt:
                # C-c mid-solve: closing the lease IS the cancel — the
                # daemon sees EOF and abandons the solve. The socket is
                # mid-protocol garbage now; drop it so a later solve (a
                # REPL user catching the interrupt) builds a fresh lease.
                sock.close()
                self.__dict__["_sock"] = None
                raise
            if reply is None:       # lease died mid-flight: retry once fresh
                sock.close()
                self.__dict__["_sock"] = None
                continue
            if not reply.get("ok"):
                raise _mapped(reply.get("error", {}))
            return DaemonSolution(reply, self._var, self._con)
        return self._fallback().solve(solver=solver, **options)

    def __del__(self):
        sock = self.__dict__.get("_sock")
        if sock is not None:
            # Closing the lease is what tells the daemon to tear the
            # instance down (once every other lease is gone too).
            import contextlib
            with contextlib.suppress(OSError):
                sock.close()

    def _all_params(self):
        """Every parameter block's values as this model sees them: the
        record's own values, with `set_parameters` overrides on top."""
        values = {("par", r["ordinal"]): r["values"]
                  for r in self._record._records if r["kind"] == "par"}
        values.update(self._overrides)
        return values

    def _fallback(self):
        """The in-process model, built once — replay grafts the caller's
        handles, so everything they hold keeps working. Going in-process
        releases the lease: the daemon must not hold an instance for a
        model that now lives here."""
        if self._eager is None:
            sock = self.__dict__.get("_sock")
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
                self.__dict__["_sock"] = None
            by_key = {h._key: h for h in self._record._issued}
            eager = Model(self._record.replay(), *self._args)
            for key, values in self._overrides.items():
                eager.set_parameters(by_key[key], values)
            self.__dict__["_eager"] = eager
        return self._eager

    def __repr__(self):
        via = "daemon" if self._eager is None else "in-process"
        return (f"<DaemonModel nvar={self.nvar} ncon={self.ncon} via {via}>")


class DaemonSolution:
    """The `Solution` surface, from the daemon's flattened reply."""

    __slots__ = ("_r", "_var", "_con", "elapsed")

    def __init__(self, reply, var, con):
        self._r, self._var, self._con = reply, var, con
        self.elapsed = reply.get("elapsed", float("nan"))

    @property
    def x(self):
        return self._r["x"]

    @property
    def y(self):
        return self._r["y"]

    @property
    def status(self):
        return self._r["status"]

    @property
    def objective(self):
        return self._r["objective"]

    @property
    def success(self):
        from .model import Solution
        return self.status in Solution._SUCCESS

    @property
    def iterations(self):
        n = self._r.get("iterations")
        if n is None:
            raise AttributeError("this solver did not report an iteration count")
        return n

    def _slice(self, table, vec, handle, what):
        if vec is None:
            raise AttributeError(f"the solver did not report {what} multipliers")
        key = getattr(handle, "_key", None)
        if key not in table:
            raise TypeError(f"{handle!r} is not a {what} block of this solution's model")
        off, n, _dims = table[key]
        v = np.asarray(vec, dtype=np.float64)[off:off + n]
        shape = getattr(handle, "shape", None)
        return v.reshape(shape, order="F") if shape else v

    def __getitem__(self, block):
        return self._slice(self._var, self._r["x"], block, "variable")

    def multipliers(self, constraint):
        return self._slice(self._con, self._r["y"], constraint, "constraint")

    def multipliers_L(self, block):
        return self._slice(self._var, self._r["zL"], block, "variable")

    def multipliers_U(self, block):
        return self._slice(self._var, self._r["zU"], block, "variable")

    def __repr__(self):
        return (f"<DaemonSolution status={self.status!r} "
                f"objective={self.objective:.6g}>")
