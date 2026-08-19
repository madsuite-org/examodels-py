"""Framing and handshake for the warm-session daemon — shared by both ends.

A connection speaks length-prefixed pickle frames (protocol 5, out-of-band
numpy buffers). Pickle is acceptable exactly because the socket is a
same-user boundary: the daemon runs as the user, executing the user's own
models. Anything crossing a wider boundary must not be pickle — which is why
there is no TCP transport here.

The handshake refuses on any version difference (`agree`): skew can never
produce wrong numbers, only a fall back to the in-process path.
"""
import io
import os
import pickle
import socket
import struct

PROTO = 1

_LEN = struct.Struct("!Q")


def default_socket_path():
    """Where the daemon listens unless `MADSUITE_DAEMON` says otherwise."""
    run = os.environ.get("XDG_RUNTIME_DIR")
    base = os.path.join(run, "madsuite") if run else f"/tmp/madsuite-{os.getuid()}"
    return os.path.join(base, "daemon.sock")


def socket_path():
    """The configured endpoint, or None when dispatch is disabled."""
    v = os.environ.get("MADSUITE_DAEMON")
    if v in ("0", "no", "off"):
        return None
    if v:
        return v
    return default_socket_path()


def identity():
    """What both ends compare: protocol, package version, backend pin, python."""
    import sys

    from . import __version__
    from ._cache import _pinned_backend
    return {"proto": PROTO, "madsuite": __version__,
            "backend_pin": _pinned_backend(),
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}"}


def agree(mine, theirs):
    return mine == theirs


def send(sock, obj):
    """One frame: 8-byte length, then pickle-5 with out-of-band buffers.

    Buffers are appended after the main payload, each with its own length
    prefix, so numpy arrays cross without a serialization copy."""
    bufs = []
    body = pickle.dumps(obj, protocol=5, buffer_callback=bufs.append)
    parts = [_LEN.pack(len(bufs)), _LEN.pack(len(body)), body]
    for b in bufs:
        raw = b.raw()
        parts.append(_LEN.pack(raw.nbytes))
        parts.append(raw)
    sock.sendall(b"".join(parts))


def recv(sock):
    """The next frame, or None on a cleanly closed connection."""
    head = _read(sock, _LEN.size, eof_ok=True)
    if head is None:
        return None
    nbufs = _LEN.unpack(head)[0]
    body = _read(sock, _LEN.unpack(_read(sock, _LEN.size))[0])
    bufs = []
    for _ in range(nbufs):
        n = _LEN.unpack(_read(sock, _LEN.size))[0]
        bufs.append(_read(sock, n))
    return pickle.loads(body, buffers=bufs)


def _read(sock, n, eof_ok=False):
    buf = io.BytesIO()
    while buf.tell() < n:
        chunk = sock.recv(n - buf.tell())
        if not chunk:
            if eof_ok and buf.tell() == 0:
                return None
            raise ConnectionError("connection closed mid-frame")
        buf.write(chunk)
    return buf.getvalue()


def connect(path, timeout=5.0):
    """A handshaken client socket, or None if no compatible daemon answers.

    Every failure mode — no socket, stale socket, version skew, a daemon too
    busy to answer the handshake — is the same None: the caller falls back
    to the in-process path and the run stays correct."""
    if path is None:
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        send(s, {"op": "HELLO", "identity": identity()})
        reply = recv(s)
        if reply is None or not reply.get("ok"):
            s.close()
            return None
        s.settimeout(None)
        return s
    except (OSError, ConnectionError):
        s.close()
        return None
