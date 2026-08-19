"""Framing and configuration for the warm session — pure, no Julia, no daemon."""
import socket
import threading

import numpy as np
import pytest

from madsuite import _wire


def test_roundtrip_carries_numpy_out_of_band():
    a, b = socket.socketpair()
    payload = {"x": np.arange(100_000, dtype=np.float64), "s": "hi", "n": 3,
               "nested": {"y": np.ones(7)}}
    t = threading.Thread(target=_wire.send, args=(a, payload))
    t.start()
    got = _wire.recv(b)
    t.join()
    a.close()
    b.close()
    np.testing.assert_array_equal(got["x"], payload["x"])
    np.testing.assert_array_equal(got["nested"]["y"], payload["nested"]["y"])
    assert got["s"] == "hi" and got["n"] == 3


def test_a_clean_close_reads_as_none():
    a, b = socket.socketpair()
    a.close()
    assert _wire.recv(b) is None
    b.close()


def test_a_mid_frame_close_raises():
    a, b = socket.socketpair()
    a.sendall(_wire._LEN.pack(1))          # promises a frame, delivers nothing
    a.close()
    with pytest.raises(ConnectionError):
        _wire.recv(b)
    b.close()


def test_the_env_var_configures_and_disables(monkeypatch):
    monkeypatch.setenv("MADSUITE_DAEMON", "0")
    assert _wire.socket_path() is None
    monkeypatch.setenv("MADSUITE_DAEMON", "/somewhere/else.sock")
    assert _wire.socket_path() == "/somewhere/else.sock"
    monkeypatch.delenv("MADSUITE_DAEMON")
    assert _wire.socket_path().endswith("daemon.sock")


def test_identity_disagreement_is_refusal():
    me = {"proto": _wire.PROTO, "madsuite": "0.1.0",
          "backend_pin": "=0.12.0", "python": "3.12"}
    assert _wire.agree(me, dict(me))
    assert not _wire.agree(me, {**me, "madsuite": "0.2.0"})
    assert not _wire.agree(me, None)


def test_connect_returns_none_when_nothing_listens(tmp_path):
    assert _wire.connect(str(tmp_path / "nobody.sock")) is None
    assert _wire.connect(None) is None
