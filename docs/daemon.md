# The warm session — GPU solves without the per-run startup

[The first call costs minutes on a GPU](tutorial/gpu.md): Julia boots, the
CUDA stack loads, the device initializes, and the solver's kernels compile —
several minutes on a typical box, paid again by every fresh process. The
[model cache](cache.md) removes that price for CPU models by compiling them
to shared libraries; device code cannot be cached that way.

The warm session removes it by keeping one process alive. In a terminal (or
a tmux pane):

```console
$ examodels
examodels: warm session on /run/user/1001/examodels/daemon.sock (C-c to close)
```

and every script on the machine now dispatches to it, **unchanged**:

```console
$ python mymodel.py     # seconds, not minutes
```

Measured on one machine (AC OPF, pglib `case1354_pegase`, `backend="cuda"`,
client always a fresh process):

| | |
|---|---|
| no daemon: cold run to first solution | **308 s** |
| daemon running: same script | **~6 s** (a replay + solve) |
| the same *process* solving again (a sweep, new data) | **~0.7 s** |

## What actually happens

Your script never leaves your process — your environment, your debugger,
your `print`s all behave normally. While a session is reachable, `Core(...)`
records the model instead of building it (pure Python, milliseconds), and
`Model(core)` ships the recording to the session, which replays it through
the ordinary eager path and keeps the built model **live** for as long as
your process holds it. Solves send parameter values and return solution
arrays; `set_parameters` sweeps hit the live model at bare solve cost.

With no session running, nothing changes at all: the same script builds
eagerly in-process, exactly as this manual describes everywhere else. Every
failure — the session going down mid-run, a version mismatch after an
upgrade — falls back to the in-process path silently and correctly. A model
a recording cannot carry (a [Python-callback oracle](api.md)) simply builds
eagerly and skips the session.

## Lifetime: models die with your process

The session holds a model only while some client process holds it — exit,
crash, or drop the `Model` object, and the session destroys the built model,
runs the garbage collector, and releases device memory. Several processes
using the same model share one live instance until the last of them leaves.
What survives across clients is *compilation* only: Julia's compiled kernels
(which is why the second-ever client of a model family builds in seconds,
not minutes) and the [model cache](cache.md)'s libraries — a `cache=True`
model that misses is compiled and stored by the session as a side effect.

## Controls

```console
$ examodels status
pid 12345, up 2h03m, 41 solves (0 errors), 0 queued, 2 clients, ...
$ examodels stop
```

- `EXAMODELS_DAEMON=0` — this process never dispatches.
- `EXAMODELS_DAEMON=/path/to.sock` — dispatch there instead of the default
  (`$XDG_RUNTIME_DIR/examodels/daemon.sock`).
- `examodels --max-instances N` — live models kept before the least recently
  used is dropped (a dropped model rebuilds transparently on next use).
- `examodels --idle-exit MINUTES` — exit after that long with no clients and
  no work. Julia never unloads compiled code, so a long-lived session only
  grows; an occasional fresh start is the reset.

## Limits, honestly

- Solves are served **one at a time**, first come first served; `status`
  shows the queue. (The session stays responsive while solving.)
- Interrupting your script mid-solve abandons the request: the session
  finishes the solve, discards it, and cleans up your models. It cannot yet
  cancel the solver mid-flight.
- One machine only. The socket is same-user; nothing crosses the network.
