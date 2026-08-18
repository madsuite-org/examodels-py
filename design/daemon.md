# Warm-session daemon — design

Status: **draft for review**. No implementation exists; this document is the
thing to disagree with. File references are against `d3b7a22`.

## What this is

A persistent session so command-line users stop paying Julia boot, package
load, CUDA initialization and first-solve JIT on every run:

```
$ examodels            # persistent session; C-c closes it; works as a daemon
```

and, in another shell or tmux pane, an **unchanged** user script

```
$ python mymodel.py    # automatically dispatches to the running daemon
```

The precompile/sysimage route (PackageCompiler) is explicitly out of scope:
GPU support there is immature, and this design must serve device solves
first.

## What a cold run costs

Measured on the group's GPU box (2× Quadro GV100, Volta; warm Julia depot;
package at `d3b7a22`; every scenario a fresh process; the first two scenarios
run twice with <5% spread; scripts in [`daemon-measurements/`](daemon-measurements/)):

| scenario (`backend="cuda"`, MadNLP + cuDSS) | cold, to first solution | same solve, warm | ratio |
|---|---|---|---|
| Lukšan–Vlček N=100 000 (`docs/tutorial/gpu.md`'s benchmark) | 244 s | 0.43 s | ~570× |
| AC OPF, pglib `case1354_pegase` | 308 s | 0.71 s | ~430× |
| `case2869_pegase`, *unseen* case, in the already-warm process | — | 8.5 s (7.4 build + 1.05 solve) | ~36× vs its own cold |

Decomposition of the fixed part (raw stack, no model): Julia boot ≈ 7 s;
`using` ExaModels + MadNLP + NLPModelsIpopt ≈ 3 s; `using` CUDA + CUDSS +
MadNLPGPU ≈ 7.5 s; **first CUDA operation ≈ 42 s** (context init; mechanism
not verified, plausibly PTX JIT for sm_70). Everything after that is
per-model-family first-solve JIT (157–200 s in the scenarios above).

The third row is the important one: JIT reuse is by Julia *expression type*,
which is broader than the cache's fingerprint. A warm process solved a
never-seen OPF case in 8.5 s because it shared expression types with the
case solved before it. A daemon inherits that for free.

## The decision: ship the model, not the script

The obvious implementation (DaemonMode.jl style) ships the *script* to the
daemon and executes it there. Rejected: the script would run in the daemon's
interpreter, so imports resolve against the daemon's environment (silent
wrong-library execution), `pdb` breaks, `__file__`/`cwd` need forwarding, and
edited user modules go stale in the warm process.

Instead the client ships the **recorded model**. The machinery already
exists for the compiled-library cache:

- `Core(cache=...)` builds a `RecordingCore` — expression trees stay pure
  Python, data stays numpy, *"Julia is never loaded"*
  (`_record.py` module docstring).
- A record already canonicalizes to a `(fingerprint, data_digest)` sha256
  pair (`_record.py:492`), structure and data separated, **parameter values
  outside both** (`_cache.py` module docstring) — precisely the keying a
  daemon needs.
- Records serialize. Verified at `d3b7a22`: a recording core with variables,
  parameters, objective and constraints pickles and round-trips with its
  fingerprint intact, without Julia entering the process (1 046 bytes for a
  toy model).
- A solution can be presented without a live Julia model in the client —
  `CachedSolution` (`_cache.py:515`) is the existing precedent.

The user's script runs in the user's process: their venv, their `pdb`, their
side effects. Only `(record, digests, parameter values, solver options)`
cross the wire; solution arrays come back.

## Architecture

`Model(core)` already resolves through a ladder; the daemon is one new rung:

```
HIT   compiled shared library, no Julia (CPU/Ipopt only)   — exists: _cache.attach, model.py:30
WARM  record replayed into the running daemon               — NEW
MISS  replay + boot Julia in this process                   — exists: _cache.materialize, model.py:47
```

**Client side.** When a daemon is reachable, `Core(...)` constructs a
`RecordingCore` even without `cache=`; at `Model(core)` the record is sent
and a `DaemonModel` returned (surface mirrors `CachedModel`). When no daemon
is reachable, `Core(...)` is eager exactly as today — the script cannot tell.

**Daemon side.** One process, owning one juliacall runtime and the device.
For each request: look up a live instance by `(fingerprint, data_digest)`;
on miss, replay the record through the existing eager path (the same walk
`materialize` does) and keep the instance; push the request's parameter
values into the instance (`ExaModels.set_value!` — the same refresh the
cache-hit path performs); solve; return arrays.

Because parameter values always ride along with the solve request, the
daemon needs no per-client mutable state in phase 1, and a client
`set_parameters(...)` is a local write — correct by construction.

**Instance reuse is the real win.** Replay+build of an unseen OPF case in a
warm process costs 7.4 s; a repeated solve of a live instance costs 0.7 s.
So: phase 1 (always replay) turns 308 s into ~8 s; phase 2 (live instances)
turns the parameter-sweep case into sub-second solves.

## Protocol, v1

- **Transport:** Unix domain socket, `$XDG_RUNTIME_DIR/examodels/daemon.sock`
  (fallback `/tmp/examodels-$UID/daemon.sock`, directory `0700`, socket
  `0600`). Same-user, same-machine only. Override / disable:
  `EXAMODELS_DAEMON=<path>` / `EXAMODELS_DAEMON=0`.
- **Framing:** length-prefixed frames; payload pickle protocol 5 with
  out-of-band numpy buffers. Pickle is acceptable exactly because the socket
  is a same-user boundary — the daemon runs as the user, with the user's own
  code. This is also why TCP transport is *deferred*: unpickling from a
  network peer is remote code execution, so remote needs a different wire
  format (see phase 4).
- **Handshake:** `{proto, examodels.__version__, ExaModels.jl pin (from
  juliapkg.json), python major.minor}`. Any mismatch: the daemon refuses,
  the client prints one info line and falls through to MISS. Skew can never
  produce wrong numbers, only a slow run.
- **Requests:** `HELLO`, `SOLVE{record, digests, param_values, solver,
  options}`, `STATUS`, `SHUTDOWN`. Phase 2 adds `RELEASE` (handle
  refcounting). Julia-side errors are caught in the daemon, translated by
  the existing `_bridge.translate` mapping, and re-raised client-side as the
  same exception types the eager path raises.
- **Concurrency:** the daemon accepts many connections but a single worker
  thread owns Julia; solves queue FIFO. (juliacall is not usefully
  re-entrant across Python threads, and the device is serialized by the
  solver anyway.)

## The invariant, and graceful degradation

**With no daemon reachable, behavior is byte-identical to today.** The
existing test suite, unmodified, must pass with dispatch code merged — that
is itself the acceptance test. Additional degradations that must be silent
and correct:

- Daemon dies between record and solve → client falls back to MISS
  (in-process replay), same answer.
- The script uses something a record cannot carry — today that is the
  nonlinear oracle with Python callbacks — → the partial record is replayed
  into an in-process eager core on the spot and construction continues
  eagerly. The run simply does not use the daemon. (Replay of a partial
  record is the same walk `materialize` already does on a full one.)
- `Core(cache=...)` models: HIT still wins over WARM (a local `.so` load
  beats IPC). On a shared-cache MISS the daemon replays — and since the
  daemon-side replay *is* `materialize`, it compiles and stores the library
  as a side effect, so the *next* run hits without any daemon at all.

## Pre-work surfaced by the measurements

Two small changes land before any daemon code, each with its own test:

1. **`set_parameters` is broken on device models** — reproduced twice:
   `ModelError: Scalar indexing is disallowed`. Mechanism (read, not yet
   fixed): `model.py:122` passes numpy through juliacall as a `PyArray`;
   upstream `set_value!` (`ExaModels/src/nlp.jl:819`) does `copyto!` into a
   `view` of the device vector, and view-of-CuArray + PyArray falls to
   generic elementwise copy, which GPUArrays forbids. Fix: stage the values
   into a device buffer first — the placement idiom already exists at
   `_bridge.py:240-245` — making the upstream `copyto!` device-to-device.
   CUDA-marked regression test that fails without the fix. This is a
   prerequisite for phase 2 (the daemon refreshes live instances with
   `set_value!`) and a user-facing bug today regardless.
2. **Recording refuses accelerator backends at construction**
   (`_record.py:283` — "a cached model compiles to a CPU shared library").
   Correct guard, wrong layer: it is the *cache consumer* that is CPU-only,
   not the recorder. Move the refusal from `RecordingCore.__init__` to the
   cache resolve, so a record can carry `backend="cuda"` for the daemon
   while `cache=` + accelerator still refuses with the same message.

One observation filed but *not* gating: eager-path expression tracing at
N=100 000 took 46 s (scenario 2). Surprising, uninvestigated, orthogonal.

## Phases, each with its acceptance test

- **P1 — transport + replay-per-solve.** `examodels` (foreground, C-c
  stops), `examodels status|stop`; dispatch; fallbacks. Accept: unmodified
  suite green with no daemon; with a daemon, LV and OPF fixtures return the
  same status/objective/x as without; second run of `case1354` completes in
  under 30 s wall (vs 308 s cold); killing the daemon mid-run still yields a
  correct solve. Shippable alone.
- **P2 — live instances.** LRU by `(fingerprint, data_digest)`; parameter
  values pushed per request. Accept: 50 parameter vectors on `case1354`
  solve in wall time ≈ 50× warm-solve (not 50× replay); daemon reports one
  replay and flat device memory after the first.
- **P3 — memory + lifetime.** Byte-budget LRU over *device bytes* (instances
  differ by orders of magnitude, so budget bytes, not counts; measure by
  device-memory delta at build), `CUDA.reclaim()` after eviction, in-flight
  and handle-held instances pinned. Julia never unloads compiled code, so
  distinct expression types grow the process monotonically — that is a
  documented restart-on-idle policy (`--idle-exit`, `status` shows RSS and
  device bytes), not a leak to chase. Accept: building past the budget
  evicts, device memory returns under budget, an evicted case still solves
  correctly (rebuilds).
- **P4 — interrupts, then remote.** Interrupting a running MadNLP solve is
  unsolved upstream; v1 documents that a client C-c abandons the request
  (the daemon finishes and discards). Remote (`tcp://gpu-node:5555`) only
  after the wire format stops being pickle; the model-not-script design is
  what makes it possible at all.

## Risks

- **GPU OOM can poison the CUDA context.** Policy: on OOM, evict all + `CUDA.reclaim()` +
  retry once; if the retry fails, exit loudly rather than serve a broken
  device. A supervised daemon that died is a cold start, not a wrong answer.
- **Fingerprint coarseness** would reuse wrong derivative code; the daemon
  widens the blast radius from one run to a long-lived process. The
  fingerprint is designed at least as fine as the backend's expression type
  (`_record.py` docstring) and is sha256 — the risk is a *bug* in
  canonicalization, not collision. Mitigation: the daemon revalidates
  `(fingerprint, data_digest)` against the record it is handed, and `status`
  names the cases it holds.
- **Version skew** — handled structurally by the handshake (refuse → fall
  back), never by trying to be compatible.
- **A queue behind one long solve.** FIFO and honest: `status` shows the
  queue; a second `examodels` on another socket is the escape hatch.

## Open questions for review

1. Dispatch default: opt-out as designed (daemon reachable ⇒ used), or
   opt-in (`EXAMODELS_DAEMON=1` required) for the first release?
2. Should `examodels` with no daemon *running* also be the way users start
   one in the background (`examodels --detach`), or is foreground-only +
   tmux the intended workflow?
3. Device-memory budget default (proposal: 60% of the device).
4. Is a per-device daemon (one socket per GPU) wanted eventually, or does
   `CUDA_VISIBLE_DEVICES` at daemon start settle it?
