# GPU

Solving on an NVIDIA device takes three backend packages: CUDA (the backend),
MadNLPGPU (the solver's device side) and CUDSS (the linear solver). Today
`install_backend("cuda")` installs only the first; the other two go through
`juliapkg` until the package grows a route of its own. Run this once and
**restart Python afterwards** — the environment cannot change under a running
Julia:

<!-- not-tested: installs backend packages; needs a CUDA device -->
```python
import examodels as exa
import juliapkg

exa.install_backend("cuda")
juliapkg.add("MadNLPGPU", "d72a61cc-809d-412f-99be-fd81f4b8a598")
juliapkg.add("CUDSS", "45b445bb-4962-46a0-9369-b4df9d0f772e")
juliapkg.resolve()
```

A device model is then ordinary code — MadNLP and a device linear solver are
chosen automatically. Measured on a Quadro GV100, the Lukšan–Vlček problem at
100 000 variables solves in **0.75 s** on the device against 18.7 s with the
same solver on the CPU and 5.6 s with Ipopt (all in a warm process):

<!-- not-tested: needs a CUDA device -->
```python
import examodels as exa

N = 100_000
core = exa.Core(backend="cuda")
x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
             over=range(1, N))
core.add_con(lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
             + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
             + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
             over=range(0, N - 2))

sol = exa.Model(core).solve()          # MadNLP + cuDSS, chosen automatically
print(sol.status, sol.objective)
```

`exa.backends()` lists what can be constructed: `serial`, `cpu` (threads), `cuda`, `rocm`,
`oneapi`, `metal`. Each is loaded only if asked for, so a host model never starts a GPU
runtime.

:::{admonition} Older GPUs need a pinned CUDA backend
:class: caution

A fresh resolve takes the newest CUDA.jl, whose bundled CUDA 12.9 runtime
does not fully support compute-capability-7.0 (Volta) devices — the symptom
is a warning at load and `EXIT: Internal Error` from MadNLP. On such a card,
pin the backend before resolving:
`juliapkg.add("CUDA", "052768ef-5323-5732-b1bb-66c8b64840ba", version="=6.2.0")`.
:::

## What runs on a device

Ordinary models, multi-dimensional blocks, product index sets, tables, constraint
augmentation, parameters and two-stage models all work on a device. Evaluation buffers and
setter values are placed where the model's arrays live, so nothing about the API changes.

The one thing that cannot is a **nonlinear oracle with Python callbacks**: a Python
function cannot run inside a device kernel. Such an oracle runs with `adapt=True`, which
copies to the host for each call — correct, but a round trip.

## Sharing memory with CuPy

See [CuPy interchange](../cupy.md): device arrays can be passed in either direction with no
host round-trip.

## A known upstream limitation

`CompressedNLPModel` divides by the constraint count, so it raises on a model with no
constraints. Nothing can be done about that from this side; it is recorded as a test so a
fix upstream shows up.
