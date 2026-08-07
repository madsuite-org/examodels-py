# GPU

```python
exa.install_backend("cuda")            # once per environment

core = exa.Core(backend="cuda")
...
sol = exa.Model(core).solve()          # MadNLP, chosen automatically
```

`exa.backends()` lists what can be constructed: `serial`, `cpu` (threads), `cuda`, `rocm`,
`oneapi`, `metal`. Each is loaded only if asked for, so a host model never starts a GPU
runtime.

MadNLP is the only solver that handles device arrays, so a device model is sent to it
without being asked. A device-capable linear solver is selected in the same way.

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
