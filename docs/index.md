# examodels

:::{admonition} Experimental
:class: warning

**This package is experimental and is not yet stable.** It is under active
development: the API may change without notice, releases are unversioned in
practice, and the compiler interface in particular is still moving. It has not
been through the round of real use that would justify calling it otherwise.

Use it for research and experiments, expect to pin a commit if you depend on
it, and please report what breaks.
:::

Python interface to [ExaModels.jl](https://github.com/madsuite-org/ExaModels.jl) — SIMD-parallel
algebraic modelling and automatic differentiation for nonlinear programs, on CPU threads or
GPUs.

```python
import examodels as exa

N = 10
core = exa.Core()
x = exa.add_var(core, N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])

exa.add_obj(core, lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
            over=range(1, N))

exa.add_con(core, lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
            + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
            + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
            over=range(0, N - 2))

sol = exa.Model(core).solve()
print(sol.status, sol.objective, sol[x])
```

You do not need Julia installed: the backend runtime is downloaded into the environment
on first use.

## Contents

```{toctree}
:maxdepth: 2

install
simd
tutorial/index
recipe
cache
cupy
api
```

## Relation to the Julia manual

The pages here mirror the ExaModels.jl manual. Three of its pages have no counterpart,
for reasons rather than by omission:

- **JuMP interface** — JuMP is a Julia modelling language; the corresponding Python entry
  points are the ones documented here.
- **Developing solvers / Upgrading** — both concern writing Julia against the backend.
- **Quadrotor and Distillation examples** — not yet ported; the constructs they use
  (multi-dimensional blocks, subexpressions, product index sets) are all supported and
  covered in the tutorial.
