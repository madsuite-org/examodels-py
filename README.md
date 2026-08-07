# examodels

Python interface to [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) — SIMD-parallel
algebraic modeling and automatic differentiation for nonlinear programs, on CPU threads or GPUs.

```python
import examodels as exa

N = 10
core = exa.Core()
x = core.add_variables(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])

core.minimize(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
              over=range(1, N))

core.constrain(lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
               + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
               + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
               over=range(0, N - 2), lower=0.0, upper=0.0)

model = exa.Model(core)
sol = model.solve(solver="ipopt")
print(sol.status, sol.objective, sol[x])
```

A `Core` accumulates the model and a `Model` is built from it, mirroring the backend's
own two-stage construction.

```
first_order 6.2324586324 [-0.95055636  0.91390082  0.98909052 ... 0.99999993]
```

## How it works

Each expression is an ordinary Python function of an index. It is called **once**, with a
symbolic index, and the operators it applies build one structured expression that describes
every row of the objective or constraint block. The loop never runs at model-build time,
which is what lets the derivatives be evaluated as a single parallel kernel over the whole
index set.

Two consequences worth knowing:

- **Expressions must not branch on the index.** `lambda i: x[i] if i > 3 else x[i-1]` raises
  a `TypeError`, because at trace time `i` has no value. Anything index-dependent belongs in
  the data — `start`, `lower`, `upper`, or the index set itself, all of which are evaluated
  per index in the ordinary way.
- **Only registered operators may appear.** Use `exa.sin`, not `math.sin`. The available
  functions are generated from the operator list the backend registers, so `dir(exa)` is the
  authoritative list (52 univariate, 9 bivariate at the time of writing).

Indices are 0-based, like the rest of Python: `x[0]` is the first variable and
`over=range(n)` means `n` rows.

## Install

```
pip install examodels
```

**You do not need Julia installed.** The backend runtime is downloaded into the
environment on first use (verified from a shell with no `julia` on `PATH`; it costs
about 260 MB and one resolve). `import examodels` does not start it —
the runtime boots the first time you build a model, so importing the package stays instant.

Solvers are backend packages rather than Python ones, so they are installed through
this package (once, per environment):

```python
import examodels as exa
exa.install_solver("ipopt")      # CPU
exa.install_solver("madnlp")     # CPU or GPU
exa.available_solvers()          # ['ipopt', 'madnlp']
```

## Parameters and subexpressions

```python
th = core.add_parameters([100.0, 1.0])
core.minimize(lambda i: th[0] * (x[i-1]**2 - x[i])**2 + (x[i-1] - th[1])**2, over=range(1, N))

model = exa.Model(core)
model.set_parameters(th, [200.0, 1.0])    # re-solve without rebuilding
```

```python
s = core.add_expression(lambda i: y[i]**2, over=range(N))
core.minimize(lambda i: (s[i] - 1)**2, over=range(N))
core.constrain(lambda i: s[i] + s[i+1], over=range(N - 1), lower=0.0)
```

Subexpressions are inlined at each use — they add no variables and no constraint rows.

## GPU

```python
exa.install_backend("cuda")               # once
core = exa.Core(backend="cuda")
...
sol = exa.Model(core).solve(solver="madnlp")
```

`exa.backends()` lists what can be constructed: `serial`, `cpu` (threads), `cuda`,
`rocm`, `oneapi`, `metal`. Each is loaded only if it is asked for, so a CPU model never
starts a GPU runtime. On a device model a device-capable linear solver is selected
automatically.

Note if you also use CuPy in the same process: it works, but CuPy's pip-installed CUDA
libraries can shadow the backend's own, which the backend will warn about. Keeping
`site-packages/nvidia/*/lib` off `LD_LIBRARY_PATH` avoids it. There is no zero-copy
handoff between CuPy arrays and device model arrays yet.

## API

| | |
|---|---|
| `Model()` | start building |
| `.add_variables(n, start=, lower=, upper=)` | a block of variables; index it with `[i]` |
| `.minimize(f, over=)` | add `sum(f(i) for i in over)` to the objective |
| `.constrain(f, over=, lower=, upper=)` | one row per index, `lower <= f(i) <= upper` |
| `.solve(solver=)` | build and solve in one step |
| `Model` | `.nvar` `.ncon` `.nnzj` `.nnzh` `.x0` `.objective(x)` `.gradient(x)` `.constraints(x)` |
| `Solution` | `.status` `.objective` `.iterations` `.x` `.y` `.elapsed` `.success`, and `sol[x]` |

Everything crossing the boundary is a Python scalar, a `range`, or a numpy array.

## Time to first solve

Everything expensive is per *process*, not per model. On one machine, a first
1000-variable solve costs about 26 s; the same model built again in the same
process costs **0.08 s**, and a model with a different expression costs **1.3 s**.

The cost divides in two:

- **Fixed per process** (~9 s): starting the runtime, loading packages, compiling
  the solver's generic code.
- **Per model *shape*** (~9 s): the backend encodes an expression — and the model
  built so far — in a Julia *type*, so each distinct shape compiles its own
  derivative kernels. This cannot be precompiled: the type does not exist until
  your function runs. It is also the reason the evaluations are fast.

So the effective fix is to keep the process alive — a session, a notebook, or a
worker — rather than to pay it per script.

If you do need a faster cold start, a prebuilt system image removes most of the
fixed part:

```python
import examodels as exa
exa.sysimage.build()      # ~10 minutes, ~340 MB, once per environment
```

Every later process picks it up automatically. Measured, it takes a first solve
from 26.6 s to 23.2 s — the compile savings are real (first solve 5.6 s -> 1.6 s)
but a large image also costs ~4 s more to load, so the net is modest. Worth it for
a fixed deployment; probably not for a laptop. It must be rebuilt when the backend
or solvers change.

## Tests

```
pip install -e ".[test]"
pytest
```

The suite checks, among other things, that a Python-written expression produces *exactly* the
same expression the backend builds for the equivalent native model — compared by structural
identity, not by printed form.
