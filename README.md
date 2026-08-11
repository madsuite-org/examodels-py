# examodels

[![test](https://github.com/MadNLP/examodels-py/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/MadNLP/examodels-py/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/MadNLP/examodels-py/branch/main/graph/badge.svg)](https://codecov.io/gh/MadNLP/examodels-py)

Python interface to [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) — SIMD-parallel
algebraic modeling and automatic differentiation for nonlinear programs, on CPU threads or GPUs.

```python
import examodels as exa

N = 10
core = exa.Core()
x = exa.add_var(core, N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])

core.add_obj(100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2 for i in range(1, N))

core.add_con((3 * x[i+1]**3 + 2 * x[i+2] - 5
              + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
              + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3
              for i in range(0, N - 2)), lcon=0.0, ucon=0.0)

model = exa.Model(core)
sol = model.solve(solver="ipopt")
print(sol.status, sol.objective, sol[x])
```

A `Core` accumulates the model and a `Model` is built from it, mirroring the backend's
own two-stage construction.

```
first_order 6.2324586324 [-0.95055636  0.91390082  0.98909052 ... 0.99999993]
```

A generator expression carries both the body and the index set, so it reads the way
the equivalent model reads in ExaModels.jl. The same thing can be written as a
function when that is clearer — they produce byte-identical models:

```python
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2, over=range(1, N))
```

## How it works

Each expression is an ordinary Python function of an index (a generator expression is
one too — its body is re-invoked, never iterated). It is called **once**, with a
symbolic index, and the operators it applies build one structured expression that describes
every row of the objective or constraint block. The loop never runs at model-build time,
which is what lets the derivatives be evaluated as a single parallel kernel over the whole
index set.

Two consequences worth knowing:

- **Expressions must not branch on the index.** `x[i] if i > 3 else x[i-1]` raises a
  `TypeError`, because at trace time `i` has no value. Anything index-dependent belongs in
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

## Recipes: one model, any size — and a shared library

A `Core` is normally built from data you already have, so the model and its data
are finished together. Ask for placeholders instead and the two come apart:

```python
core, N, x0 = exa.recipe(nargs=2)          # or: exa.Core(nargs=2), then core.args
x = core.add_var(N, start=x0)
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
             over=exa.srange(1, N))

exa.Model(core, 10, [-1.2] * 10)           # one value per placeholder, in order
exa.Model(core, 10_000, [-1.2] * 10_000)   # the same core, again
```

A placeholder is used *as the value it stands for* — `N` is the number of
variables, not a namespace to reach into — and arithmetic on it is deferred, so
`N - 1` and `srange(1, N)` describe sizes that are computed when the model is
built. `srange` is half-open like `range`, and exists only because `range`
demands integers the moment a bound is symbolic.

Anything that needs a real value has to be computed before building and passed
in: `len(N)`, `int(N)`, `if N > 3` and iteration all raise, naming the fix. A
placeholder that quietly satisfied `__index__` would give you a model whose
shape depended on a value nobody had supplied, and the failure would surface far
from its cause.

### Why it exists

Ahead-of-time compilation needs the *structure* to become code while the *data*
stays a run-time input. Writing a model against placeholders is that separation,
so a recipe that builds and instantiates can be compiled — you never have to
work out which modelling constructs survive trimming:

```python
exa.install_compiler()                     # once per environment
lib = exa.compile_library(core, "rosenrock", arg=10)
```

The result is a shared library behind a plain C interface, which
[cnlpmodels](https://github.com/MadNLP/cnlpmodels-py) loads with ctypes and
numpy — no Julia in the process — and
[CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl) loads from Julia.
Given a bare name rather than a path, the library is installed on the
`CNLPMODELS_PATH` search path, where both find it by that name.

Compiling needs a backend able to run it, which today means **Julia 1.12**. The
Julia version is chosen by `juliapkg`, which caps it at 1.11 when the Python
process links OpenSSL older than 3.5 — Julia 1.12 requires 3.5. A system Python
on an older OpenSSL therefore gets recipes but not compilation; a conda-forge
Python (or any build on OpenSSL >= 3.5) gets both, with nothing to configure.

## Parameters and subexpressions

```python
th = core.add_par([100.0, 1.0])
core.add_obj(lambda i: th[0] * (x[i-1]**2 - x[i])**2 + (x[i-1] - th[1])**2, over=range(1, N))

model = exa.Model(core)
model.set_value(th, [200.0, 1.0])    # re-solve without rebuilding
```

```python
s = core.add_expr(lambda i: y[i]**2, over=range(N))
core.add_obj(lambda i: (s[i] - 1)**2, over=range(N))
core.add_con(lambda i: s[i] + s[i+1], over=range(N - 1), lcon=0.0)
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
| `.add_var(n, start=, lvar=, uvar=)` | a block of variables; index it with `[i]` |
| `.add_var(T, N, ...)` | a multi-dimensional block; index it with `[t, i]` |
| `.add_obj(f, over=)` | add `sum(f(i) for i in over)` to the objective |
| `.add_con(f, over=, lcon=, ucon=)` | one row per index, `lower <= f(i) <= upper` |
| `.solve(solver=)` | build and solve in one step |
| `Model` | `.nvar` `.ncon` `.nnzj` `.nnzh` `.x0` `.objective(x)` `.gradient(x)` `.constraints(x)` |
| `Solution` | `.status` `.objective` `.iterations` `.x` `.y` `.elapsed` `.success`, and `sol[x]` |

Everything crossing the boundary is a Python scalar, a `range`, or a numpy array.

## Several dimensions

```python
x = core.add_var(T, N, lvar=np.zeros((T, N)))

Cell = namedtuple("Cell", "t i")
grid = [Cell(t, i) for t in range(1, T) for i in range(N)]
core.add_con(x[c.t, c.i] - x[c.t - 1, c.i] for c in grid)
```

Bounds and starting points are given, and read back, in the shape you passed —
the backend's own column-major storage never surfaces.

## Indexing over data

An index set can be a table of rows rather than a range, with the fields available
on the traced row. A row can be a named tuple, a dataclass, a class with
`__slots__`, or a row of a numpy structured array — anything with named fields. A
pandas frame converts with `df.to_records(index=False)`. Nothing needs wrapping:

```python
Gen = namedtuple("Gen", "i bus cost1")
gen = [Gen(0, 3, 1100.0), ...]

exa.add_obj(core, lambda g: g.cost1 * pg[g.i]**2, over=gen)

balance = exa.add_con(core, lambda b: b.pd + b.gs * vm[b.i]**2, over=bus)
exa.add_con(core, balance, lambda a: (a.bus, p[a.i]), over=arc)   # terms into rows
```

Whole-numbered columns stay integers, so a field can be used as a variable index;
everything else becomes a float.

`examples/ac_opf.py` builds AC optimal power flow this way and solves the PGLib
benchmark cases; `examples/matpower.py` reads the `.m` files, so the example needs
nothing outside this package.

## Time to first solve

Everything expensive is per *process*, not per model. A first 1000-variable solve
costs roughly 25 s; the same model built again in the same process costs **0.08 s**,
and a model with a different expression costs **1.3 s**.

About half of that is fixed start-up (runtime, packages, solver code) and about half
is compiling *your* model: the backend encodes an expression — and the model built so
far — in a type, so each distinct shape compiles its own derivative kernels. That part
cannot be precompiled, since the type does not exist until your function runs, and it
is also the reason evaluation is fast.

So keep the process alive — a session, a notebook, or a worker — rather than paying it
per script. (A PackageCompiler system image was measured and is not recommended: it
saves about 13 %, and a custom image cannot load the GPU backends at all.)

## Coverage of the backend's interface

`tests/test_parity.py` reads the backend's export list **at run time** and requires
every name to be classified, so anything added upstream fails the suite until it is
accounted for. Today, of 71 exported names:

| | | |
|---|---|---|
| 31 | reachable | `Core`, `Model`, `add_var/par/obj/con/expr`, `add_con!`, `Constant`, `SumNode`/`ProdNode`, `solution`, `multipliers`(`_L`/`_U`), and all twelve `get_*`/`set_*` |
| 6 | different spelling | the `@add_*` macros — written here as generator expressions |
| 7 | out of scope | the backend's own deprecated API (`variable`, `objective`, …) |
| 2 | out of scope | `@register_univariate`/`@register_bivariate` — defining an operator needs Julia |
| 24 | **not exposed yet** | nonlinear oracles, two-stage models, tags, NLPModel wrappers |

So **37 of the 61 in-scope exports** are reachable, and a single test builds a model
that exercises every one of them.

## Tests

```
pip install -e ".[test]"
pytest
```

The suite checks, among other things, that a Python-written expression produces *exactly* the
same expression the backend builds for the equivalent native model — compared by structural
identity, not by printed form.
