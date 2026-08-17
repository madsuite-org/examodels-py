# The model cache — no Julia after the first run

[The first call costs seconds](tutorial/performance.md) because the backend
compiles, and the price is per *process*. That suits a long Julia session and
punishes the way Python is actually used: scripts re-run from scratch, so
every run pays the startup and the kernels again.

`Core(cache=True)` removes the per-run price. The first run of a script
compiles the model ahead of time into a shared library and stores it, keyed
by the model's content; every later run of the unchanged script loads the
library and solves — **with no Julia in the process at all**.

Measured on one machine (the same script, run repeatedly):

| | |
|---|---|
| first run ever (1 000 variables): build + compile + store | **90 s** |
| every later run: load + solve, no Julia | **0.35 s** |
| a second solve inside one process | **0.04 s** |

## Recording instead of building

With `cache=`, a `Core` records the model in pure Python — expressions become
small trees, data stays in numpy, and Julia is not started:

```python
import examodels as exa

core = exa.Core(cache=True)
x = core.add_var(1000, start=0.5)
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
             over=range(1, 1000))

fingerprint, data_digest = core.fingerprint()
```

The pair of digests is the cache key. The **fingerprint** covers what
determines the generated code — the expressions, operators, block shapes and
names, index-set kinds. The **data digest** covers the values the library
bakes in: starts, bounds, index-set contents. Change either and the next
`Model(core)` compiles a fresh entry; change neither and it loads the stored
one.

## The two runs

`Model(core)` does the rest. On a **miss** it replays the record through the
ordinary eager path — Julia starts, exactly as without `cache=` — then
compiles the model with [the compiler backend](install.md) and stores the
library before returning, so the run is slower by the compile time, once. On
a **hit** it loads the library through
[cnlpmodels](https://github.com/madsuite-org/cnlpmodels-py) and the model
solves with Ipopt through cyipopt:

<!-- not-tested: the first run compiles, which needs the compiler backend and Julia 1.12 -->
```python
model = exa.Model(core)      # run 1: Julia + one compile; run 2+: julia-free
sol = model.solve()          # Ipopt via cyipopt on a hit
print(sol.status, sol.objective, sol[x])
```

Both runs present the same surface: named blocks, `sol[x]`, multipliers,
`parameters` / `set_parameters`. Two solver caveats on a hit: `solver=` must
be Ipopt (MadNLP would need Julia, which a hit never boots — build without
`cache=` to use it), and starts and bounds are baked into the library (they
are data, so changing them is a new entry; their setters say so).

## Parameters stay live

Parameter *values* are deliberately outside both digests: the ABI keeps them
settable on a loaded library. A script whose data lives in `add_par` blocks
hits the same entry at any values:

```python
core = exa.Core(cache=True)
x = core.add_var(4)
p = core.add_par([1.0, 2.0, 3.0, 4.0])       # change these freely: still a hit
core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(4))
```

## Recipes: one entry, any instantiation

The stronger form. A [recipe](recipe.md) separates structure from data, and a
compiled recipe library takes its data per instance — so the entry is keyed
by the *types* of the instantiation values, never their contents. Compile at
one size, hit at every size:

<!-- not-tested: the first run compiles, which needs the compiler backend and Julia 1.12 -->
```python
core = exa.Core(nargs=2, cache=True)
n, x0 = core.args
x = core.add_var(n, start=x0)
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
             over=exa.srange(1, n))

model = exa.Model(core, 1000, [0.0] * 1000)   # first ever: compiles once
model = exa.Model(core, 5000, [0.5] * 5000)   # any later run, any size: a hit
```

An argument of a different *type* — `1000.0` where an integer was compiled,
an integer array where floats were — is a different entry. One corner: a
dims-only constraint block (`add_con(2, ...)`) cannot be addressed by handle
on a recipe hit, because the eager surface cannot name it into the library's
layout; read the whole vector instead, or build without `cache=`.

## What it needs, and how it degrades

Loading entries needs the `[cache]` extra (cnlpmodels and cyipopt); storing
them needs the [compiler backend](install.md), which needs Julia 1.12 and
therefore a Python linking OpenSSL ≥ 3.5:

```
pip install "examodels[cache] @ git+https://github.com/madsuite-org/examodels-py"
```

Nothing breaks where those are missing. Without the compiler, `Model(core)`
warns once and returns the ordinary eager model, storing nothing. In a
process where Julia is already running — you built an eager model first, or
rebuilt the same model right after a miss — a matching entry is served
eagerly too, without recompiling: a compiled library cannot start its runtime
beside a live one, so the cache steps aside rather than crash.

## Where entries live

`cache=True` stores under `$EXAMODELS_CACHE` (default `~/.cache/examodels`),
content-addressed. `cache="@name"` installs the library on `CNLPMODELS_PATH`
under that name, where any cnlp consumer finds it; `cache="/some/path"` uses
that directory as the entry. Each entry is a shared library plus a JSON
sidecar carrying the digests it answers to; deleting an entry's directory is
always safe and costs one recompile.
