# The model cache — no compilation overhead after the first run

[The first call costs seconds](tutorial/performance.md) because the backend
compiles, and the price is per *process*. That suits a long Julia session and
punishes the way Python is actually used: scripts re-run from scratch, so
every run pays the startup and the kernels again.

`Core(cache=True)` removes the per-run price. The first run of a script
compiles the model ahead of time into a shared library and stores it, keyed
by the model's content; every later run of the unchanged script loads the
library and solves — **nothing compiles, nothing warms up**. (The stored
library is self-contained, so those runs happen not to start Julia at all —
a mechanism, not the point: the point is that the overhead is gone.)

Measured on one machine (the same script, run repeatedly):

| | |
|---|---|
| first run ever (1 000 variables): build + compile + store | **90 s** |
| every later run: load + solve | **0.35 s** |
| a second solve inside one process | **0.04 s** |

## Write it as a recipe

The natural form for a cached model is a [recipe](recipe.md): the structure
is written against placeholders, so the data — sizes included — stays a
per-run input, and **one compiled entry serves every instantiation**. With
`cache=`, the core records the model in pure Python (expressions become
small trees, and Julia is not started):

```python
import examodels as exa

core = exa.Core(nargs=2, cache=True)
n, x0 = core.args
x = core.add_var(n, start=x0)
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
             over=exa.srange(1, n))
```

`Model` does the rest. On a **miss** it replays the record through the
ordinary eager path — Julia starts, exactly as without `cache=` — then
compiles the model with [the compiler backend](install.md) and stores the
library before returning, so that run is slower by the compile time, once.
Every later `Model` of the same structure is a **hit**: the library is
loaded through [cnlpmodels](https://github.com/madsuite-org/cnlpmodels-py),
instantiated with this run's values, and solved with Ipopt through cyipopt:

<!-- not-tested: the first run compiles, which needs the compiler backend and Julia 1.12 -->
```python
model = exa.Model(core, 1000, [0.0] * 1000)   # first ever: compiles once
sol = model.solve()
print(sol.status, sol.objective, sol[x])

model = exa.Model(core, 5000, [0.5] * 5000)   # any later run, ANY size: a hit
```

The entry is keyed by the *types* of the instantiation values, never their
contents — an integer size and a float vector here — so new sizes and new
data keep hitting. An argument of a different type (`1000.0` where an
integer was compiled, an integer array where floats were) is a different
entry. Both runs present the same surface: named blocks, `sol[x]`,
multipliers, `parameters` / `set_parameters`.

## What the key covers

`core.fingerprint()` returns the pair of digests behind all of this. The
**fingerprint** covers what determines the generated code — the expressions,
operators, block shapes and names, index-set kinds, placeholder-sized
dimensions. The **data digest** covers the values the library bakes in:
literal starts, bounds, index-set contents. Change either and the next
`Model` compiles a fresh entry; change neither and it loads the stored one.

Parameter *values* are deliberately outside both digests — the ABI keeps
them settable on a loaded library — so data routed through `add_par` blocks
changes freely without ever touching the key:

```python
core = exa.Core(cache=True)
x = core.add_var(4)
p = core.add_par([1.0, 2.0, 3.0, 4.0])       # change these freely: still a hit
core.add_obj(lambda i: (x[i] - p[i]) ** 2, over=range(4))
```

## Fixed models cache too

A core with no placeholders — plain `Core(cache=True)` — is the simpler
case: everything but parameter values is baked, so the entry answers to
exactly one model and any data change is one recompile. It is the right
form when the model genuinely is one instance; reach for the recipe form
the moment sizes or data vary between runs.

Two caveats that apply on any hit: `solve(solver=)` must be Ipopt (MadNLP
would need Julia, which a hit never starts — build without `cache=` to use
it), and starts and bounds are baked (they are data; their setters say so).
On a *recipe* hit there is one more: a dims-only constraint block
(`add_con(2, ...)`) cannot be addressed by handle, because the eager surface
cannot name it into the library's layout — read the whole vector instead.

## What it needs, and how it degrades

Loading entries needs the `[cache]` extra (cnlpmodels and cyipopt); storing
them needs the [compiler backend](install.md), which needs Julia 1.12 and
therefore a Python linking OpenSSL ≥ 3.5:

```
pip install "examodels[cache] @ git+https://github.com/madsuite-org/examodels-py"
```

Nothing breaks where those are missing. Without the compiler, `Model`
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
