# The first call, and every call after it

The backend compiles. The first model you build in a process starts the Julia runtime and
compiles the derivative kernels for your model's shape, which takes seconds; every model
after it, in that same process, costs a fraction of a second. Coming from Python this is
surprising, because nothing else in Python behaves that way — so it is worth seeing the
numbers before deciding it is slow.

Measured on one machine, one process, in order:

| | |
|---|---|
| build a 10-variable model, first ever | **18.96 s** |
| solve it, first ever | **5.40 s** |
| build the same model again | **0.00 s** |
| solve it again | **0.05 s** |
| build the same structure at 1 000 variables | **0.00 s** |
| solve that | **0.10 s** |
| build the same structure at 100 000 variables | **0.05 s** |

The first build and solve together cost about 25 s. The second cost 0.05 s — some 500
times faster — and that is the number that describes the rest of your session.

All of this is per process. If your workflow is re-running a script rather
than one long session, [the model cache](../cache.md) moves the whole cost to
a single ahead-of-time compile: after it, each run of the script takes a
fraction of a second and never starts Julia at all.

## The cost is per *structure*, not per instance

The last three rows are the useful part. A model 10 000 times larger, built from the same
expressions, costs no compilation at all: the backend encodes an expression in a *type*,
so what compiles is the shape of your model, and the size is data. Growing a problem is
therefore free in the sense that matters here, while writing a *different* expression is
not.

You can watch that directly, in a session that is already warm:

```python
import time

import examodels as exa


def rosenbrock(n):
    core = exa.Core()
    x = core.add_var(n, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)])
    core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2, over=range(1, n))
    return core


exa.Model(rosenbrock(10))                     # pay the compilation once

for n in (10, 1_000, 100_000):
    t0 = time.perf_counter()
    model = exa.Model(rosenbrock(n))
    print(f"{n:>7} variables: {time.perf_counter() - t0:.3f} s")
```

Model construction scales with the number of distinct *expressions*, not with the number
of rows: a constraint over a million-row index set costs the same to build as one over ten
rows, and only the data array differs. That is the property worth designing a model
around.

To see the first call itself you need a *fresh* process, since it happens once and only
once:

<!-- not-tested: the effect is only visible in a cold process, and the docs suite is warm by the time it reaches this page -->
```python
import subprocess, sys, textwrap

print(subprocess.run([sys.executable, "-c", textwrap.dedent("""
    import time
    import examodels as exa
    t0 = time.perf_counter()
    core = exa.Core()
    x = core.add_var(10, start=1.0)
    core.add_obj(lambda i: (x[i] - 2.0)**2, over=range(10))
    exa.Model(core)
    print(f"first build in a cold process: {time.perf_counter() - t0:.1f} s")
""")], capture_output=True, text=True).stdout)
```

## What follows for how you work

**Keep the process alive.** A REPL, a notebook, or a worker that stays up pays the cost
once and then behaves like ordinary Python; a script that runs and exits pays it every
time. The practical workflow is to keep a session open and re-run your model code into it
as you edit — the same habit Julia users have, for the same reason.

The cost divides in two, which is why:

- **Fixed per process** (~9 s): starting the runtime, loading packages, compiling the
  solver's generic code.
- **Per model shape** (~9 s here): each distinct expression compiles its own derivative
  kernels. This cannot be precompiled — the type does not exist until your function runs —
  and it is the same mechanism that makes evaluation fast afterwards.

A PackageCompiler system image was measured and is **not** recommended: it saves about
13%, and a custom image cannot load the GPU backends at all.

**When a cold start is not acceptable, compile the model instead.** A recipe can be built
into a shared library that carries its derivative kernels with it, so the consumer pays
nothing to load it and needs no Julia at all — see [](../recipe.md). That is the route for
shipping a model to someone else, or into a service, rather than for your own development
loop.
