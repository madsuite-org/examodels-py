# Recipes and compiled libraries

A `Core` is normally built from data you already have, so the model and its data
are finished together. Ask for placeholders instead and the two come apart:

```python
import examodels as exa

core, N, x0 = exa.recipe(nargs=2)          # or: exa.Core(nargs=2), then core.args
x = core.add_var(N, start=x0)
core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
             over=exa.srange(1, N))

exa.Model(core, 10, [-1.2] * 10)           # one value per placeholder, in order
exa.Model(core, 10_000, [-1.2] * 10_000)   # the same core, again
```

A placeholder is used *as the value it stands for* — `N` is the number of
variables, not a namespace to reach into — and arithmetic on it is deferred, so
`N - 1` and `srange(1, N)` describe sizes computed when the model is built.
`srange` is half-open like `range`, and exists only because `range` demands
integers the moment a bound is symbolic.

Anything needing a real value must be computed before building and passed in:
`len(N)`, `int(N)`, `if N > 3` and iteration all raise, naming the fix. A
placeholder that quietly satisfied `__index__` would give a model whose shape
depended on a value nobody supplied, and the failure would surface far from its
cause.

:::{admonition} A placeholder-sized block cannot be read back by handle
:class: caution

`sol[x]` and `model.get_start(x)` need a block whose size was known when the
core was written, so on a recipe they refuse and say so. Read the whole vector
instead — `sol.x`, `model.x0`, `model.lvar` — and slice it. ExaModels.jl has
the same limit.
:::

## Compiling

Ahead-of-time compilation needs the *structure* to become code while the *data*
stays a run-time input, which is exactly what a recipe separates:

<!-- not-tested: compiling needs the compiler backend and Julia 1.12 -->
```python
exa.install_compiler()                     # once per environment; see Installation
lib = exa.compile_library("@rosenbrock", core, 10)    # 10: an example size
```

The example value's *type* is baked in — the compiler needs the call graph
resolved statically — while its value is supplied per instance at run time.
(You rarely need to call this yourself: `Core(nargs=..., cache=True)` compiles
and reuses the library automatically — see [](cache.md).)

`"@name"` installs the library on the `CNLPMODELS_PATH` search path, where both
consumers find it by that name; anything else is an ordinary path.

A core with **no** placeholders is a fixed model, compiled with no example at
all, and several models can share one library:

<!-- not-tested: compiling needs the compiler backend and Julia 1.12 -->
```python
lib = exa.compile_library("@grid", {"acopf": (ac_core, 100),
                                    "dcopf": (dc_core, 100),
                                    "small": fixed_core})
lib.prefixes                               # ('acopf', 'dcopf', 'small')
```

`bundle=True` carries a privatized copy of the Julia runtime, so the library
needs no Julia at the far end — and it is the only form Julia itself can load,
since a library sharing the host's `libjulia` aborts on its first call. The
default emits a single small library instead.

## Using the result

The library exposes the model through a plain C interface, so who loads it is
your business. Neither consumer is a dependency of this package:

<!-- not-tested: loading a compiled library needs one to have been compiled -->
```python
import cnlpmodels                          # ctypes + numpy, no Julia
m = cnlpmodels.CModel("@rosenbrock", 10)
m.nvar, m.obj(m.x0), m.grad(m.x0)
```

```julia
using CNLPModels                           # from Julia
m = CNLPModel("@rosenbrock", 10)
```

## Giving the data to the library instead

The examples above hand the model its data at instantiation, which crosses the
C boundary: scalars, arrays, and tables of them all do. That is the Python-native
route and needs no Julia at all, since your data is already in Python.

The alternative is to have the library *carry* the data processing, so that it
is handed one string — a case file, a dataset name — and derives the rest
itself. That is what an argument function is for:

<!-- not-tested: needs the compiler backend and ExaPowerIO -->
```python
exa.compile_library("@grid", core, "case14.m", argfun="ExaPowerIO.parse_case")
```

`argfun` names a function an installed Julia package already has; the compiler
resolves it by name out of that package, which is why it cannot be a Python
function, a lambda, or anything built at run time. If your data processing is
in Python, pass the data as example values as above — that route needs no Julia
and costs a Python caller nothing, since the data is already on that side.
