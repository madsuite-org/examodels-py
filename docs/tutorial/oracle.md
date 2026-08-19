# Oracle constraints

An oracle is a block of constraints, or an objective term, that you evaluate yourself —
useful when the residual comes from a simulation, an external solver, or code that is not
expressible as an algebraic expression.

```python
import madsuite as exa

core = exa.Core()
x = exa.add_var(core, 2, start=0.5)


def f(c, x):
    c[0] = x[0]**2 + x[1]**2 - 1.0

def jac(v, x):
    v[0], v[1] = 2 * x[0], 2 * x[1]

def hess(v, x, y):
    v[0] = v[1] = 2 * y[0]

oracle = exa.VectorNonlinearOracle(
    nvar=2, ncon=1, f=f, jac=jac, hess=hess,
    jac_rows=[1, 1], jac_cols=[1, 2],
    hess_rows=[1, 2], hess_cols=[1, 2],
    lcon=[0.0], ucon=[0.0])

exa.add_con(core, oracle)
```

Sparsity patterns are declared once, with 1-based indices, and never change.

Instead of explicit derivatives you may supply matrix-free products — `jvp(Jv, x, v)`,
`vjp(Jtv, x, w)` and `hvp(Hv, x, w, v)`. `has_matfree_jac` and `has_matfree_hess` report
which path an oracle uses.

An objective term works the same way:

```python
o = exa.ScalarNonlinearOracle(
    nvar=2,
    f=lambda x: float(x[0]**2 + x[1]**2),
    grad=lambda g, x: g.__setitem__(slice(None), 2 * x))
exa.add_obj(core, o)

model = exa.Model(core)
```

## `adapt`

`adapt=True` — the default here, and the opposite of the backend's — copies arrays to the
host before each call. A Python callback cannot run inside a device kernel, so this is what
makes an oracle usable at all from Python. Set it `False` only for a callback that is
genuinely device-capable, which a Python one is not.
