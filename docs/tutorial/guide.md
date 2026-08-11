# Getting started

A model is built in two stages, mirroring the backend: a `Core` accumulates variables,
parameters, objectives and constraints, and a `Model` is built from it.

```python
import examodels as exa

core = exa.Core()
```

## Variables

```python
N = 10
x = exa.add_var(core, N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])
```

`start`, `lvar` and `uvar` take a number or an array. A dimension may be an integer or a
range, and there may be several:

```python
y = exa.add_var(core, T, N)               # index as y[t, i]
z = exa.add_var(core, range(2, 11))       # indices 2..10
```

Indices are 0-based, and an index is a number: `x[i] - i` means what it says.

## Objective

```python
exa.add_obj(core, lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
            over=range(1, N))
```

Terms are summed; call it more than once to add more.

## Constraints

```python
exa.add_con(core, lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
            + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
            + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
            over=range(0, N - 2))
```

Constraints are two-sided; `lcon` and `ucon` default to zero, and take a number or an
array. Use `-inf` or `inf` for a one-sided constraint.

## Solving

```python
model = exa.Model(core)
sol = model.solve()

sol.status        # 'first_order'
sol.objective     # 6.232458632
sol[x]            # the values of block x, as a numpy array
sol.multipliers(con)
```

Without `solver=`, Ipopt is used on the host and MadNLP on an accelerator — Ipopt cannot
take device arrays. Install one with `exa.install_solver("ipopt")`.

## Two ways to write the same call

Everything is available both as a function taking the core first, matching the backend's
argument order, and as a method:

```python
exa.add_obj(core, lambda i: x[i]**2, over=range(N))
core.add_obj(lambda i: x[i]**2, over=range(N))
```

A generator expression also works, and reads closest to the backend's macros. As a bare
argument it must be the only one, so it suits the method form:

```python
core.add_obj(x[i]**2 for i in range(N))
```

## Naming

Passing `name=` registers a block, retrievable from either the core or the model:

```python
x = exa.add_var(core, N, name="x")
model = exa.Model(core)
model.get_start(model.x)
```
