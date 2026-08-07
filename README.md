# examodels

Python interface to [ExaModels.jl](https://github.com/exanauts/ExaModels.jl) — SIMD-parallel
algebraic modeling and automatic differentiation for nonlinear programs, on CPU threads or GPUs.

```python
import examodels as exa

N = 10
m = exa.Model()
x = m.add_variables(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])

m.minimize(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
           over=range(1, N))

m.constrain(lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
            + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
            + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
            over=range(0, N - 2), lower=0.0, upper=0.0)

sol = m.solve(solver="ipopt")
print(sol.status, sol.objective, sol[x])
```

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

The backend is fetched automatically on first use. `import examodels` does not start it —
the runtime boots the first time you build a model, so importing the package stays instant.

Solvers are backend packages rather than Python ones, so they are installed through
this package (once, per environment):

```python
import examodels as exa
exa.install_solver("ipopt")      # CPU
exa.install_solver("madnlp")     # CPU or GPU
exa.available_solvers()          # ['ipopt', 'madnlp']
```

## GPU

```python
m = exa.Model(backend="cuda")     # then solve with madnlp
```

## API

| | |
|---|---|
| `Model()` | start building |
| `.add_variables(n, start=, lower=, upper=)` | a block of variables; index it with `[i]` |
| `.minimize(f, over=)` | add `sum(f(i) for i in over)` to the objective |
| `.constrain(f, over=, lower=, upper=)` | one row per index, `lower <= f(i) <= upper` |
| `.build()` | finish; returns a `Problem` |
| `.solve(solver=)` | build and solve; returns a `Solution` |
| `Problem` | `.nvar` `.ncon` `.nnzj` `.nnzh` `.x0` `.objective(x)` `.gradient(x)` `.constraints(x)` |
| `Solution` | `.status` `.objective` `.iterations` `.x` `.y` `.elapsed` `.success`, and `sol[x]` |

Everything crossing the boundary is a Python scalar, a `range`, or a numpy array.

## Tests

```
pip install -e ".[test]"
pytest
```

The suite checks, among other things, that a Python-written expression produces *exactly* the
same expression the backend builds for the equivalent native model — compared by structural
identity, not by printed form.
