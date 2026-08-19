# Index sets

An index set decides what an expression is evaluated over. Four kinds are supported.
Every block on this page runs, in order, against this model:

```python
import madsuite as exa

N, T = 8, 4
core = exa.Core()
x = exa.add_var(core, N, start=1.0)
```

## Ranges

```python
exa.add_con(core, lambda i: x[i] - x[i+1], over=range(N - 1))
exa.add_con(core, lambda i: x[i] - 1.0, over=range(1, 9, 2))    # stepped
```

A stepped range works as an index set. It cannot be a *dimension* of a variable block —
the backend defines lengths only for whole ranges — and saying so is the error you get.

## Tables

For anything data-driven, rows are named tuples:

```python
from collections import namedtuple

Gen = namedtuple("Gen", "i bus cost1 cost2 cost3")
gen = [Gen(0, 3, 1100.0, 500.0, 0.0),
       Gen(1, 5, 850.0, 400.0, 0.0)]

pg = exa.add_var(core, len(gen))
exa.add_obj(core, lambda g: g.cost1 * pg[g.i]**2 + g.cost2 * pg[g.i] + g.cost3, over=gen)
```

Fields are read off the traced row. Those named in `index` hold positions of variables and
are kept as integers; the rest become floats.

## Products

For a rectangular set of indices, declare the block with one dimension per axis:

```python
y = exa.add_var(core, T, N)                       # index as y[t, i]
exa.add_con(core, lambda t, i: y[t, i] - y[t-1, i],
            over=exa.product(range(1, T), range(N)))
```

The traced function takes one index per dimension. In the generator form the same thing is
written `for t, i in exa.product(...)`.

## Subexpressions

A named expression, reusable across objectives and constraints. It is **inlined** at each
use — it adds no variables and no constraint rows:

```python
sq = exa.add_expr(core, lambda i: x[i]**2, over=range(N))
exa.add_obj(core, lambda i: (sq[i] - 1.0)**2, over=range(N))
exa.add_con(core, lambda i: sq[i] + sq[i+1], over=range(N - 1))

model = exa.Model(core)
model.nvar, model.ncon
```

Uses that share a structure share derivative code, exactly as if written out.
