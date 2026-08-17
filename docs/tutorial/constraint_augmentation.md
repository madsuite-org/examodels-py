# Adding terms to constraints

Some constraints are assembled from several sources: a power balance at a bus collects a
term from every line and every generator touching it. Building an explicit sum per row
would defeat the point, so terms are added into rows already declared.

```python
from collections import namedtuple

import examodels as exa

Bus = namedtuple("Bus", "i pd gs")
Arc = namedtuple("Arc", "i bus")
Gen = namedtuple("Gen", "i bus")
bus = [Bus(0, 1.0, 0.1), Bus(1, 0.5, 0.0)]
arc = [Arc(0, 0), Arc(1, 1)]
gen = [Gen(0, 0)]

core = exa.Core()
vm = exa.add_var(core, len(bus), start=1.0)
p = exa.add_var(core, len(arc))
pg = exa.add_var(core, len(gen))

balance = exa.add_con(core, lambda b: b.pd + b.gs * vm[b.i]**2, over=bus)

exa.add_con(core, balance, lambda a: (a.bus, p[a.i]),  over=arc)
exa.add_con(core, balance, lambda g: (g.bus, -pg[g.i]), over=gen)
```

The first call declares one row per bus and returns a handle. Each later call takes that
handle and a function returning `(row, expression)`; the expression is added into that row.
This mirrors the backend's `add_con` / `add_con!` pair.

No rows are added by an augmentation — `model.ncon` is unchanged by the second and third
calls. A block may be augmented any number of times, from index sets of different shapes.

An empty block can also be declared first and filled entirely afterwards:

```python
N = 4
y = exa.add_var(core, N)

blk = exa.add_con(core, N, lcon=2.0, ucon=2.0)      # N rows, no terms yet
exa.add_con(core, blk, lambda i: (i, y[i]), over=range(N))
```
