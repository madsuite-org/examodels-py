# AC optimal power flow

A worked model that uses most of the package at once: tables as index sets,
per-row field access, bounds that vary by row, and a balance constraint
assembled from several sources. The complete script is
[`examples/ac_opf.py`](https://github.com/madsuite-org/madsuite-py/blob/main/examples/ac_opf.py),
which runs against the [PGLib](https://github.com/power-grid-lib/pglib-opf)
benchmark cases:

```
python examples/ac_opf.py case14.m [--backend cuda] [--solver madnlp]
```

## The data are the index sets

Each block of constraints is written once and evaluated over a table of rows —
a numpy structured array, or any sequence of named tuples. The row is handed to
the function, and its fields are read off it:

<!-- not-tested: excerpts from examples/ac_opf.py, which needs a PGLib case file -->
```python
va = add_var(core, len(data["bus"]))
vm = add_var(core, len(data["bus"]), start=1.0,
             lvar=data["vmin"], uvar=data["vmax"])
pg = add_var(core, len(data["gen"]), lvar=data["pmin"], uvar=data["pmax"])
```

`lvar` and `uvar` are arrays here, one entry per row: bounds differ per bus and
per generator, and that is data rather than structure.

## One pattern, every row

The generation cost is a single quadratic pattern applied at every generator,
and the branch flow equations are four patterns applied at every branch:

<!-- not-tested: excerpts from examples/ac_opf.py, which needs a PGLib case file -->
```python
add_obj(core, lambda g: g.cost1 * pg[g.i]**2 + g.cost2 * pg[g.i] + g.cost3,
        over=gen)

add_con(core, lambda b: p[b.f_idx] - b.c5 * vm[b.f_bus]**2
        - b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
        - b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])),
        over=branch)
```

`b.f_bus` is a *field of the row*, used as an index into a variable block:
the trace runs once, with a symbolic row, and the resulting expression describes
every branch. Nothing loops at build time, which is what lets the derivatives be
evaluated as one parallel kernel over the whole table.

Bounds that are themselves per-row arrays are passed the same way:

<!-- not-tested: excerpts from examples/ac_opf.py, which needs a PGLib case file -->
```python
add_con(core, lambda b: va[b.f_bus] - va[b.t_bus], over=branch,
        lcon=data["angmin"], ucon=data["angmax"])
```

## A balance assembled from many sources

Power balance at a bus is a sum over things that are not known per bus — every
line touching it, every generator on it. Rather than materialising that sum,
the rows are created first and terms are added *into* them:

<!-- not-tested: excerpts from examples/ac_opf.py, which needs a PGLib case file -->
```python
pbal = add_con(core, lambda b: b.pd + b.gs * vm[b.i]**2, over=bus)
add_con(core, pbal, lambda a: (a.bus, p[a.i]), over=arc)     # each arc
add_con(core, pbal, lambda g: (g.bus, -pg[g.i]), over=gen)   # each generator
```

The augmenting function returns `(row, expression)`: which row to add to, and
what to add. This mirrors the backend's own `add_con` / `add_con!` pair, and is
described in detail in [](constraint_augmentation.md).

## Solving it

<!-- not-tested: excerpts from examples/ac_opf.py, which needs a PGLib case file -->
```python
core, blocks = ac_opf(data)
sol = exa.Model(core).solve(solver="ipopt")
print(sol.status, sol.objective)
print(sol[blocks["vm"]])           # voltage magnitudes, as a numpy array
```

The same model runs on a GPU by building the core with a device backend —
`exa.Core(backend="cuda")` — and solving with MadNLP, which the example does
with `--backend cuda --solver madnlp`. See [](gpu.md).
