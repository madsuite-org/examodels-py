# Two-stage stochastic models

```python
from collections import namedtuple

import examodels as exa

nscen = 3
Row = namedtuple("Row", "i t")
rows = [Row(i, 0.5 * i) for i in range(nscen * 4)]

core = exa.TwoStageCore(nscen)

d = exa.add_var(core, 2)                            # design, shared
v = exa.add_var(core, exa.EachScenario(), 4)        # recourse, per scenario

exa.add_obj(core, lambda r: (v[r.i] - r.t)**2, over=rows)
exa.add_con(core, exa.EachScenario(), lambda i: v[i] - d[0],
            over=range(nscen * 4))
```

`EachScenario()` marks a declaration as recourse rather than design. A per-scenario
*variable* declaration is replicated by the backend, so the block holds `nscen` times what
was asked for and is indexed flat. A per-scenario *constraint* is tagged rather than
replicated: supply the full index set, spanning all scenarios.

Once built:

```python
model = exa.Model(core)

exa.get_nscen(model)       # how many scenarios
exa.get_var_scen(model)    # scenario of each variable, 0 for the shared first stage
exa.get_con_scen(model)    # scenario of each constraint row
```

Tags are available directly — `FirstStageTag`, `SecondStageTag`,
`FirstStageConstraintTag`, `SecondStageConstraintTag` — and `new_tag(name, kind)` defines
one of your own, which can be attached with `tag=` on a declaration.
