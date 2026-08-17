# Parameters

A parameter behaves like a fixed variable inside an expression, and its value can be
changed afterwards without rebuilding the model — the derivative code is unaffected.

```python
import examodels as exa

N = 10
core = exa.Core()
th = exa.add_par(core, [100.0, 1.0])
x = exa.add_var(core, N, start=1.0)

exa.add_obj(core, lambda i: th[0] * (x[i-1]**2 - x[i])**2 + (x[i-1] - th[1])**2,
            over=range(1, N))

model = exa.Model(core)
first = model.solve()

model.set_value(th, [200.0, 1.0])
second = model.solve()                 # same model, new values
```

`get_value` reads them back. On a device model the values are placed on the device;
nothing about the call changes.

## Reading and changing a built model

Every one of the backend's accessors is available, on variable, parameter and constraint
blocks:

| read | change |
|---|---|
| `get_value` | `set_value` |
| `get_start` | `set_start` |
| `get_lvar`, `get_uvar` | `set_lvar`, `set_uvar` |
| `get_lcon`, `get_ucon` | `set_lcon`, `set_ucon` |

All of them work in place, and values are given and returned in the shape you used — a
multi-dimensional block is never handed back transposed or flattened.
