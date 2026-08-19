# Mathematical abstraction

The backend represents a nonlinear program in the form

$$
\min_{x} \; \sum_{l} \sum_{k} f^{(l)}(x; s^{(l)}_k)
\quad \text{s.t.} \quad
g^\flat \le \Big[ g^{(m)}(x; q_j) \Big]_j \le g^\sharp, \qquad \ell \le x \le u,
$$

that is, as a small number of *expression patterns*, each evaluated over many data points.
That repetition is what makes the derivatives a single parallel kernel rather than a graph
walk.

This package preserves that structure rather than flattening it. An expression is written
once, as a function of an index:

```python
import madsuite as exa

N = 10
core = exa.Core()
x = exa.add_var(core, N, start=1.0)

exa.add_obj(core, lambda i: 100 * (x[i-1]**2 - x[i])**2, over=range(1, N))
```

and is evaluated **once**, with a symbolic index. The operators applied to that index build
one structured expression describing every row; the loop never runs at build time. What is
sent to the backend is a pattern plus a data array, not `N` separate expressions.

Two consequences follow directly, and both are visible in the API:

- **An expression may not branch on its index.** `x[i] if i > 3 else x[i-1]` raises a
  `TypeError`, because at trace time `i` has no value. Anything index-dependent belongs in
  the data — `start`, `lvar`, `ucon`, or the index set itself — all of which are evaluated
  per index in the ordinary way.
- **Only registered operators may appear.** Use `exa.sin`, not `math.sin`. The available
  functions are generated from the backend's own registry, so `dir(madsuite)` is the
  authoritative list.

Each distinct expression *shape* compiles its own derivative kernel the first time it is
built. Building the same shape again is free; see [performance](tutorial/performance.md).
