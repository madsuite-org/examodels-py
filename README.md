# examodels

*The Python interface to [ExaModels.jl](https://github.com/madsuite-org/ExaModels.jl) — an [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming), on CPU threads and GPUs.*

[![test](https://github.com/madsuite-org/examodels-py/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/madsuite-org/examodels-py/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/madsuite-org/examodels-py/branch/main/graph/badge.svg)](https://codecov.io/gh/madsuite-org/examodels-py)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://madsuite.org/examodels-py/)

> [!WARNING]
> **Experimental — not yet stable.** The API may change without notice. Pin a
> commit if you depend on it, and please report what breaks.

## Overview

Models are written in Python; ExaModels.jl evaluates them — objective,
constraints, and sparse first and second derivatives — through
pattern-specialized kernels, on CPU threads or GPUs. A model is a small number
of algebraic patterns, each paired with an iterator over the data points where
the pattern applies; a generator expression carries both, so the model reads
the way it reads in ExaModels.jl:

```python
import examodels as exa

N = 10
core = exa.Core()
x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])

core.add_obj(100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2 for i in range(1, N))

core.add_con((3 * x[i+1]**3 + 2 * x[i+2] - 5
              + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
              + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3
              for i in range(0, N - 2)), lcon=0.0, ucon=0.0)

model = exa.Model(core)
sol = model.solve(solver="ipopt")
print(sol.status, sol.objective, sol[x])
```

```
first_order 6.2324586324 [-0.95055636  0.91390082  0.98909052 ... 0.99999993]
```

What the SIMD abstraction buys — pattern-specialized derivative kernels,
coloring-free sparse automatic differentiation, native GPU execution — is
[ExaModels.jl's story](https://github.com/madsuite-org/ExaModels.jl#overview);
this package carries it into Python:

- **The same model, CPU or GPU.** `exa.Core(backend="cuda")` builds the model
  where a GPU-capable solver such as
  [MadNLP.jl](https://github.com/madsuite-org/MadNLP.jl) can consume it
  in place.
- **Recipes.** `exa.Core(nargs=...)` writes the model against placeholders, so
  one core instantiates at any size and data — and can be compiled
  ahead-of-time into a self-contained shared library with a plain C interface
  (`exa.compile_library`).
- **A compiled-model cache.** `exa.Core(cache=True)` keys a compiled library
  by the model's structure: the first run compiles once, and every later run
  of the unchanged script loads the library through
  [cnlpmodels](https://github.com/madsuite-org/cnlpmodels-py) and solves with
  no compilation overhead at all — about half a second end to end.

The [manual](https://madsuite.org/examodels-py/) covers all of it: parameters,
multi-dimensional blocks, indexing over data tables, GPUs, recipes, the
compiler, and what the first call costs.

## Installation

```
pip install "examodels @ git+https://github.com/madsuite-org/examodels-py"
```

Backends install per-environment, once — `examodels.install_solver("ipopt")`,
`examodels.install_compiler()` — and Julia itself arrives automatically the
first time it is needed. Details, including the GPU and compiler
requirements, are in the [install guide](https://madsuite.org/examodels-py/install/).

## Citation

If you use this package in your research, please cite:

```bibtex
@article{shin2024accelerating,
  title   = {Accelerating optimal power flow with {GPUs}: {SIMD} abstraction of nonlinear programs and condensed-space interior-point methods},
  author  = {Shin, Sungho and Anitescu, Mihai and Pacaud, Fran\c{c}ois},
  journal = {Electric Power Systems Research},
  volume  = {236},
  pages   = {110651},
  year    = {2024},
  doi     = {10.1016/j.epsr.2024.110651}
}
```

## Supporting examodels

- Report issues and feature requests via the
  [GitHub issue tracker](https://github.com/madsuite-org/examodels-py/issues).
- Questions are welcome at the
  [ExaModels.jl discussion forum](https://github.com/madsuite-org/ExaModels.jl/discussions).
