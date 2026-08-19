# API

## Building a model

```{eval-rst}
.. autoclass:: madsuite.Core
   :members:
```

## The built model

```{eval-rst}
.. autoclass:: madsuite.Model
   :members:

.. autoclass:: madsuite.Solution
   :members:
```

## Handles

```{eval-rst}
.. autoclass:: madsuite.Block
   :members:

.. autoclass:: madsuite.Constraint
   :members:

.. autoclass:: madsuite.Expression
   :members:

.. autoclass:: madsuite.Node
   :members:
```

## Index sets

```{eval-rst}
.. autofunction:: madsuite.product
```

## Recipes

```{eval-rst}
.. autofunction:: madsuite.recipe
.. autofunction:: madsuite.srange
```

## The compiler and the model cache

```{eval-rst}
.. autofunction:: madsuite.compile_library
.. autofunction:: madsuite.compiler_available
.. autofunction:: madsuite.install_compiler
.. autoclass:: madsuite.CompiledLibrary
```

`Core(cache=True)` needs no API of its own — recording, lookup, compile and
load all hang off `Core` and `Model`; see [](cache.md).

## Solvers and backends

```{eval-rst}
.. autofunction:: madsuite.solve
.. autofunction:: madsuite.available_solvers
.. autofunction:: madsuite.install_solver
.. autofunction:: madsuite.backends
.. autofunction:: madsuite.install_backend
```

## Oracles, two-stage models and wrappers

```{eval-rst}
.. autofunction:: madsuite.VectorNonlinearOracle
.. autofunction:: madsuite.ScalarNonlinearOracle
.. autofunction:: madsuite.has_matfree_jac
.. autofunction:: madsuite.has_matfree_hess
.. autofunction:: madsuite.TwoStageCore
.. autofunction:: madsuite.get_nscen
.. autofunction:: madsuite.get_var_scen
.. autofunction:: madsuite.get_con_scen
.. autofunction:: madsuite.new_tag
.. autofunction:: madsuite.WrapperNLPModel
.. autofunction:: madsuite.TimedNLPModel
.. autofunction:: madsuite.CompressedNLPModel
```

## CuPy interchange

```{eval-rst}
.. autofunction:: madsuite.as_cupy
.. autofunction:: madsuite.from_cupy
```

## Elementwise functions

`sin`, `cos`, `exp`, `log`, … are generated from the backend's own registry of supported
operators, so `dir(madsuite)` is the authoritative list. Use these rather than `math` or
`numpy` equivalents inside a traced expression.
