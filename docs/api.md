# API

## Building a model

```{eval-rst}
.. autoclass:: examodels.Core
   :members:
```

## The built model

```{eval-rst}
.. autoclass:: examodels.Model
   :members:

.. autoclass:: examodels.Solution
   :members:
```

## Handles

```{eval-rst}
.. autoclass:: examodels.Block
   :members:

.. autoclass:: examodels.Constraint
   :members:

.. autoclass:: examodels.Expression
   :members:

.. autoclass:: examodels.Records
   :members:

.. autoclass:: examodels.Node
   :members:
```

## Index sets

```{eval-rst}
.. autofunction:: examodels.product
```

## Solvers and backends

```{eval-rst}
.. autofunction:: examodels.solve
.. autofunction:: examodels.available_solvers
.. autofunction:: examodels.install_solver
.. autofunction:: examodels.backends
.. autofunction:: examodels.install_backend
```

## Oracles, two-stage models and wrappers

```{eval-rst}
.. autofunction:: examodels.VectorNonlinearOracle
.. autofunction:: examodels.ScalarNonlinearOracle
.. autofunction:: examodels.has_matfree_jac
.. autofunction:: examodels.has_matfree_hess
.. autofunction:: examodels.TwoStageCore
.. autofunction:: examodels.get_nscen
.. autofunction:: examodels.get_var_scen
.. autofunction:: examodels.get_con_scen
.. autofunction:: examodels.new_tag
.. autofunction:: examodels.WrapperNLPModel
.. autofunction:: examodels.TimedNLPModel
.. autofunction:: examodels.CompressedNLPModel
```

## CuPy interchange

```{eval-rst}
.. autofunction:: examodels.as_cupy
.. autofunction:: examodels.from_cupy
```

## Elementwise functions

`sin`, `cos`, `exp`, `log`, … are generated from the backend's own registry of supported
operators, so `dir(examodels)` is the authoritative list. Use these rather than `math` or
`numpy` equivalents inside a traced expression.
