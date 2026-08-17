# CuPy interchange

Device memory is shared with CuPy in both directions, with no host round-trip.

<!-- not-tested: needs a CUDA device and cupy -->
```python
import cupy

model.set_start(x, cupy.full(n, 2.5))     # given straight to the model
model.objective(cupy_point)               # evaluated where the data already is
view = exa.as_cupy(backend_array)         # a CuPy view of the model's own memory
```

`as_cupy` publishes the backend array through CUDA's array interface — the backend does not
expose that interface itself, so it is built from the array's pointer, length and element
size. The result **aliases** the same memory: writing through the view writes into the
model. The test for this asserts pointer identity rather than equal values, because equal
values would pass on a copy.

`from_cupy` is the other direction, and is what the setters and the evaluation helpers use
when handed anything exposing `__cuda_array_interface__`.

## Ownership

The backend keeps ownership of its memory; a view holds a reference to the owner so it
cannot be freed underneath. Going the other way, the CuPy array must outlive the wrapped
view — the backend will not keep it alive for you.

## Coexisting in one process

CuPy and the backend's CUDA runtime work side by side; an allocation made by one survives
the other using the device. There is one caveat that is a library-path problem rather than
an interop one: CuPy's pip-installed CUDA libraries can shadow the backend's own, which the
backend warns about. Keeping `site-packages/nvidia/*/lib` off `LD_LIBRARY_PATH` avoids it.

Install with `pip install examodels[cuda]`.
