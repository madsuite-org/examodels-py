# Performance

Everything expensive is per *process*, not per model. On one machine, a first
1000-variable solve costs about 25 s; the same model built again in the same process costs
**0.08 s**, and a model with a different expression costs **1.3 s**.

The cost divides in two:

- **Fixed per process** (~9 s): starting the runtime, loading packages, compiling the
  solver's generic code.
- **Per model *shape*** (~9 s here): the backend encodes an expression — and the model
  built so far — in a type, so each distinct shape compiles its own derivative kernels.
  This cannot be precompiled: the type does not exist until your function runs. It is also
  the reason evaluation is fast.

So keep the process alive — a session, a notebook, or a worker — rather than paying it per
script.

A PackageCompiler system image was measured and is **not** recommended: it saves about
13%, and a custom image cannot load the GPU backends at all.

## Sizing

Model construction scales with the number of distinct *expressions*, not with the number
of rows. A constraint over a million-row index set costs the same to build as one over ten
rows; only the data array differs. This is the property worth designing a model around.
