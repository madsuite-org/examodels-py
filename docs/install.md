# Installation

```
pip install git+https://github.com/madsuite-org/madsuite-py
```

The package is not on PyPI yet, so it is installed from the repository. Python
3.9 or newer; numpy and `juliacall` come with it.

:::{admonition} You do not need Julia installed
:class: note

The backend is Julia, but you never install or invoke it: `juliapkg` downloads
a private Julia and resolves the backend into your environment the first time
you build a model. `import madsuite` does **not** start it — importing stays
instant, and the runtime boots on first use.
:::

That first build therefore takes a while and needs a network; afterwards
everything is local. The runtime lives under `~/.julia`, and the resolved
environment beside your virtualenv in `.venv/julia_env`.

## Solvers

Solvers are backend packages rather than Python ones, so they are installed
through this package, once per environment:

<!-- not-tested: installs a backend package: needs a network, and changes the environment -->
```python
import madsuite as exa

exa.install_solver("ipopt")      # CPU
exa.install_solver("madnlp")     # CPU or GPU
exa.available_solvers()          # ['ipopt', 'madnlp']
```

Without one, models still build and evaluate — only `solve` needs a solver.

## GPUs

<!-- not-tested: needs a CUDA device; installs a backend package -->
```python
exa.install_backend("cuda")      # or "rocm", "oneapi", "metal"
core = exa.Core(backend="cuda")
```

`exa.backends()` lists what can be constructed. Each backend package is loaded
only when asked for, so a CPU model never starts a GPU runtime. For sharing
device memory with CuPy, install the extra as well:

```
pip install "madsuite[cuda] @ git+https://github.com/madsuite-org/madsuite-py"
```

## The compiler

Compiling a model into a shared library ([](recipe.md)) needs one more backend
package:

<!-- not-tested: installs the compiler backend, which also needs Julia 1.12 -->
```python
exa.install_compiler()
```

It has a requirement the rest of the package does not, and it is worth knowing
before you meet it: compilation needs **Julia 1.12**, and `juliapkg` refuses to
install a Julia newer than 1.11 when the Python process links **OpenSSL older
than 3.5** — which Julia 1.12 requires. A system Python on an older OpenSSL
therefore gets recipes but not compilation, and the failure appears as an
unsatisfiable `OpenSSL_jll` when resolving:

```
ERROR: Unsatisfiable requirements detected for package OpenSSL_jll
  restricted to versions 3.0 by project — no versions left
```

A conda-forge Python (or any build on OpenSSL ≥ 3.5) resolves it. Check yours
with:

```python
import ssl; print(ssl.OPENSSL_VERSION)
```

Compiled models come back through the `[cache]` extra — the julia-free
loading half of [the model cache](cache.md):

```
pip install "madsuite[cache] @ git+https://github.com/madsuite-org/madsuite-py"
```

## Development

```
git clone https://github.com/madsuite-org/madsuite-py
cd madsuite
pip install -e ".[test]"
pytest -q -m "not slow"
```

`.[docs]` builds this manual (`python -m sphinx -b html docs docs/_build`), and
`.[lint]` installs the pinned linter the CI uses.
