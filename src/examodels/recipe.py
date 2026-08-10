"""Recipes: a core written against placeholders, given its data later.

A `Core` is normally built from data you already have, so the model and its data
are finished together. Ask for placeholders instead and the two come apart:

    core = exa.Core(nargs=2)
    N, x0 = core.args
    x = core.add_var(N, start=x0)
    core.add_obj(lambda i: (x[i] - 1)**2, over=exa.srange(0, N))

    model = exa.Model(core, 10, [0.0] * 10)      # one value per placeholder
    other = exa.Model(core, 50, [0.0] * 50)      # the same core, again

`exa.recipe(nargs=2)` is the same thing in one line, for when unpacking reads
better than a separate `.args`:

    core, N, x0 = exa.recipe(nargs=2)

The reason the concept exists is ahead-of-time compilation: it needs the
*structure* to become code while the *data* stays a run-time input. Writing a
model against placeholders is that separation, so a recipe that builds and
instantiates can be compiled without the author reasoning about what survives
trimming.

A placeholder is used *as the value it stands for* — `N` is the number of
variables, not a namespace to reach into. Arithmetic on it is deferred:
`N - 1` and `srange(0, N)` describe what to compute, and compute it when the
model is built.

What a placeholder cannot do is take part in Python control flow. Its value does
not exist while the model is being written, so `len(N)`, `int(N)`, `if N > 3`
and iteration all raise — anything of that kind belongs in the data you pass at
instantiation, or is computed by the caller beforehand.
"""

from . import _bridge as _b


class Arg:
    """A stand-in for one of the values supplied when the model is built.

    Obtained from `Core(nargs=...)`, never constructed directly. Supports the
    arithmetic needed to describe sizes and index sets; everything else raises,
    because the value is not known yet.
    """

    __slots__ = ("_jl",)

    def __init__(self, jl):
        self._jl = jl

    # ── arithmetic, deferred ────────────────────────────────────────────────
    #
    # Each operation builds a backend node rather than a number. The reflected
    # forms matter as much as the direct ones: `2 * N` is as natural to write as
    # `N * 2`, and a size expression like `3 * N - 1` uses both.

    def _op(self, name, other, flip=False):
        a, b = (_unwrap(other), self._jl) if flip else (self._jl, _unwrap(other))
        return Arg(_b.ops[name](a, b))

    def __add__(self, o):
        return self._op("+", o)

    def __radd__(self, o):
        return self._op("+", o, flip=True)

    def __sub__(self, o):
        return self._op("-", o)

    def __rsub__(self, o):
        return self._op("-", o, flip=True)

    def __mul__(self, o):
        return self._op("*", o)

    def __rmul__(self, o):
        return self._op("*", o, flip=True)

    def __truediv__(self, o):
        return self._op("/", o)

    def __rtruediv__(self, o):
        return self._op("/", o, flip=True)

    def __neg__(self):
        return Arg(_b.ops["-"](0, self._jl))

    def __pow__(self, n):
        return Arg(_b.ops["^"](self._jl, _unwrap(n)))

    # ── what a placeholder deliberately refuses ─────────────────────────────
    #
    # Each of these has a value only after instantiation. Raising here names the
    # problem where it is made; allowing them would produce a model whose shape
    # silently depended on a value nobody had supplied yet.

    def _refuse(self, what):
        raise TypeError(
            f"cannot {what} a placeholder: its value is not known until the model "
            f"is built. Compute it before building and pass it as one of the "
            f"arguments to `Model(core, ...)`, or use it symbolically — `N - 1` "
            f"and `srange(0, N)` are fine."
        )

    def __len__(self):
        self._refuse("take len() of")

    def __int__(self):
        self._refuse("convert")

    def __index__(self):
        self._refuse("index with")

    def __iter__(self):
        self._refuse("iterate")

    def __bool__(self):
        self._refuse("branch on")

    def __lt__(self, o):
        self._refuse("compare")

    __le__ = __gt__ = __ge__ = __lt__

    def __repr__(self):
        return f"<placeholder {_b.typestr(self._jl)}>"


class SRange:
    """A half-open index set whose bounds may be placeholders — `srange(0, N)`.

    Half-open like `range`, so `srange(0, N)` is `N` indices starting at 0. It
    exists because `range` requires integers: the moment a bound is a
    placeholder, Python cannot build the set, and the backend has to be told the
    bounds instead.
    """

    __slots__ = ("start", "stop")

    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def _jl(self):
        """The backend's index set: half-open here, inclusive there."""
        return _b.arg_range(_unwrap(self.start), _unwrap(self.stop))

    def __len__(self):
        raise TypeError(
            "an srange has no length until the model is built — its bounds are "
            "placeholders. Pass it as `over=` and let the backend size it."
        )

    def __repr__(self):
        return f"srange({self.start!r}, {self.stop!r})"


def srange(start, stop=None):
    """`srange(stop)` or `srange(start, stop)` — half-open, like `range`.

    Use it wherever an index set involves a placeholder; plain `range` stays
    correct everywhere else, and reads better.
    """
    return SRange(0, start) if stop is None else SRange(start, stop)


def is_placeholder(x):
    """Whether `x` is a placeholder or an expression over one."""
    return isinstance(x, Arg)


def _unwrap(x):
    """The backend value behind a placeholder, or `x` itself."""
    return x._jl if isinstance(x, Arg) else x


def recipe(nargs=1, **kwargs):
    """A `Core` and its placeholders in one go — `core, N = exa.recipe()`.

    Identical to `Core(nargs=...)` followed by unpacking `.args`; which reads
    better depends on the model, so both spellings exist.
    """
    from .core import Core
    core = Core(nargs=nargs, **kwargs)
    return (core, *core.args)
