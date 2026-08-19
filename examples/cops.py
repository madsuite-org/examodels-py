"""Three problems from the COPS benchmark set, written in Python.

    python examples/cops.py [elec|goddard|minsurf] [N]

Ported from COPSBenchmark.jl so the two can be compared directly; the test suite
builds the Julia model alongside each of these and checks they agree.
"""
import sys
import time
from collections import namedtuple

import numpy as np

import madsuite as exa
from madsuite import add_con, add_obj, add_var

Pair = namedtuple("Pair", "i j")


def elec(npoints, seed=2713, backend=None, start=None):
    """Electrons on a sphere: minimise Coulomb potential, all points on the unit ball."""
    if start is None:                       # a quasi-uniform scatter
        rng = np.random.default_rng(seed)
        theta, phi = 2 * np.pi * rng.random(npoints), np.pi * rng.random(npoints)
        start = (np.cos(theta) * np.sin(phi),
                 np.sin(theta) * np.sin(phi),
                 np.cos(phi))
    core = exa.Core(backend=backend)
    x = add_var(core, npoints, start=start[0])
    y = add_var(core, npoints, start=start[1])
    z = add_var(core, npoints, start=start[2])

    pairs = [Pair(i, j) for i in range(npoints - 1)
                         for j in range(i + 1, npoints)]
    add_obj(core, lambda p: 1.0 / exa.sqrt((x[p.i] - x[p.j])**2
                                           + (y[p.i] - y[p.j])**2
                                           + (z[p.i] - z[p.j])**2), over=pairs)
    add_con(core, lambda i: x[i]**2 + y[i]**2 + z[i]**2 - 1.0, over=range(npoints))
    return core, dict(x=x, y=y, z=z)


def goddard(nh, backend=None):
    """The Goddard rocket: maximise final altitude."""
    h_0, v_0, m_0, g_0 = 1.0, 0.0, 1.0, 1.0
    T_c, h_c, v_c, m_c = 3.5, 500.0, 620.0, 0.6
    c = 0.5 * np.sqrt(g_0 * h_0)
    m_f = m_c * m_0
    D_c = 0.5 * v_c * (m_0 / g_0)
    T_max = T_c * m_0 * g_0

    core = exa.Core(backend=backend, minimize=False)
    h = add_var(core, range(0, nh + 1), start=1.0, lvar=1.0)
    v = add_var(core, range(0, nh + 1),
                start=[i / nh * (1 - i / nh) for i in range(nh + 1)], lvar=0.0)
    m = add_var(core, range(0, nh + 1),
                start=[(m_f - m_0) * (i / nh) + m_0 for i in range(nh + 1)],
                lvar=m_f, uvar=m_0)
    T = add_var(core, range(0, nh + 1), start=T_max / 2, lvar=0.0, uvar=T_max)
    step = add_var(core, 1, start=1 / nh, lvar=0.0)

    add_obj(core, h[nh], over=range(1))

    def drag(i):
        return (T[i] - D_c * v[i]**2 * exa.exp(-h_c * (h[i] - h_0)) / h_0
                - m[i] * g_0 * (h_0 / h[i])**2) / m[i]

    add_con(core, lambda i: -h[i] + h[i-1] + 0.5 * step[0] * (v[i] + v[i-1]),
            over=range(1, nh + 1))
    add_con(core, lambda i: -v[i] + v[i-1] + 0.5 * step[0] * (drag(i) + drag(i - 1)),
            over=range(1, nh + 1))
    add_con(core, lambda i: -m[i] + m[i-1] + 0.5 * step[0] * (-T[i] / c + -T[i-1] / c),
            over=range(1, nh + 1))
    add_con(core, h[0] - h_0, over=range(1))
    add_con(core, v[0] - v_0, over=range(1))
    add_con(core, m[0] - m_0, over=range(1))
    add_con(core, m[nh] - m_f, over=range(1))
    return core, dict(h=h, v=v, m=m, T=T, step=step)


def minsurf(nx, ny=None, backend=None):
    """Minimal surface over a square, with the boundary pinned: a PDE-shaped problem."""
    ny = nx if ny is None else ny
    hx, hy = 1 / (nx + 1), 1 / (ny + 1)
    area = 0.5 * hx * hy
    mesh = np.linspace(0, 1, nx + 2)
    v0 = np.tile((1 - (2 * mesh - 1)**2)[:, None], (1, ny + 2))

    core = exa.Core(backend=backend)
    v = add_var(core, nx + 2, ny + 2, start=v0)

    add_obj(core, lambda i, j: area * (1 + ((v[i+1, j] - v[i, j]) / hx)**2
                                       + ((v[i, j+1] - v[i, j]) / hy)**2)**0.5,
            over=exa.product(range(nx + 1), range(ny + 1)))
    add_obj(core, lambda i, j: area * (1 + ((v[i-1, j] - v[i, j]) / hx)**2
                                       + ((v[i, j-1] - v[i, j]) / hy)**2)**0.5,
            over=exa.product(range(1, nx + 2), range(1, ny + 2)))

    add_con(core, lambda j: v[0, j], over=range(ny + 2))
    add_con(core, lambda j: v[nx + 1, j], over=range(ny + 2))
    add_con(core, lambda i: v[i, 0] - 1 + (2 * i * hx - 1)**2, over=range(nx + 2))
    add_con(core, lambda i: v[i, ny + 1] - 1 + (2 * i * hx - 1)**2, over=range(nx + 2))
    # the surface stays above zero everywhere, and above one over the middle square
    add_con(core, lambda i, j: v[i, j], over=exa.product(range(nx + 2), range(ny + 2)),
            lcon=0.0, ucon=np.inf)
    lo_i, hi_i = int(np.floor(0.25 / hx)), int(np.ceil(0.75 / hx))
    lo_j, hi_j = int(np.floor(0.25 / hy)), int(np.ceil(0.75 / hy))
    add_con(core, lambda i, j: v[i, j],
            over=exa.product(range(lo_i, hi_i + 1), range(lo_j, hi_j + 1)),
            lcon=1.0, ucon=np.inf)
    return core, dict(v=v)


Coll = namedtuple("Coll", "i k s rho")
IS = namedtuple("IS", "i s")
IJ = namedtuple("IJ", "i j")
SB = namedtuple("SB", "s bc")


def catmix(nh, backend=None):
    """Catalyst mixing: collocation on a two-state DAE.

    The trickiest of these to express — three-dimensional blocks, and index sets
    that mix a range with a table of collocation points. `Records` covers that,
    where `product` (ranges only) does not.
    """
    ne, nc, tf = 2, 3, 1
    h = tf / nh
    rho = [0.11270166537926, 0.50000000000000, 0.88729833462074]
    bc, alpha = [1.0, 0.0], 0.0
    fact = [1, 1, 2, 6]

    core = exa.Core(backend=backend)
    u = add_var(core, nh, nc, lvar=np.zeros((nh, nc)), uvar=np.ones((nh, nc)),
                start=np.zeros((nh, nc)))
    v = add_var(core, nh, ne, start=np.tile([(j + 1) % ne for j in range(ne)], (nh, 1)))
    w = add_var(core, nh, nc, ne, start=np.zeros((nh, nc, ne)))
    pp = add_var(core, nh, nc, ne,
                 start=np.broadcast_to([(k + 1) % ne for k in range(ne)],
                                       (nh, nc, ne)).copy())
    Dpp = add_var(core, nh, nc, ne, start=np.zeros((nh, nc, ne)))
    ppf = add_var(core, ne, start=[(i + 1) % ne for i in range(ne)])

    add_obj(core, -1.0 + ppf[0] + ppf[1], over=range(1))
    if alpha:
        add_obj(core, lambda i, j: alpha / h * (u[i+1, j] - u[i, j])**2,
                over=exa.product(range(nh - 1), range(nc)))

    coll = [Coll(i, k, s, rho[k]) for i in range(nh)
                        for k in range(nc) for s in range(ne)]
    add_con(core, lambda c: pp[c.i, c.k, c.s] - v[c.i, c.s]
            - h * exa.sum([w[c.i, j, c.s] * (c.rho**(j + 1) / fact[j + 1])
                           for j in range(nc)]), over=coll)
    add_con(core, lambda c: Dpp[c.i, c.k, c.s]
            - exa.sum([w[c.i, j, c.s] * (c.rho**j / fact[j]) for j in range(nc)]),
            over=coll)

    add_con(core, lambda s: ppf[s] - v[nh - 1, s]
            - h * exa.sum([w[nh - 1, j, s] / fact[j + 1] for j in range(nc)]),
            over=range(ne))

    isr = [IS(i, s) for i in range(nh - 1) for s in range(ne)]
    add_con(core, lambda r: v[r.i, r.s]
            + exa.sum([w[r.i, j, r.s] * h / fact[j + 1] for j in range(nc)])
            - v[r.i + 1, r.s], over=isr)

    ij = [IJ(i, j) for i in range(nh) for j in range(nc)]
    add_con(core, lambda r: Dpp[r.i, r.j, 0]
            - u[r.i, r.j] * (10.0 * pp[r.i, r.j, 1] - pp[r.i, r.j, 0]), over=ij)
    add_con(core, lambda r: Dpp[r.i, r.j, 1]
            - u[r.i, r.j] * (pp[r.i, r.j, 0] - 10.0 * pp[r.i, r.j, 1])
            + (1 - u[r.i, r.j]) * pp[r.i, r.j, 1], over=ij)

    sb = [SB(i, bc[i]) for i in range(ne)]
    add_con(core, lambda r: v[0, r.s] - r.bc, over=sb)
    return core, dict(u=u, v=v, w=w, pp=pp, Dpp=Dpp, ppf=ppf)


MODELS = {"elec": (elec, 50), "goddard": (goddard, 400), "minsurf": (minsurf, 50),
          "catmix": (catmix, 100)}


def main(argv):
    which = argv[0] if argv else "elec"
    build, default_n = MODELS[which]
    n = int(argv[1]) if len(argv) > 1 else default_n
    backend = argv[argv.index("--backend") + 1] if "--backend" in argv else None

    t = time.perf_counter()
    core, var = build(n, backend=backend)
    model = exa.Model(core)
    built = time.perf_counter() - t
    print(f"{which}(N={n}) {model}  built in {built:.2f} s")

    sol = model.solve()
    print(f"  status     : {sol.status}")
    print(f"  objective  : {sol.objective:.8f}")
    print(f"  iterations : {sol.iterations}")
    print(f"  solve      : {sol.elapsed:.2f} s")
    print(f"  violation  : {model.violation(sol.x):.2e}")


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    main(sys.argv[1:])
