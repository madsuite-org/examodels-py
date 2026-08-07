"""AC optimal power flow, written in Python, on the PGLib benchmark cases.

    python examples/ac_opf.py [case.m] [--backend cuda] [--solver madnlp]

The model follows the standard polar-form AC OPF: bus voltage magnitudes and
angles, generator real and reactive dispatch, and per-arc power flows, subject to
the branch flow equations, thermal limits, angle-difference limits, and power
balance at every bus.
"""
import os
import sys
import time

import matpower
import numpy as np

import examodels as exa
from examodels import add_con, add_obj, add_var

DEFAULT_CASE = os.environ.get(
    "PGLIB_CASE",
    os.path.expanduser("~/git/pglib-opf/pglib_opf_case3_lmbd.m"))


def ac_opf(data, backend=None):
    cos, sin = exa.cos, exa.sin
    core = exa.Core(backend=backend)

    rate = np.array(data["rate_a"])
    ninf = np.full(len(data["branch"]), -np.inf)

    va = add_var(core, len(data["bus"]))
    vm = add_var(core, len(data["bus"]), start=1.0, lvar=data["vmin"], uvar=data["vmax"])
    pg = add_var(core, len(data["gen"]), lvar=data["pmin"], uvar=data["pmax"])
    qg = add_var(core, len(data["gen"]), lvar=data["qmin"], uvar=data["qmax"])
    p = add_var(core, len(data["arc"]), lvar=-rate, uvar=rate)
    q = add_var(core, len(data["arc"]), lvar=-rate, uvar=rate)

    bus = data["bus"]
    gen = data["gen"]
    arc = data["arc"]
    branch = data["branch"]
    refs = data["ref_buses"]

    # generation cost
    add_obj(core, lambda g: g.cost1 * pg[g.i]**2 + g.cost2 * pg[g.i] + g.cost3, over=gen)

    # reference bus angle
    add_con(core, lambda r: va[r.b], over=refs)

    # branch flow equations, from and to ends
    add_con(core, lambda b: p[b.f_idx] - b.c5 * vm[b.f_bus]**2
            - b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
            - b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])),
            over=branch)
    add_con(core, lambda b: q[b.f_idx] + b.c6 * vm[b.f_bus]**2
            + b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
            - b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])),
            over=branch)
    add_con(core, lambda b: p[b.t_idx] - b.c7 * vm[b.t_bus]**2
            - b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
            - b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])),
            over=branch)
    add_con(core, lambda b: q[b.t_idx] + b.c8 * vm[b.t_bus]**2
            + b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
            - b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])),
            over=branch)

    # angle differences and thermal limits
    add_con(core, lambda b: va[b.f_bus] - va[b.t_bus], over=branch,
            lcon=data["angmin"], ucon=data["angmax"])
    add_con(core, lambda b: p[b.f_idx]**2 + q[b.f_idx]**2 - b.rate_a_sq,
            over=branch, lcon=ninf, ucon=0.0)
    add_con(core, lambda b: p[b.t_idx]**2 + q[b.t_idx]**2 - b.rate_a_sq,
            over=branch, lcon=ninf, ucon=0.0)

    # power balance: start from load and shunt, then add every line and generator
    pbal = add_con(core, lambda b: b.pd + b.gs * vm[b.i]**2, over=bus)
    qbal = add_con(core, lambda b: b.qd - b.bs * vm[b.i]**2, over=bus)
    add_con(core, pbal, lambda a: (a.bus, p[a.i]), over=arc)
    add_con(core, qbal, lambda a: (a.bus, q[a.i]), over=arc)
    add_con(core, pbal, lambda g: (g.bus, -pg[g.i]), over=gen)
    add_con(core, qbal, lambda g: (g.bus, -qg[g.i]), over=gen)

    return core, dict(va=va, vm=vm, pg=pg, qg=qg, p=p, q=q)


def main(argv):
    case = next((a for a in argv if a.endswith(".m")), DEFAULT_CASE)
    backend = argv[argv.index("--backend") + 1] if "--backend" in argv else None
    solver = argv[argv.index("--solver") + 1] if "--solver" in argv else "ipopt"

    data = matpower.read(case)
    print(f"{case.split('/')[-1]}: {len(data['bus'])} buses, "
          f"{len(data['gen'])} generators, {len(data['branch'])} branches")

    t = time.perf_counter()
    core, var = ac_opf(data, backend=backend)
    model = exa.Model(core)
    build = time.perf_counter() - t
    print(f"{model}  (built in {build:.2f} s)")

    sol = model.solve(solver=solver)
    print(f"status      : {sol.status}")
    print(f"objective   : {sol.objective:,.2f} $/hr")
    print(f"iterations  : {sol.iterations}")
    print(f"solve time  : {sol.elapsed:.2f} s")
    print(f"violation   : {model.violation(sol.x):.2e}")
    print(f"|V| range   : {sol[var['vm']].min():.4f} .. {sol[var['vm']].max():.4f} pu")
    print(f"generation  : {sol[var['pg']].sum() * data['baseMVA']:.2f} MW")


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    main(sys.argv[1:])
