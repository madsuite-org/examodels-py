"""AC optimal power flow, written in Python, on the PGLib benchmark cases.

    python examples/ac_opf.py [case.m] [--backend cuda] [--solver madnlp]

The model follows the standard polar-form AC OPF: bus voltage magnitudes and
angles, generator real and reactive dispatch, and per-arc power flows, subject to
the branch flow equations, thermal limits, angle-difference limits, and power
balance at every bus.
"""
import os
import os
import sys
import time

import numpy as np

import examodels as exa
import matpower

DEFAULT_CASE = os.environ.get(
    "PGLIB_CASE",
    os.path.expanduser("~/git/pglib-opf/pglib_opf_case3_lmbd.m"))


def ac_opf(data, backend=None):
    cos, sin = exa.cos, exa.sin
    core = exa.Core(backend=backend)

    rate = np.array(data["rate_a"])
    ninf = np.full(len(data["branch"]), -np.inf)

    va = core.add_var(len(data["bus"]))
    vm = core.add_var(len(data["bus"]), start=1.0, lower=data["vmin"], upper=data["vmax"])
    pg = core.add_var(len(data["gen"]), lower=data["pmin"], upper=data["pmax"])
    qg = core.add_var(len(data["gen"]), lower=data["qmin"], upper=data["qmax"])
    p = core.add_var(len(data["arc"]), lower=-rate, upper=rate)
    q = core.add_var(len(data["arc"]), lower=-rate, upper=rate)

    bus = exa.Records(data["bus"], index=["i"])
    gen = exa.Records(data["gen"], index=["i", "bus"])
    arc = exa.Records(data["arc"], index=["i", "bus"])
    branch = exa.Records(data["branch"], index=["f_idx", "t_idx", "f_bus", "t_bus"])
    refs = exa.Records(data["ref_buses"], index=["b"])

    # generation cost
    core.add_obj(g.cost1 * pg[g.i]**2 + g.cost2 * pg[g.i] + g.cost3 for g in gen)

    # reference bus angle
    core.add_con(va[r.b] for r in refs)

    # branch flow equations, from and to ends
    core.add_con(p[b.f_idx] - b.c5 * vm[b.f_bus]**2
                 - b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
                 - b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
                 for b in branch)
    core.add_con(q[b.f_idx] + b.c6 * vm[b.f_bus]**2
                 + b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus]))
                 - b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
                 for b in branch)
    core.add_con(p[b.t_idx] - b.c7 * vm[b.t_bus]**2
                 - b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
                 - b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
                 for b in branch)
    core.add_con(q[b.t_idx] + b.c8 * vm[b.t_bus]**2
                 + b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus]))
                 - b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
                 for b in branch)

    # angle differences and thermal limits
    core.add_con((va[b.f_bus] - va[b.t_bus] for b in branch),
                 lower=data["angmin"], upper=data["angmax"])
    core.add_con((p[b.f_idx]**2 + q[b.f_idx]**2 - b.rate_a_sq for b in branch),
                 lower=ninf, upper=0.0)
    core.add_con((p[b.t_idx]**2 + q[b.t_idx]**2 - b.rate_a_sq for b in branch),
                 lower=ninf, upper=0.0)

    # power balance: start from load and shunt, then add every line and generator
    pbal = core.add_con(b.pd + b.gs * vm[b.i]**2 for b in bus)
    qbal = core.add_con(b.qd - b.bs * vm[b.i]**2 for b in bus)
    core.add_con(pbal, ((a.bus, p[a.i]) for a in arc))
    core.add_con(qbal, ((a.bus, q[a.i]) for a in arc))
    core.add_con(pbal, ((g.bus, -pg[g.i]) for g in gen))
    core.add_con(qbal, ((g.bus, -qg[g.i]) for g in gen))

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
