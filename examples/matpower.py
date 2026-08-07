"""A small MATPOWER `.m` reader — enough for AC OPF, with no Julia involved.

Produces the per-unit quantities the OPF model needs, following the same
conventions as PowerModels: per-unit power, branch admittances and the composed
coefficients c1..c8, angle limits in radians.
"""
import cmath
import math
import re

_MATRIX = re.compile(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", re.S)
_SCALAR = re.compile(r"mpc\.(\w+)\s*=\s*([\d.eE+-]+)\s*;")


def _matrices(text):
    out = {}
    for name, body in _MATRIX.findall(text):
        rows = []
        for line in body.splitlines():
            line = line.split("%")[0].strip().rstrip(";").strip()
            if line:
                rows.append([float(v) for v in line.split()])
        out[name] = rows
    return out


def read(filename):
    text = open(filename).read()
    mats = _matrices(text)
    base = float(dict(_SCALAR.findall(text))["baseMVA"])

    buses = mats["bus"]
    busidx = {int(r[0]): k for k, r in enumerate(buses)}          # 0-based

    bus = {"i": [], "pd": [], "gs": [], "qd": [], "bs": []}
    vmin, vmax, ref_buses = [], [], []
    for k, r in enumerate(buses):
        bus["i"].append(k)
        bus["pd"].append(r[2] / base)
        bus["qd"].append(r[3] / base)
        bus["gs"].append(r[4] / base)
        bus["bs"].append(r[5] / base)
        vmax.append(r[11]); vmin.append(r[12])
        if int(r[1]) == 3:
            ref_buses.append(k)

    costs = mats.get("gencost", [])
    gen = {"i": [], "bus": [], "cost1": [], "cost2": [], "cost3": []}
    pmin, pmax, qmin, qmax = [], [], [], []
    g = 0
    for j, r in enumerate(mats["gen"]):
        if int(r[7]) == 0:                                        # out of service
            continue
        c = costs[j] if j < len(costs) else [2, 0, 0, 3, 0, 0, 0]
        ncost = int(c[3])
        coeffs = c[4:4 + ncost]
        c2, c1, c0 = ([0.0] * (3 - len(coeffs)) + list(coeffs))[-3:]
        gen["i"].append(g); gen["bus"].append(busidx[int(r[0])])
        gen["cost1"].append(c2 * base**2)                         # per-unit rescale
        gen["cost2"].append(c1 * base)
        gen["cost3"].append(c0)
        pmax.append(r[8] / base); pmin.append(r[9] / base)
        qmax.append(r[3] / base); qmin.append(r[4] / base)
        g += 1

    branches = [r for r in mats["branch"] if int(r[10]) != 0]
    nb = len(branches)
    branch = {k: [] for k in ("f_idx", "t_idx", "f_bus", "t_bus",
                              "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "rate_a_sq")}
    arc = {"i": [], "bus": [], "rate_a": []}
    angmin, angmax, rate_a = [], [], []

    for l, r in enumerate(branches):
        f, t = busidx[int(r[0])], busidx[int(r[1])]
        rr, xx, bb = r[2], r[3], r[4]
        y = 1.0 / complex(rr, xx)
        gg, bshunt = y.real, y.imag
        tap = r[8] if r[8] != 0.0 else 1.0
        shift = math.radians(r[9])
        tr, ti = tap * math.cos(shift), tap * math.sin(shift)
        ttm = tr * tr + ti * ti
        g_fr = g_to = 0.0
        b_fr = b_to = bb / 2.0
        rate = (r[5] / base) if r[5] > 0 else 1e3

        branch["f_idx"].append(l);        branch["t_idx"].append(nb + l)
        branch["f_bus"].append(f);        branch["t_bus"].append(t)
        branch["c1"].append((-gg * tr - bshunt * ti) / ttm)
        branch["c2"].append((-bshunt * tr + gg * ti) / ttm)
        branch["c3"].append((-gg * tr + bshunt * ti) / ttm)
        branch["c4"].append((-bshunt * tr - gg * ti) / ttm)
        branch["c5"].append((gg + g_fr) / ttm)
        branch["c6"].append((bshunt + b_fr) / ttm)
        branch["c7"].append(gg + g_to)
        branch["c8"].append(bshunt + b_to)
        branch["rate_a_sq"].append(rate ** 2)
        angmin.append(math.radians(r[11])); angmax.append(math.radians(r[12]))

    for l, r in enumerate(branches):                              # arcs: from, then to
        arc["i"].append(l); arc["bus"].append(busidx[int(r[0])])
        arc["rate_a"].append(branch["rate_a_sq"][l] ** 0.5)
    for l, r in enumerate(branches):
        arc["i"].append(nb + l); arc["bus"].append(busidx[int(r[1])])
        arc["rate_a"].append(branch["rate_a_sq"][l] ** 0.5)
    rate_a = list(arc["rate_a"])

    return dict(bus=bus, gen=gen, arc=arc, branch=branch, ref_buses=ref_buses,
                vmin=vmin, vmax=vmax, pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax,
                rate_a=rate_a, angmin=angmin, angmax=angmax, baseMVA=base)
