"""P0 scenario C: a realistic model — AC OPF from pglib, backend="cuda".
Fresh process. case1354_pegase cold, then case2869_pegase in the SAME process:
the second case has a different fingerprint (different index sets/data) but
identical expression types, so it measures what a warm daemon pays for a new
case of a known model family — the reuse that matters for real workflows."""
import pathlib
import sys
import time

T0 = time.perf_counter()
LAST = [T0]


def mark(label):
    now = time.perf_counter()
    print(f"{label:44s} {now - LAST[0]:8.2f}s  (cum {now - T0:7.2f}s)", flush=True)
    LAST[0] = now


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples"))
import matpower  # noqa: E402

import examodels as exa  # noqa: E402,F401
from ac_opf import ac_opf  # noqa: E402

mark("imports (pure python)")

PGLIB = pathlib.Path.home() / "git/pglib-opf"

data = matpower.read(str(PGLIB / "pglib_opf_case1354_pegase.m"))
mark("read case1354_pegase")

core, _var = ac_opf(data, backend="cuda")
mark("trace + Core (Julia boot lands here)")

m = exa.Model(core)
mark("Model(core)")

s1 = m.solve()
mark(f"solve #1 cold ({s1.status})")

s2 = m.solve()
mark(f"solve #2 warm ({s2.status})")

data2 = matpower.read(str(PGLIB / "pglib_opf_case2869_pegase.m"))
core2, _ = ac_opf(data2, backend="cuda")
m2 = exa.Model(core2)
mark("case2869: read + trace + Model")

s3 = m2.solve()
mark(f"case2869 solve, warm process ({s3.status})")
