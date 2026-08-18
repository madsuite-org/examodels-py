# Cold-start measurements behind `../daemon.md`

Three standalone scripts, each run as a **fresh process** from the repo's
venv on the group's GPU box (2× Quadro GV100, warm Julia depot, package at
`d3b7a22`). `raw_stack.py` and `user_level.py` were run twice each;
phase-timing spread between runs was under 5%.

- `raw_stack.py` — the fixed stack cost, no model: Julia boot, each `using`,
  first CUDA operation. ≈ 60 s to a live CUDA context.
- `user_level.py` — user-visible phases for Lukšan–Vlček N=100 000 on
  `backend="cuda"`: cold 244 s to first solution, warm re-solve 0.43 s.
  The legs after the warm re-solve exercise `set_parameters` and rebuilds;
  the `set_parameters` leg currently dies with `ModelError: Scalar indexing
  is disallowed` — the device bug described in the design doc.
- `opf.py` — AC OPF realism (needs pglib, `PGLIB_DIR` or `~/git/pglib-opf`):
  `case1354_pegase` cold 308 s / warm re-solve 0.71 s, then
  `case2869_pegase` in the same process: 8.5 s total despite being a
  different fingerprint — JIT reuse is by expression type, which is what a
  warm daemon inherits.
