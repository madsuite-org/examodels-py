"""P0 scenario B: user-visible cold breakdown + what a warm daemon could serve.
Fresh process. Model: Luksan-Vlcek at N=100_000 on backend="cuda" (the size
docs/tutorial/gpu.md quotes 0.75s warm for), objective carrying a parameter
block so the set_parameters leg is measurable.

Warm-process legs at the end:
  - re-solve            = live-instance hit, same data
  - set_parameters+solve = live-instance hit, new parameter values
  - full rebuild+solve  = replay-per-solve cost, same expression types
  - new-structure model = marginal cost of an unseen fingerprint in a warm
                          daemon (iterations capped: JIT overhead is the datum,
                          not solve quality)
"""
import time

T0 = time.perf_counter()
LAST = [T0]


def mark(label):
    now = time.perf_counter()
    print(f"{label:38s} {now - LAST[0]:8.2f}s  (cum {now - T0:7.2f}s)", flush=True)
    LAST[0] = now


import examodels as exa  # noqa: E402

mark("import examodels")

N = 100_000
start = [-1.2 if i % 2 == 0 else 1.0 for i in range(N)]

core = exa.Core(backend="cuda")
mark("Core(backend='cuda')")

x = core.add_var(N, start=start)
p = core.add_par([1.0] * N)
core.add_obj(lambda i: 100 * (x[i - 1] ** 2 - x[i]) ** 2 + (x[i - 1] - p[i]) ** 2,
             over=range(1, N))
core.add_con(lambda i: 3 * x[i + 1] ** 3 + 2 * x[i + 2] - 5
             + exa.sin(x[i + 1] - x[i + 2]) * exa.sin(x[i + 1] + x[i + 2])
             + 4 * x[i + 1] - x[i] * exa.exp(x[i] - x[i + 1]) - 3,
             over=range(0, N - 2))
mark("trace expressions")

m = exa.Model(core)
mark("Model(core)")

s1 = m.solve()
mark(f"solve #1 cold ({s1.status})")

s2 = m.solve()
mark(f"solve #2 warm ({s2.status})")

m.set_parameters(p, [1.001] * N)
s3 = m.solve()
mark(f"set_parameters + solve ({s3.status})")

core2 = exa.Core(backend="cuda")
x2 = core2.add_var(N, start=start)
p2 = core2.add_par([1.0] * N)
core2.add_obj(lambda i: 100 * (x2[i - 1] ** 2 - x2[i]) ** 2 + (x2[i - 1] - p2[i]) ** 2,
              over=range(1, N))
core2.add_con(lambda i: 3 * x2[i + 1] ** 3 + 2 * x2[i + 2] - 5
              + exa.sin(x2[i + 1] - x2[i + 2]) * exa.sin(x2[i + 1] + x2[i + 2])
              + 4 * x2[i + 1] - x2[i] * exa.exp(x2[i] - x2[i + 1]) - 3,
              over=range(0, N - 2))
m2 = exa.Model(core2)
s4 = m2.solve()
mark(f"rebuild same structure + solve ({s4.status})")

core3 = exa.Core(backend="cuda")
x3 = core3.add_var(N, start=1.0)
core3.add_obj(lambda i: (x3[i] - 2) ** 4 + exa.cos(x3[i]), over=range(N))
core3.add_con(lambda i: exa.tanh(x3[i]) + x3[i + 1] ** 2, over=range(N - 1),
              lcon=-10.0, ucon=10.0)
m3 = exa.Model(core3)
s5 = m3.solve(max_iter=3)
mark(f"NEW structure build + solve, max_iter=3 ({s5.status})")
