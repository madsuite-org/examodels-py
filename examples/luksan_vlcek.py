"""The Luksan-Vlcek problem, written in Python, solved with Ipopt.

    min  sum_{i=2..N} 100 (x_{i-1}^2 - x_i)^2 + (x_{i-1} - 1)^2
    s.t. 3x_{i+1}^3 + 2x_{i+2} - 5 + sin(x_{i+1}-x_{i+2}) sin(x_{i+1}+x_{i+2})
         + 4x_{i+1} - x_i exp(x_i - x_{i+1}) - 3 = 0,   i = 1..N-2
"""
import examodels as exa

N = 10

core = exa.Core()
x = core.add_var(N, start=[-1.2 if i % 2 == 0 else 1.0 for i in range(N)])

core.add_obj(lambda i: 100 * (x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2,
           over=range(1, N))

core.add_con(lambda i: 3 * x[i+1]**3 + 2 * x[i+2] - 5
            + exa.sin(x[i+1] - x[i+2]) * exa.sin(x[i+1] + x[i+2])
            + 4 * x[i+1] - x[i] * exa.exp(x[i] - x[i+1]) - 3,
            over=range(0, N - 2), lower=0.0, upper=0.0)

model = exa.Model(core)
print(problem)

sol = problecore.solve(solver="ipopt")
print(f"status     : {sol.status}")
print(f"objective  : {sol.objective:.10f}")
print(f"iterations : {sol.iterations}")
print(f"solve time : {sol.elapsed:.3f} s")
print(f"x          : {sol[x]}")
print(f"max |c(x)| : {abs(problecore.constraints(sol.x)).max():.2e}")
