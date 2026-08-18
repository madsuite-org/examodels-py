"""P0 scenario A: raw cold-stack decomposition, independent of examodels' surface.
Fresh process, warm Julia depot, timing each phase once — boot/load/JIT phases
are one-shot by nature; run-to-run spread is checked by running the script twice."""
import time

T0 = time.perf_counter()
LAST = [T0]


def mark(label):
    now = time.perf_counter()
    print(f"{label:38s} {now - LAST[0]:8.2f}s  (cum {now - T0:7.2f}s)", flush=True)
    LAST[0] = now


from juliacall import Main  # noqa: E402

mark("import juliacall (Julia boot)")
Main.seval("using ExaModels")
mark("using ExaModels")
Main.seval("using NLPModelsIpopt")
mark("using NLPModelsIpopt")
Main.seval("using MadNLP")
mark("using MadNLP")
Main.seval("using CUDA")
mark("using CUDA")
Main.seval("using CUDSS, MadNLPGPU")
mark("using CUDSS, MadNLPGPU")
Main.seval("sum(CUDA.zeros(Float64, 8))")
mark("first CUDA op (context init)")
Main.seval("sum(CUDA.zeros(Float64, 8))")
mark("second CUDA op (warm)")
