"""AC optimal power flow: records as index sets, field access, constraint augmentation."""
import pathlib
import sys

import pytest

import examodels as exa

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))

PGLIB = pathlib.Path("/home/sushin/git/pglib-opf")
pglib = pytest.mark.skipif(not PGLIB.is_dir(), reason="PGLib cases not available here")


@pglib
def test_case3_matches_the_published_objective():
    """The case file header states the optimal value; we must reproduce it."""
    import matpower
    from ac_opf import ac_opf

    case = PGLIB / "pglib_opf_case3_lmbd.m"
    published = float(next(line.split(":")[1].split()[0]
                           for line in case.read_text().splitlines()
                           if "opt objective value" in line))
    assert published == pytest.approx(5812.64)

    core, var = ac_opf(matpower.read(str(case)))
    model = exa.Model(core)
    sol = model.solve()
    assert sol.success, sol.status
    assert sol.objective == pytest.approx(published, rel=1e-5)
    assert model.violation(sol.x) < 1e-6


@pglib
def test_case118_solves_feasibly():
    import matpower
    from ac_opf import ac_opf

    core, var = ac_opf(matpower.read(str(PGLIB / "pglib_opf_case118_ieee.m")))
    model = exa.Model(core)
    assert model.nvar == 1088 and model.ncon == 1539
    sol = model.solve()
    assert sol.success and model.violation(sol.x) < 1e-6
    vm = sol[var["vm"]]
    assert vm.min() > 0.9 and vm.max() < 1.11


def test_augmentation_adds_terms_to_existing_rows():
    """add_con(handle, ...) must add into rows, not create new ones."""
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    y = core.add_var(3, start=1.0)
    rows = exa.Records({"i": [0, 1, 2], "c": [1.0, 1.0, 1.0]}, index=["i"])
    con = core.add_con(lambda r: r.c + x[r.i], over=rows)
    core.add_con(con, lambda r: (r.i, y[r.i]), over=rows)
    model = exa.Model(core)
    assert model.ncon == 3, "augmentation must not add constraint rows"
    # each row is c + x + y = 1 + 1 + 1
    import numpy as np
    assert np.allclose(model.constraints(np.ones(6)), 3.0)
