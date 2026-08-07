"""AC optimal power flow: records as index sets, field access, constraint augmentation."""
import os
import pathlib
import sys
from collections import namedtuple

import pytest

import examodels as exa

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))

PGLIB = pathlib.Path(os.environ.get("PGLIB_DIR", "~/git/pglib-opf")).expanduser()
pglib = pytest.mark.skipif(not PGLIB.is_dir(), reason="PGLib cases not available here")


def cases(include_large=False):
    import tomllib
    table = tomllib.loads((pathlib.Path(__file__).parent / "data" / "cases.toml")
                          .read_text())["case"]
    return [c for c in table if include_large or not c.get("large")]


@pglib
@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_a_case_reproduces_its_expected_objective(case):
    """Both interfaces read this table, so a divergence between them fails here."""
    import matpower
    from ac_opf import ac_opf

    path = PGLIB / f"{case['name']}.m"
    data = matpower.read(str(path))
    assert len(data["bus"]) == case["buses"]
    assert len(data["gen"]) == case["generators"]
    assert len(data["branch"]) == case["branches"]

    if case["source"] == "published":
        # the file states it itself; read it rather than trusting the table
        stated = float(next(line.split(":")[1].split()[0]
                            for line in path.read_text().splitlines()
                            if "opt objective value" in line))
        assert stated == pytest.approx(case["objective"], rel=1e-6), \
            "the table disagrees with the case file's own header"

    core, var = ac_opf(data)
    model = exa.Model(core)
    sol = model.solve()
    assert sol.success, sol.status
    assert sol.objective == pytest.approx(case["objective"], rel=case["rtol"])
    assert model.violation(sol.x) < 1e-6
    vm = sol[var["vm"]]
    assert 0.5 < vm.min() and vm.max() < 1.5


def test_augmentation_adds_terms_to_existing_rows():
    """add_con(handle, ...) must add into rows, not create new ones."""
    core = exa.Core()
    x = core.add_var(3, start=1.0)
    y = core.add_var(3, start=1.0)
    Row = namedtuple("Row", "i c")
    rows = [Row(k, 1.0) for k in range(3)]
    con = core.add_con(r.c + x[r.i] for r in rows)
    core.add_con(con, ((r.i, y[r.i]) for r in rows))
    model = exa.Model(core)
    assert model.ncon == 3, "augmentation must not add constraint rows"
    # each row is c + x + y = 1 + 1 + 1
    import numpy as np
    assert np.allclose(model.constraints(np.ones(6)), 3.0)


def test_an_index_set_of_dicts_is_refused_clearly():
    """The obvious wrong thing to reach for must say so, not fail further along."""
    core = exa.Core()
    x = core.add_var(2, start=1.0)
    with pytest.raises(TypeError, match="no field order"):
        core.add_obj(lambda r: x[r["i"]], over=[{"i": 0}, {"i": 1}])


def test_a_generator_over_a_table_recovers_it():
    Row = namedtuple("Row", "i c")
    rows = [Row(k, float(k)) for k in range(4)]
    core = exa.Core()
    x = core.add_var(4, start=1.0)
    core.add_obj(r.c * x[r.i]**2 for r in rows)
    m = exa.Model(core)
    assert m.nvar == 4
    assert m.objective([1.0] * 4) == pytest.approx(0 + 1 + 2 + 3)
