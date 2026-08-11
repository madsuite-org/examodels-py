"""Every example must import and build. A broken one shipped once, unnoticed."""
import os
import pathlib
import runpy
import subprocess
import sys

import pytest

EXAMPLES = sorted(p for p in (pathlib.Path(__file__).parents[1] / "examples").glob("*.py")
                  if p.stem != "matpower")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_an_example_imports_cleanly(path):
    """Catches a rename that mangled a name — which is how the last one broke."""
    sys.path.insert(0, str(path.parent))
    try:
        compile(path.read_text(), str(path), "exec")
        runpy.run_path(str(path), run_name="not_main")
    finally:
        sys.path.remove(str(path.parent))


NEEDS_DATA = {"ac_opf": pathlib.Path(
    os.environ.get("PGLIB_DIR", pathlib.Path.home() / "git/pglib-opf"))}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_an_example_runs_end_to_end(path):
    needed = NEEDS_DATA.get(path.stem)
    if needed is not None and not needed.is_dir():
        pytest.skip(f"{path.stem} needs benchmark data at {needed}")
    out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                         cwd=path.parent, timeout=1800)
    assert out.returncode == 0, out.stderr[-800:]
    assert "objective" in out.stdout
