"""Every example must import and build. A broken one shipped once, unnoticed."""
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


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_an_example_runs_end_to_end(path):
    out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                         cwd=path.parent, timeout=1800)
    assert out.returncode == 0, out.stderr[-800:]
    assert "objective" in out.stdout
