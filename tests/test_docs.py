"""Every Python block in the manual is executed, in order, per page.

The examples had drifted into fragments -- `T`, `core` and `x` used without ever
being defined -- so a reader who typed one in got a `NameError` rather than a
model. Only 11 of 38 blocks ran when this was first measured.

Each page is executed as a whole, top to bottom, in one namespace: that is how a
reader meets it, and it lets a page introduce a model once and build on it,
rather than repeating the setup in every block.

A block that cannot run here says so in the source, in an HTML comment directly
above it (invisible in the rendered page):

    <!-- not-tested: needs a CUDA device and cupy -->

which is a claim about the environment, not a licence to be wrong: the reason is
printed by this test, so a page cannot go quiet by marking everything.
"""
import pathlib
import re

import pytest

DOCS = pathlib.Path(__file__).parents[1] / "docs"

#: a fenced block, and whatever HTML comment immediately precedes it
BLOCK = re.compile(r"(?:<!--\s*not-tested:\s*(?P<why>[^>]*?)\s*-->\s*\n)?"
                   r"^```(?P<lang>\w*)\n(?P<code>.*?)^```", re.S | re.M)


def _pages():
    return sorted(p for p in DOCS.rglob("*.md"))


def _blocks(page):
    """The python blocks of a page: (index, code, why_not_tested)."""
    out = []
    for i, m in enumerate(BLOCK.finditer(page.read_text())):
        if m.group("lang") == "python":
            out.append((i, m.group("code"), m.group("why")))
    return out


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_the_examples_on_this_page_run(page, capsys):
    blocks = _blocks(page)
    if not blocks:
        pytest.skip("no Python examples")

    ns = {}
    ran = skipped = 0
    for i, code, why in blocks:
        if why:
            skipped += 1
            with capsys.disabled():
                print(f"\n  {page.relative_to(DOCS)} block {i}: not tested -- {why}")
            continue
        try:
            exec(compile(code, f"{page.relative_to(DOCS)}#{i}", "exec"), ns)
        except Exception as e:                                   # noqa: BLE001
            first = code.strip().splitlines()[0]
            raise AssertionError(
                f"{page.relative_to(DOCS)}, block {i} raised {type(e).__name__}: {e}\n"
                f"  the block starts: {first}\n"
                f"  Examples run in page order in one namespace, so a name this "
                f"block needs must be introduced by this page. If it genuinely "
                f"cannot run here, say why with an HTML comment above it:\n"
                f"  <!-- not-tested: needs a CUDA device -->"
            ) from None
        ran += 1
    assert ran or skipped


def test_most_of_the_manual_is_actually_executed():
    """A page-level gate cannot notice the whole manual going un-run.

    Marking a block `not-tested` is one comment, so the cheapest way to make
    this suite pass would be to mark everything. Counting keeps that visible.
    """
    ran = skipped = 0
    for page in _pages():
        for _i, _code, why in _blocks(page):
            if why:
                skipped += 1
            else:
                ran += 1
    total = ran + skipped
    assert total, "no Python examples found at all -- has the fence syntax changed?"
    assert ran / total >= 0.6, (
        f"only {ran} of {total} documented examples are executed; the rest are "
        f"marked not-tested. Prefer making an example runnable to excusing it.")
