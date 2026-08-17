"""Sphinx configuration. The page structure mirrors the ExaModels.jl manual."""
project = "examodels"
copyright = "Sungho Shin"
author = "Sungho Shin"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# `dollarmath` is what makes `$...$` and `$$...$$` into math at all: without
# it MyST leaves them as text, no math nodes are produced, and Sphinx never
# loads MathJax -- so the formula renders as its own source. `amsmath` is
# for the environments (align, cases) that a longer derivation reaches for.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
source_suffix = {".md": "markdown"}
master_doc = "index"

# Pages serves `dirhtml` output, so a page is `/install/` rather than
# `/install.html`; the canonical URL has to be written the same way, and
# both CI and the deploy build with `-b dirhtml` so they cannot diverge.
html_baseurl = "https://madsuite.org/examodels-py/"

html_theme = "furo"
html_title = "examodels"

autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_mock_imports = ["juliacall", "juliapkg", "cupy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

nitpicky = False
