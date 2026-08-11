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

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".md": "markdown"}
master_doc = "index"

html_theme = "furo"
html_title = "examodels"

autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_mock_imports = ["juliacall", "juliapkg", "cupy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

nitpicky = False
