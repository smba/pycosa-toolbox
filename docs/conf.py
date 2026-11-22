"""Sphinx configuration for the pycosa API documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pycosa"
copyright = "Stefan Mühlbauer"
author = "Stefan Mühlbauer"

release = "0.2.0"
version = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

html_theme = "furo"
html_title = "pycosa"

exclude_patterns = ["_build"]
