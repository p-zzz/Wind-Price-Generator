"""Sphinx configuration for the DK1 scenario generator docs (setup mirrors WINPACT's)."""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Anchor on this file, not the CWD: sphinx-build runs from docs/ locally and from the
# repo root in CI.
_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_DOCS_DIR, "..")))

project = "DK1 Scenario Generator"
author = "Pablo Zan Nieto"
copyright = f"{datetime.now().year}, {author}"

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("dk1-scenario-generator")
except Exception:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Docstrings are NumPy style and use single backticks for cross-references.
default_role = "py:obj"
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# Understand "optional", "default x" and {"a", "b"} literal sets in type fields.
napoleon_preprocess_types = True
napoleon_type_aliases = {"Path": "pathlib.Path", "Config": "generator.config.Config"}

# NumPy type descriptions like "numpy.ndarray, shape (n_hours,)" contain words that
# aren't types; short names in signatures come from `from pathlib import Path` etc.
nitpick_ignore = [("py:class", "Path"), ("py:class", "Module"), ("py:class", "numpy.float32"), ("py:class", "float32")]
nitpick_ignore_regex = [("py:class", r"shape \(.*\)")]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "DK1 Scenario Generator"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False, "sticky_navigation": True}
