# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
project = "EICViBE"
copyright = "2025, EICViBE Team"
author = "EICViBE Team"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",           # Markdown support
    "autodoc2",              # Modern autodoc for Pydantic
    "sphinx.ext.viewcode",   # Add source links
    "sphinx.ext.intersphinx",# Link to external docs
    "sphinx_copybutton",     # Copy button for code blocks
]

# MyST configuration
myst_enable_extensions = [
    "colon_fence",      # ::: directive syntax
    "deflist",          # Definition lists
    "fieldlist",        # Field lists
    "dollarmath",       # $math$ and $$math$$
    "tasklist",         # - [x] task lists
]
myst_heading_anchors = 3

# Autodoc2 configuration
autodoc2_packages = [
    {
        "path": "../src/eicvibe",
        "auto_mode": False,  # Manual control over what to document
    }
]
autodoc2_render_plugin = "myst"

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xsuite": ("https://xsuite.readthedocs.io/en/latest/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "EICViBE"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2980b9",
        "color-brand-content": "#2980b9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#56b4e9",
        "color-brand-content": "#56b4e9",
    },
}

# -- Source file patterns ----------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
root_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
