# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SAR Geometry'
copyright = 'ONERA/DEMR'
author = 'Olivier LEVEQUE'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx_subfigure",
    "sphinx_design",
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "numpydoc",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = "SARGeom"
html_static_path = ['_static']

# -- Options for Numpydoc ----------------------------------------------------
#
numpydoc_show_class_members = False

# -- Options for TODOs -------------------------------------------------------
#
todo_include_todos = True

# -- Options for Intersphinx -------------------------------------------------
#
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None)
}