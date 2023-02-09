# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../intel_extension_for_transformers/'))
import version as ver

version= ver.__version__
release = version

with open("version.txt", "w") as f:
    f.write(version)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Intel® Extension for Transformers'
copyright = '2022, Intel® Extension for Transformers, Intel'
author = 'Intel® Extension for Transformers developers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'recommonmark',
        'sphinx_markdown_tables',
        'sphinx.ext.coverage',
        'sphinx.ext.autosummary',
        'sphinx_md',
        'autoapi.extension',
        'sphinx.ext.napoleon',
        'sphinx.ext.githubpages',
        'breathe'
        ]

autoapi_dirs = ['../../intel_extension_for_transformers']
autoapi_root = "autoapi"
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autosummary_generate = True
autoapi_options = ['members',  'show-inheritance',
                   'show-module-summary', 'imported-members', ]
autoapi_ignore = []

templates_path = ['_templates']

highlight_language = 'c++'

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

exclude_patterns = ['_build_doxygen']

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

# -- Breathe configuration -------------------------------------------------
breathe_projects = {
	"Intel® Extension for Transformers": "../_build_doxygen/xml/"
}
breathe_default_project = "Intel® Extension for Transformers"
breathe_default_members = ('members', 'undoc-members')
