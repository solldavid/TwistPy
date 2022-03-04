# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# -- Path setup --------------------------------------------------------------
from sphinx_gallery.sorting import ExampleTitleSortKey

sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    # 'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    "sphinx.ext.intersphinx",
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "obspy": ("https://docs.obspy.org/", None),
}

exclude_patterns = ['_build', '**.ipynb_checkpoints']
source_suffix = '.rst'
# The encoding of source files.
master_doc = 'index'

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ['png']

## Generate autodoc stubs with summaries from code
autosummary_generate = True

## Include Python objects as they appear in source files
autodoc_member_order = 'bysource'

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': ['../../tutorials', '../../examples', ],
    # path where to save gallery generated examples
    'gallery_dirs': ['tutorials', 'examples'],
    'filename_pattern': '\.py',
    'ignore_pattern': '_init_/',
    # Remove the "Download all examples" button from the top level gallery
    'download_all_examples': False,
    # Sort gallery example by file name instead of number of lines (default)
    'within_subsection_order': ExampleTitleSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': 'api/generated/backreferences',
    # Modules for which function level galleries are created.
    'doc_module': 'twistpy',
    # Insert links to documentation of objects in the examples
    'reference_url': {'twistpy': None},
    'first_notebook_cell': ("%matplotlib inline"),
    'run_stale_examples': True,
}

## Default flags used by autodoc directives
autodoc_default_flags = ['members']

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = False

html_theme_options = {
    "repository_url": "https://github.com/solldavid/TwistPy",
    "use_repository_button": True,
    "logo_only": True,
}

# -- Project information -----------------------------------------------------
year = datetime.date.today().year
project = 'TwistPy'
copyright = f'{year}, David Sollberger'

# The full version, including alpha/beta/rc tags
release = '0.1'

html_last_updated_fmt = '%b %d, %Y'
html_title = 'TwistPy'
html_short_title = 'TwistPy'
html_logo = '_static/logo_textbelow_oneaxis_adobe.png'
html_favicon = '_static/twistpy.ico'
html_static_path = ['_static']
html_extra_path = []
pygments_style = 'default'
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    'doc_path': 'docs/source',
    'galleries': sphinx_gallery_conf['gallery_dirs'],
    'gallery_dir': dict(zip(sphinx_gallery_conf['gallery_dirs'],
                            sphinx_gallery_conf['examples_dirs'])),
    'github_project': 'TwistPy',
    'github_repo': 'twistpy',
    'github_version': 'master',
}


# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_css_file("style.css")
    app.registry.source_suffix.pop(".ipynb", None)
