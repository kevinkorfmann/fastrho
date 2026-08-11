"""Minimal, user-facing Sphinx configuration for fastrho."""

from __future__ import annotations

project = "fastrho"
author = "fastrho authors"
copyright = "2026, fastrho authors"
release = "0.1.1"
version = "0.1.1"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = [
    "_build",
    "_scripts",
    "_static",
    "data/**",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "furo"
html_title = "fastrho documentation"
html_short_title = "fastrho"
html_static_path = ["_static_public"]
html_css_files = ["custom.css"]
html_favicon = "_static_public/favicon.svg"
html_show_sphinx = False
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#116a9e",
        "color-brand-content": "#116a9e",
        "color-admonition-background": "#f4f9fc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fd0ee",
        "color-brand-content": "#8fd0ee",
        "color-admonition-background": "#152631",
    },
}
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
