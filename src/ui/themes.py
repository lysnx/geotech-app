"""
Theme definitions for the Geotechnical Analysis TUI.
Professional color schemes and styling using Rich library.
"""

from rich.theme import Theme
from rich.style import Style

GEOTECH_THEME = Theme({
    "primary": "bold cyan",
    "secondary": "bold blue",
    "accent": "bold magenta",
    "success": "bold green",
    "warning": "bold yellow", 
    "error": "bold red",
    "info": "bold blue",
    "value": "bold white",
    "unit": "dim white",
    "label": "cyan",
    "fs_safe": "bold green",
    "fs_marginal": "bold yellow",
    "fs_critical": "bold red",
    "soil_clay": "yellow",
    "header": "bold white on blue",
    "menu_item": "cyan",
    "menu_selected": "bold white on cyan",
    "border": "blue",
    "title": "bold cyan",
    "subtitle": "dim cyan",
})

LOGO = '''
======================================================================
       GEOTECH ANALYSIS SYSTEM v2.0
       Plane Strain Consolidation with Mohr-Coulomb Criterion
======================================================================
'''

MENU_ICONS = {
    "new": "[1]",
    "load": "[2]",
    "settings": "[3]",
    "help": "[4]",
    "exit": "[5]",
    "soil": ">>",
    "foundation": ">>",
    "analysis": ">>",
    "results": ">>",
    "export": ">>",
}

def get_fs_style(fs_value):
    if fs_value > 1.5:
        return "fs_safe"
    elif fs_value > 1.0:
        return "fs_marginal"
    else:
        return "fs_critical"

def format_fs_display(fs_value):
    style = get_fs_style(fs_value)
    status = "SAFE" if fs_value > 1.5 else ("MARGINAL" if fs_value > 1.0 else "CRITICAL")
    return f"[{style}]{fs_value:.3f} {status}[/{style}]"
