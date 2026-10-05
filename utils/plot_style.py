"""Estilo común de gráficas del portafolio (Ronald Barberi).

Paleta categórica validada para daltonismo (orden fijo, no reciclar):
azul, naranja, aqua. Texto siempre en tinta neutra, nunca en color de serie.
Uso:
    from plot_style import apply_style, SERIES, INK, save
    apply_style()
"""
from pathlib import Path

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # texto principal
INK_2 = "#52514e"        # texto secundario
MUTED = "#898781"        # ejes y etiquetas
GRID = "#e1e0d9"         # cuadrícula
BASELINE = "#c3c2b7"     # línea base
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # slots 1-3 (válidos en todos los pares)
NEUTRAL = "#c3c2b7"      # referencia / línea base del modelo


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def headline(ax, title, sub=None):
    """Título a la izquierda y, debajo, un subtítulo gris con la conclusión."""
    ax.set_title(title, pad=20 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
