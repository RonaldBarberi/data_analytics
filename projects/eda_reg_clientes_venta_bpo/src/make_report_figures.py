# -*- coding: utf-8 -*-
"""
Figuras resumen del EDA de ventas BPO (para el README).
El dataset es sintético (generado aleatoriamente), por lo que se espera ausencia de patrones.
Uso:  python src/make_report_figures.py
@author: Ronald Barberi
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parents[1] / "utils"))
from plot_style import apply_style, headline, save, SERIES, INK, INK_2, NEUTRAL  # noqa: E402

FIG = ROOT / "reports" / "figures"
COLS = ["operador", "canal_venta", "tipo_plan", "region", "resultado_llamada", "genero"]


def cargar():
    df = pd.read_csv(ROOT / "data" / "dataset_clientes_venta.zip", sep="|",
                     usecols=COLS + ["venta", "tiempo_llamada", "edad_cliente"])
    df["venta"] = (df["venta"] == "si").astype(int)
    return df


def volumen(df):
    v = df["canal_venta"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(v.index, v.values / 1000, color=SERIES[0], height=0.55)
    for i, x in enumerate(v.values):
        ax.text(x / 1000, i, f" {x / 1000:,.0f} mil", va="center", color=INK_2, fontsize=9)
    ax.set_xlabel("Contactos (miles)"); ax.grid(axis="y", visible=False)
    headline(ax, "Volumen por canal de venta", f"{len(df):,} contactos; el IVR concentra el 40%".replace(",", "."))
    save(fig, FIG / "01_volumen_por_canal.png")


def tasas(df):
    base = df["venta"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, c in zip(axes.flat, COLS):
        g = df.groupby(c)["venta"].agg(["mean", "count"]).sort_values("mean")
        ic = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
        ax.errorbar(g["mean"], range(len(g)), xerr=ic, fmt="o", ms=6, color=SERIES[0], elinewidth=1.5)
        ax.axvline(base, color=NEUTRAL, ls="--", lw=1.2)
        ax.set_yticks(range(len(g)), g.index, fontsize=8)
        ax.set_title(c, loc="left", fontsize=10); ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlim(base - 0.03, base + 0.03)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    fig.suptitle("Tasa de venta por categoría (± IC 95%)", x=0.01, ha="left", fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.935, f"Todas las categorías coinciden con la tasa global ({base:.1%}): "
             "el dataset sintético no contiene patrones", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, FIG / "02_tasa_venta_por_categoria.png")


if __name__ == "__main__":
    apply_style()
    data = cargar()
    volumen(data); tasas(data)
    print("ok", FIG)
