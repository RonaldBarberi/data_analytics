# -*- coding: utf-8 -*-
"""
Figuras resumen del EDA de uso de tarjeta (para el README).
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


def cargar():
    df = pd.read_csv(ROOT / "data" / "tj_clients.csv").merge(
        pd.read_csv(ROOT / "data" / "tj_label.csv"), on="Ind_ID")
    df["edad_anios"] = -df["Birthday_count"] / 365.25
    df["anios_empleo"] = np.where(df["Employed_days"] == 365243, np.nan, -df["Employed_days"] / 365.25)
    return df


def calidad(df):
    nulos = (df.isna().mean().loc[lambda s: s > 0].sort_values() * 100)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(nulos.index, nulos.values, color=SERIES[0], height=0.55)
    for i, v in enumerate(nulos.values):
        ax.text(v, i, f" {v:.1f}%", va="center", color=INK_2, fontsize=9)
    ax.set_xlabel("% de registros nulos"); ax.grid(axis="y", visible=False)
    headline(ax, "Calidad de datos", "Type_Occupation requiere imputación; el resto tiene < 2% de nulos")
    save(fig, FIG / "01_valores_nulos.png")


def distribuciones(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, xl in zip(axes, ["edad_anios", "anios_empleo"], ["Edad (años)", "Años de empleo"]):
        bins = np.histogram_bin_edges(df[col].dropna(), bins=25)
        for lab, color, nombre in [(0, SERIES[0], "No usa"), (1, SERIES[1], "Usa")]:
            ax.hist(df.loc[df["label"] == lab, col].dropna(), bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=nombre)
        ax.set_xlabel(xl); ax.set_yticks([]); ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper right")
    headline(axes[0], "Distribución por grupo", "Densidad normalizada: compara formas, no tamaños")
    headline(axes[1], " ", "Quienes usan la tarjeta tienden a tener menos años de empleo")
    fig.tight_layout(); save(fig, FIG / "02_distribuciones.png")


def tasas(df):
    cols = ["Type_Income", "EDUCATION", "Marital_status", "Housing_type"]
    base = df["label"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, c in zip(axes.flat, cols):
        g = df.groupby(c)["label"].agg(["mean", "count"]).query("count >= 15").sort_values("mean")
        ic = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
        ax.errorbar(g["mean"], range(len(g)), xerr=ic, fmt="o", ms=6, color=SERIES[0], elinewidth=1.5)
        ax.axvline(base, color=NEUTRAL, ls="--", lw=1.2)
        ax.set_yticks(range(len(g)), [f"{i} (n={n})" for i, n in zip(g.index, g["count"])], fontsize=8)
        ax.set_title(c, loc="left", fontsize=10); ax.grid(axis="y", visible=False)
    axes[1, 0].xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    axes[1, 1].xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    fig.suptitle("Tasa de uso de tarjeta por categoría (± IC 95%)", x=0.01, ha="left", fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.935, f"Línea gris: tasa global ({base:.1%}). Solo categorías con al menos 15 clientes.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, FIG / "03_tasa_por_categoria.png")


if __name__ == "__main__":
    apply_style()
    data = cargar()
    calidad(data); distribuciones(data); tasas(data)
    print("ok", FIG)
