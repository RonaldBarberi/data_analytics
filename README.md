<div align="center">

# Data Analytics · Análisis Exploratorio de Datos

**Español** · [English](README.en.md)

Análisis exploratorios (EDA) que convierten datos crudos en datasets listos para modelar:
calidad de datos, imputación, outliers, reducción de ruido y validación estadística de cada variable.

![Python](https://img.shields.io/badge/Python-3.10-2a78d6?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-2a78d6?style=flat-square&logo=pandas&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-utils-2a78d6?style=flat-square&logo=apachespark&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Matplotlib-2a78d6?style=flat-square)

</div>

## Resumen

| Proyecto | Datos | Qué se hizo | Resultado |
|----------|-------|-------------|-----------|
| [Uso de tarjeta de crédito](#1-uso-de-tarjeta-de-crédito) | 1.548 clientes, 18 variables, 11% positivos | Imputación por coeficiente de variación, outliers por IQR, agrupación de ruido, Pearson, Spearman y Chi² | Dataset para el [modelo de clasificación](https://github.com/RonaldBarberi/data_science#1-propensión-de-uso-de-tarjeta-de-crédito) (ROC-AUC 0.74 en CV) |
| [Ventas BPO](#2-ventas-de-telemercadeo-bpo) | 500.000 contactos sintéticos, 23 variables | Tipado eficiente, control de cardinalidad, agrupación de interacciones, codificación ordinal | Dataset para el [modelo de propensión](https://github.com/RonaldBarberi/data_science#4-propensión-de-venta-validación-de-señal); se confirma que no hay señal |

---

## 1. Uso de tarjeta de crédito

[`projects/eda_clientes_uso_tarjeta`](projects/eda_clientes_uso_tarjeta) ·
[Notebook](projects/eda_clientes_uso_tarjeta/src/eda_xbt_clientes_uso_tarjeta.ipynb) ·
[![Colab](https://img.shields.io/badge/Abrir_en-Colab-eb6834?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/RonaldBarberi/data_analytics/blob/main/projects/eda_clientes_uso_tarjeta/src/eda_xbt_clientes_uso_tarjeta.ipynb)

**Proceso**
1. Unión de clientes y etiquetas, con tipado de columnas para reducir memoria.
2. Nulos: si la variable tiene coeficiente de variación > 30% se imputa con la mediana; si no, con la media.
3. Outliers por rango intercuartílico. Los filtros se aplican solo a la clase mayoritaria para no perder positivos.
4. Agrupación por percentiles en variables con mucho ruido.
5. Codificación e importancia de variables con Pearson, Spearman y Chi².

**Hallazgos**
- `Type_Occupation` tiene **31.5% de nulos**; las demás variables, menos del 2%.
- Los pensionados usan la tarjeta más que el promedio (15.6% frente a 11.3%) y los servidores públicos menos (5.2%).
- Quienes viven en apartamentos municipales tienen una tasa de 30.2%, el triple de la de casa propia (10.6%), aunque son solo 53 clientes.

<p align="center">
  <img src="projects/eda_clientes_uso_tarjeta/reports/figures/01_valores_nulos.png" width="70%" alt="Porcentaje de nulos por variable">
  <img src="projects/eda_clientes_uso_tarjeta/reports/figures/03_tasa_por_categoria.png" width="95%" alt="Tasa de uso por categoría">
  <img src="projects/eda_clientes_uso_tarjeta/reports/figures/02_distribuciones.png" width="95%" alt="Distribución de edad y años de empleo">
</p>

## 2. Ventas de telemercadeo (BPO)

[`projects/eda_reg_clientes_venta_bpo`](projects/eda_reg_clientes_venta_bpo) ·
[Notebook](projects/eda_reg_clientes_venta_bpo/src/eda_validator_data.ipynb)

**Contexto.** Son 500.000 contactos de una campaña simulada (operador, canal, plan, región, resultado de llamada…). Los datos fueron **generados aleatoriamente** para practicar el flujo completo con un volumen realista.

**Proceso**
- Tipado con `category` e `int8`/`int16` para reducir memoria.
- Eliminación de identificadores y fechas sin poder predictivo.
- Control de cardinalidad y agrupación de las llamadas (predictivo, blaster, IVR, SMS) en una sola variable de interacciones.
- Codificación ordinal y correlación de Pearson con la variable objetivo.

**Conclusión.** Todas las categorías tienen la misma tasa de venta (39.9%) dentro de su intervalo de confianza. Es el comportamiento esperado de datos aleatorios, y el [notebook de modelado](https://github.com/RonaldBarberi/data_science#4-propensión-de-venta-validación-de-señal) lo confirma con AUC 0.50.

<p align="center">
  <img src="projects/eda_reg_clientes_venta_bpo/reports/figures/01_volumen_por_canal.png" width="70%" alt="Volumen por canal">
  <img src="projects/eda_reg_clientes_venta_bpo/reports/figures/02_tasa_venta_por_categoria.png" width="95%" alt="Tasa de venta por categoría">
</p>

---

## Utilidades reutilizables

| Archivo | Contenido |
|---------|-----------|
| [`utils/cls_statistics_dt_scientist_rebr.py`](utils/cls_statistics_dt_scientist_rebr.py) | Funciones de EDA en **pandas y PySpark**: desbalance, mapa de nulos, imputación por coeficiente de variación, outliers, agrupación de ruido por percentiles y correlación e importancia (Pearson, Spearman, Chi²). |
| [`utils/cls_dt_engeerin.py`](utils/cls_dt_engeerin.py) | Configuración y sesión de Spark, conversión de pandas a PySpark y consultas SQL sobre RDD. |
| [`utils/plot_style.py`](utils/plot_style.py) | Estilo común de gráficas (paleta apta para daltonismo). |

## Cómo ejecutar

```bash
git clone https://github.com/RonaldBarberi/data_analytics.git
cd data_analytics/projects/eda_clientes_uso_tarjeta
pip install -r config/requirimients.txt
jupyter notebook src/eda_xbt_clientes_uso_tarjeta.ipynb
python src/make_report_figures.py      # regenera las figuras del README
```

---

<p align="center">
  <b>Ronald Barberi</b> · Data Scientist & Data Engineer ·
  <a href="https://www.linkedin.com/in/ronald-eduardo-barberi-ria%C3%B1o-rebr/">LinkedIn</a> ·
  <a href="https://github.com/RonaldBarberi">GitHub</a>
</p>
