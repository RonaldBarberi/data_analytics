""""
@author: Ronald Barberi
create_at: 2025-11-23 10:21
"""

#%% Imported libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
from sklearn.feature_selection import chi2
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import (
    col, round as Fround, mean as Fmean, stddev_pop,
    min as Fmin, max as Fmax, when, lit
)

#%% Create Class

class statistics_dt_scientist_pd:

    @staticmethod
    def identificar_desbalanceo(
        df_in,
        nam_col: str
    ):
        print(
            df_in.groupby(nam_col)
                .agg(count=(nam_col, 'count'))
                .assign(percent=lambda x: (x['count'] / x['count'].sum()).round(4))
        )


    @staticmethod
    def heatmap_valores_null(
        df_in
    ):
        sns.heatmap(
            df_in.isnull(), cbar=False
        )
        plt.title('Mapa de Calor - Valores Faltantes')
        plt.show()
    

    @staticmethod
    def input_val_nulls_cof_vacn(
        df_in,
        nam_col: str,
        input_automatico:bool = False
    ):

        if not isinstance(input_automatico, bool):
            raise ValueError('[ERROR] input_automatico debe ser True o False.')
        
        vector_1d = df_in[nam_col].dropna()
        media = np.nanmean(vector_1d).round(4)
        mediana = np.nanmedian(vector_1d).round(4)
        desv_est = np.nanstd(vector_1d, ddof=0).round(4)
        cv = (desv_est / media).round(4)

        print(f'\n[OK] la media para {nam_col} es: {media}')
        print(f'[OK] la mediana para {nam_col} es: {mediana}')
        print(f'[OK] el coeficiente de variacion para {nam_col} es: {cv}')
        if cv < -0.3 or cv > 0.3:
            print(f'[INFO] Alta dispersión, mejor usar MEDIANA para representatividad.')
        
        sns.kdeplot(data=df_in[nam_col], fill=True,  linewidth=1)
        
        plt.axvline(media, color='red', linestyle='--', label=f'Media = {media:.4f}')
        plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana = {mediana:.4f}')
        
        plt.title(f'Distribución de {nam_col}\nCV = {cv:.4f}')
        plt.xlabel(nam_col)
        plt.ylabel('Densidad')
        plt.legend()
        plt.show()

        if input_automatico is True:
            if cv < -0.3 or cv > 0.3:
                mask = np.isnan(vector_1d)
                vector_1d[mask] = mediana
            else:
                mask = np.isnan(vector_1d)
                vector_1d[mask] = media

        return df_in, media, mediana, cv


    @staticmethod
    def manejo_statistics_outliers(
        df_in,
        nam_col: str,
        visual_ouliers:bool = False,
        clear_automatico:bool = False
    ):

        list_valid_bool = [clear_automatico, visual_ouliers]
        for item_valid in list_valid_bool:
            if not isinstance(item_valid, bool):
                raise ValueError(f'[ERROR] {item_valid} debe ser True o False.')
        
        if visual_ouliers is True:
            plt.figure(figsize=(5,5))
            sns.boxplot(
                df_in[nam_col],
                showmeans=True
            )
            plt.title(f'Boxplot por {nam_col}')
            plt.xlabel(nam_col)
            plt.ylabel('Frecuencia')
            plt.show()

        vector_1d = df_in[nam_col].dropna().to_numpy()

        valores_unicos = set(vector_1d)
        if valores_unicos.issubset({0, 1}):
            print(f'\n[INFO] Columna {nam_col} detectada como binaria. Se omite validación de outliers.')
            return df_in, -float('inf'), float('inf')
        
        else:
            q1 = np.quantile(vector_1d, 0.25)
            q3 = np.quantile(vector_1d, 0.75)

            iqr = q3 - q1
            li = q1 - 1.5 * iqr
            ls = q3 + 1.5 * iqr

            print(f'\n[INFO] Cuartiles para {nam_col}:')
            print(f'[INFO] Q1: {q1}')
            print(f'[INFO] Q3: {q3}')
            print(f'[INFO] IQR: {iqr}')
            print(f'[INFO] Rango aceptado: [{li}, {ls}]')
            cant_outliers = df_in.query(f'{nam_col} < @li or {nam_col} > @ls').shape[0]
            print(f'[INFO] Cantidad de outliers en {nam_col}: {cant_outliers}')

            if clear_automatico is True:
                df_in = df_in.query(f'{nam_col} >= @li or {nam_col} <= @ls')
                print('[OK] Se han excluido los outliers correctamente.')
            
            return df_in, li, ls


    @staticmethod
    def agrupar_ruido_percentiles(
        df_in,
        nam_col: str,
        print_statistics: bool = False
    ):
        
        vector_1d = df_in[nam_col].to_numpy()

        if print_statistics:
            print(f'\n[INFO] tipo de dato de {col} es: {df_in[nam_col].dtype}')
            print(f'[INFO] valor minimo es {np.nanmin(vector_1d)}')
            print(f'[INFO] valor maximo es {np.nanmax(vector_1d)}')

        percentiles = np.nanpercentile(
            vector_1d,
            [10,20,30,40,50,60,70,80,90,100]
        )
        
        p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = percentiles

        col_vec = vector_1d

        conditions = [
            (col_vec < p1),
            (col_vec >= p1) & (col_vec < p2),
            (col_vec >= p2) & (col_vec < p3),
            (col_vec >= p3) & (col_vec < p4),
            (col_vec >= p4) & (col_vec < p5),
            (col_vec >= p5) & (col_vec < p6),
            (col_vec >= p6) & (col_vec < p7),
            (col_vec >= p7) & (col_vec < p8),
            (col_vec >= p8) & (col_vec < p9),
            (col_vec >= p9) & (col_vec <= p10),
            (col_vec > p10),
        ]

        choices_visual = np.array([
            f'menor a {p1}',
            f'{p1}-{p2}',
            f'{p2}-{p3}',
            f'{p3}-{p4}',
            f'{p4}-{p5}',
            f'{p5}-{p6}',
            f'{p6}-{p7}',
            f'{p7}-{p8}',
            f'{p8}-{p9}',
            f'{p9}-{p10}',
            f'mayor a {p10}',
        ])

        choices_index = np.array([1,2,3,4,5,6,7,8,9,10,11])

        df_in[nam_col] = np.select(conditions, choices_visual, default='sin rango')
        df_in[nam_col + '_index_cod'] = np.select(conditions, choices_index, default=12).astype('int8')
    
        return df_in


    @staticmethod
    def correlacion_importancia_variables(
        df_in,
        col_y: str,
        print_statistics: bool = False
    ):
        df_num = df_in.select_dtypes(include=['number'])
        matriz_np = df_num.to_numpy()
        cols = df_num.columns

        corr_matriz_pearson = np.corrcoef(matriz_np, rowvar=False)

        def rank_matrix(matriz_np):
            return np.apply_along_axis(lambda x: x.argsort().argsort(), 0, matriz_np)

        ranked = rank_matrix(matriz_np)
        spearman_matrix = np.corrcoef(ranked, rowvar=False)

        dic_corr = {
            'Pearson': corr_matriz_pearson,
            'Spearman': spearman_matrix,
        }

        if print_statistics:
            for type_corr, matrix in dic_corr.items():
                df_matrix = pd.DataFrame(matrix, columns=cols, index=cols)
                plt.figure(figsize=(10, 10))
                sns.heatmap(df_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title(f'Correlación de {type_corr}')
                plt.show()

        # Chi-Cuacdrado
        X = df_in.drop(columns=[col_y])
        y = df_in[col_y]
        chi_scores, p_values = chi2(X, y)

        results = pd.DataFrame({
            'feature': X.columns,
            'chi2_score': chi_scores,
            'p_value': p_values
        })

        results = results.sort_values(by='chi2_score', ascending=False)
        print(results)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=results, x='chi2_score', y='feature')
        plt.title('Importancia por Chi-cuadrado')
        plt.show()


class statistics_dt_scientist_sp:

    @staticmethod
    def identificar_desbalanceo(
        df_in,
        nam_col: str
    ):
        (
            df_in.groupBy(nam_col).count()
                .withColumn('percent', Fround(col('count') / df_in.count(), 4))
        ).show()


    @staticmethod
    def heatmap_valores_null(
        df_in
    ):
        df_in = df_in.toPandas()

        sns.heatmap(
            df_in.isnull(), cbar=False
        )
        plt.title('Mapa de Calor - Valores Faltantes')
        plt.show()
    

    @staticmethod
    def input_val_nulls_cof_vacn(
        df_in,
        nam_col: str,
        print_grafic:bool = False,
        input_automatico:bool = False
    ):
        list_valid_bools = [print_grafic, input_automatico]
        for param_valid in list_valid_bools:
            if not isinstance(param_valid, bool):
                raise ValueError(f'[ERROR] {param_valid} debe ser True o False.')

        df_clean = df_in.filter(col(nam_col).isNotNull())
        media = round(df_clean.agg(Fmean(nam_col)).first()[0], 4)
        mediana = round(df_clean.approxQuantile(nam_col, [0.5], 1e-9)[0], 4)
        desv_est = round(df_clean.agg(stddev_pop(nam_col)).first()[0], 4)
        cv = cv = round(desv_est / media, 4) if media != 0 else None

        print(f'\n[OK] la media para {nam_col} es: {media}')
        print(f'[OK] la mediana para {nam_col} es: {mediana}')
        print(f'[OK] el coeficiente de variacion para {nam_col} es: {cv}')
        if cv < -0.3 or cv > 0.3:
            print(f'[INFO] Alta dispersión, mejor usar MEDIANA para representatividad.')
        
        if print_grafic:
            df_pd_tmp = df_in.toPandas()

            sns.kdeplot(data=df_pd_tmp[nam_col], fill=True,  linewidth=1)
            
            plt.axvline(media, color='red', linestyle='--', label=f'Media = {media:.4f}')
            plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana = {mediana:.4f}')
            
            plt.title(f'Distribución de {nam_col}\nCV = {cv:.4f}')
            plt.xlabel(nam_col)
            plt.ylabel('Densidad')
            plt.legend()
            plt.show()

        if input_automatico is True:
            if cv < -0.3 or cv > 0.3:
                df_in = df_in.fillna({nam_col: mediana})
            else:
                df_in = df_in.fillna({nam_col: media})

        return df_in, media, mediana, cv


    @staticmethod
    def manejo_statistics_outliers(
        df_in,
        nam_col,
        print_ouliers:bool = False,
        clear_automatico:bool = False
    ):
        list_valid_bool = [clear_automatico, print_ouliers]
        for item_valid in list_valid_bool:
            if not isinstance(item_valid, bool):
                raise ValueError(f'[ERROR] {item_valid} debe ser True o False.')
        
        if print_ouliers is True:

            df_pd_tmp = df_in.toPandas()

            plt.figure(figsize=(5,5))
            sns.boxplot(
                df_pd_tmp[nam_col],
                showmeans=True
            )
            plt.title(f'Boxplot por {nam_col}')
            plt.xlabel(nam_col)
            plt.ylabel('Frecuencia')
            plt.show()


        valores_no_binarios = (
            df_in
            .select(nam_col)
            .where(col(nam_col).isNotNull() & ~col(nam_col).isin(0, 1))
            .limit(1)
        )

        if valores_no_binarios.count() == 0:
            print(f'\n[INFO] Columna {nam_col} detectada como binaria. Se omite validación de outliers.')
            return df_in, -float('inf'), float('inf')
        
        else:
            q1, q3 = df_in.approxQuantile(nam_col, [0.25, 0.75], 0.01)

            iqr = q3 - q1
            li = q1 - 1.5 * iqr
            ls = q3 + 1.5 * iqr

            print(f'\n[INFO] Cuartiles para {nam_col}:')
            print(f'[INFO] Q1: {q1}')
            print(f'[INFO] Q3: {q3}')
            print(f'[INFO] IQR: {iqr}')
            print(f'[INFO] Rango aceptado: [{li}, {ls}]')

            cant_outliers = (
                df_in
                .filter((col(nam_col) < li) | (col(nam_col) > ls))
                .count()
            )
            print(f'[INFO] Cantidad de outliers en {nam_col}: {cant_outliers}')

            if clear_automatico is True:
                df_in = df_in.filter((col(nam_col) >= li) & (col(nam_col) <= ls))
                print('[OK] Se han excluido los outliers correctamente.')
            
            return df_in, li, ls


    @staticmethod
    def agrupar_ruido_percentiles(
        df_in,
        nam_col: str,
        print_statistics: bool = False
    ):

        if print_statistics:
            print(f'\n[INFO] tipo de dato de {col} es: {df_in.select(nam_col).printSchema()}')
            print(f'[INFO] valor minimo es {df_in.select(nam_col).Fmin().printSchema()}')
            print(f'[INFO] valor maximo es {df_in.select(nam_col).Fmax().printSchema()}')

        percentiles = df_in.approxQuantile(
                        nam_col, 
                        [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 
                        1e-9
                    )
        
        p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = percentiles

        c = col(nam_col)

        df_in = df_in.withColumn(
            nam_col,
            when(c < p1, f'menor a {p1}')
            .when((c >= p1) & (c < p2), f'{p1}-{p2}')
            .when((c >= p2) & (c < p3), f'{p2}-{p3}')
            .when((c >= p3) & (c < p4), f'{p3}-{p4}')
            .when((c >= p4) & (c < p5), f'{p4}-{p5}')
            .when((c >= p5) & (c < p6), f'{p5}-{p6}')
            .when((c >= p6) & (c < p7), f'{p6}-{p7}')
            .when((c >= p7) & (c < p8), f'{p7}-{p8}')
            .when((c >= p8) & (c < p9), f'{p8}-{p9}')
            .when((c >= p9) & (c <= p10), f'{p9}-{p10}')
            .otherwise(f'mayor a {p10}')
        )

        df_in = df_in.withColumn(
            nam_col + '_index_cod',
            when(c < p1, 1)
            .when((c >= p1) & (c < p2), 2)
            .when((c >= p2) & (c < p3), 3)
            .when((c >= p3) & (c < p4), 4)
            .when((c >= p4) & (c < p5), 5)
            .when((c >= p5) & (c < p6), 6)
            .when((c >= p6) & (c < p7), 7)
            .when((c >= p7) & (c < p8), 8)
            .when((c >= p8) & (c < p9), 9)
            .when((c >= p9) & (c <= p10), 10)
            .otherwise(11)
        )

        return df_in


    @staticmethod
    def correlacion_importancia_variables(
        df_in,
        col_y: str,
        print_statistics: bool = False
    ):
        num_cols = [c for c, t in df_in.dtypes if t in ('int', 'double', 'float', 'bigint')]
        df_num = df_in.select(num_cols)

        assembler = VectorAssembler(inputCols=num_cols, outputCol='features')
        df_vec = assembler.transform(df_num).select('features')

        corr_pearson = Correlation.corr(df_vec, 'features', 'pearson').head()[0].toArray()

        corr_spearman = Correlation.corr(df_vec, 'features', 'spearman').head()[0].toArray()

        dic_corr = {
            'Pearson': corr_pearson,
            'Spearman': corr_spearman
        }

        if print_statistics:
            for tipo, matrix in dic_corr.items():
                df_matrix = pd.DataFrame(matrix, columns=num_cols, index=num_cols)
                plt.figure(figsize=(10, 10))
                sns.heatmap(df_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title(f'Correlación de {tipo}')
                plt.show()

        df_pd_tmp = df_in.toPandas()

        X = df_pd_tmp.drop(columns=[col_y])
        y = df_pd_tmp[col_y]

        chi_scores, p_values = chi2(X.select_dtypes(include=[np.number]), y)

        results = pd.DataFrame({
            'feature': X.select_dtypes(include=[np.number]).columns,
            'chi2_score': chi_scores,
            'p_value': p_values
        })

        results = results.sort_values(by='chi2_score', ascending=False)
        print(results)

        if print_statistics:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=results, x='chi2_score', y='feature')
            plt.title('Importancia por Chi-cuadrado')
            plt.show()