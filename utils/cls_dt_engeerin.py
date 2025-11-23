""""
@author: Ronald Barberi
create_at: 2025-10-31 12:50
"""

#%% Imported libraries

import os
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, round as Fround
)

#%% Create Class

class dt_engineer_spark:

    @staticmethod
    def config_entorno_spark():
        path_main = os.path.abspath(os.path.dirname(__file__))

        # os.environ['JAVA_HOME'] = r'C:\tools\jdk'
        # os.environ['PATH'] += os.pathsep + os.path.join(os.environ['JAVA_HOME'], 'bin')
        os.environ['JAVA_HOME'] = r'C:\Program Files\Eclipse Adoptium\jdk-17.0.17.10-hotspot'

        os.environ['HADOOP_HOME'] = r'C:\tools\hadoop'
        os.environ['hadoop.home.dir'] = r'C:\tools\hadoop'
        os.environ['PATH'] += os.pathsep + os.path.join(os.environ['HADOOP_HOME'], 'bin')

        # os.environ['PYSPARK_PYTHON'] = os.path.join(path_main, '..', 'venv', 'Scripts', 'python.exe')
        # os.environ['PYSPARK_DRIVER_PYTHON'] = os.path.join(path_main, '..', 'venv', 'Scripts', 'python.exe')
        os.environ['PYSPARK_PYTHON'] = os.path.join(path_main, '..', '..', 'venv_global', 'Scripts', 'python.exe')
        os.environ['PYSPARK_DRIVER_PYTHON'] = os.path.join(path_main, '..', '..', 'venv_global', 'Scripts', 'python.exe')
        
        os.environ['JAR_SQLSERVER'] = r'C:\tools\jdbc\sqlserver_jar\enu\jars\mssql-jdbc-12.8.1.jre11.jar'
        os.environ['PATH'] += os.pathsep + r'C:\tools\jdbc\sqlserver_jar\enu\auth\x64'
        os.environ['JAR_ORACLE'] = r'C:\tools\jdbc\oracle\ojdbc8.jar'        
        os.environ['JAR_POSTGRESQL'] = r'C:\tools\jdbc\postgresql\postgresql-42.7.5.jar'
        os.environ['JAR_MYSQL'] = r'C:\tools\jdbc\mysql\mysql.jar'
        os.environ['JAR_EXCEL'] = r'C:\tools\jdbc\excel\spark-excel_2.12-3.5.1_0.20.4.jar'

        if 'JAVA_HOME' in os.environ and 'HADOOP_HOME' in os.environ:
            print('[OK] config entorno correcta.')
        else:
            print('[ERROR] config entorno fallida.')


    @staticmethod
    def start_sesion_spark(
        name_sesion: str
    ):

        conf = SparkConf()
        conf.set('spark.sql.orc.write.batch.size', '100_000')
        conf.set('spark.sql.shuffle.partitions', '100')

        spark = SparkSession.builder \
            .appName(name_sesion) \
            .config('spark.jars', os.environ['JAR_POSTGRESQL']) \
            .config(conf=conf) \
            .enableHiveSupport() \
            .getOrCreate()

        return spark
    
    @staticmethod
    def read_pandas_to_pyspark(
        spark_sesion,
        path_file: str,
        cols_rename=None,
        delimiter_value=None,
        name_sheet=None
    ):
        with open(path_file, 'rb') as hdfs_file:
            if delimiter_value and name_sheet:
                df_pd = pd.read_excel(hdfs_file, sep=delimiter_value, sheet_name=name_sheet, dtype=str).fillna('')
            elif name_sheet:
                df_pd = pd.read_excel(hdfs_file, sheet_name=name_sheet, dtype=str).fillna('')
            else:
                df_pd = pd.read_excel(hdfs_file, dtype=str).fillna('')
            
            if cols_rename:
                df_ps = spark_sesion.createDataFrame(df_pd).toDF(*cols_rename)
            else:
                df_ps = spark_sesion.createDataFrame(df_pd)
            
            return df_ps
    

    @staticmethod
    def funCreateRDDToSQL(
        sparkSession: str,
        url: str,
        user: str,
        password: str,
        driver: str,
        pathQuery: str
    ):
        with open(pathQuery, 'r', encoding='Latin1') as file:
            sql_query = file.read()

        df = (
            sparkSession.read
                .format('jdbc')
                .option('url', url)
                .option('dbtable', f'({sql_query})')
                .option('user', user)
                .option('password', password)
                .option('driver', driver)
                .option('charset', 'Latin1')
                .option('encoding', 'Latin1')
                .load()
        )
        df.printSchema()
        return df


    @staticmethod
    def print_info_data(
        df_in,
        num_show: int = 5
    ):
        df_in.printSchema()
        df_in.show(num_show, truncate=False)
