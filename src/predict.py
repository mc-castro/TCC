# Databricks notebook source
# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %pip install mlflow==1.24.0

# COMMAND ----------

import mlflow.pyfunc
from pyspark.sql.types import ArrayType, DoubleType, FloatType, StringType
from pyspark.sql.functions import struct, when, lit
from pyspark.sql.functions import *
from pyspark.sql import DataFrame
from pyspark.sql import *

from lightgbm import LGBMClassifier
import pandas as pd
import shap
import yaml

import re

class MakePredictionPipeline(object):

    def __init__(self, registered_model_name: str = None):
        self.registered_model_name = registered_model_name


    def load_data(self) -> DataFrame:
        # Leitura da base tratada e utilizada no treino
        
        data = spark.read.csv("dbfs:/FileStore/tables/feats.csv", header="true", inferSchema="true")
        futuro = spark.read.csv("dbfs:/FileStore/tables/base_futuro_tcc_csv.csv", header="true", inferSchema="true")
        
        futuro = futuro.select(data.columns)
        
        return futuro

    def load_model(self, stage: str = None):
        """
        This method will load the model in the desired stage. Accepts 
        None, Staging or Production.

        :param stage: desired model stage to be loaded
        :type stage: str

        :return: model UDF
        :rtype: spark_udf 
        """
        model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{self.registered_model_name}/{stage}", result_type=ArrayType(StringType()))

        return model_udf


    def make_predictions(self, data: DataFrame, model_udf):

        udf_inputs = struct(*(data.columns[1:-1]))

        data = data.withColumn("result", model_udf(udf_inputs))

        data = data.select(data.codigo_processo, data.cargo, data.result[0].alias("predict"), data.result[1].alias("probability"),data.result[2].alias("provas"),\
                    data.result[3].alias("value importance"), data.result[4].alias("impacto"))
         
        provas = re.findall(r"'(.*?)'", str(data.select(data.columns[4]).take(1)))
        value_importance = re.findall(r"\: (.*?)\,", str(data.select(data.columns[5]).take(1)))
        impacto = re.findall(r"'(.*?)'", str(data.select(data.columns[6]).take(1)))
        
        df = sqlContext.createDataFrame(zip(provas, value_importance, impacto), ['provas', 'value_importance', 'impacto'])
        data = data.select(data.codigo_processo, data.cargo, data.predict, data.probability)
 
        return data, df


    def write_predictions_to_datalake(self, predicted_data: DataFrame, evidences_data: DataFrame) -> None:
        #dbutils.fs.rm("dbfs:/FileStore/tables/predictions.csv", True)
        predicted_data.write.csv("dbfs:/FileStore/tables/predictions.csv", header="true")
        
        #dbutils.fs.rm("dbfs:/FileStore/tables/evidences.csv", True)
        evidences_data.write.csv("dbfs:/FileStore/tables/evidences.csv", header="true")

        return
        

    def run(self):
        # 1. LOAD DATA
        data = self.load_data()
        # 2. LOAD MODEL
        model_udf = self.load_model(stage = "None")
        # 3. MAKE PREDICTIONS
        df_preds, provas = self.make_predictions(data, model_udf)
        display (df_preds)
        # 4. WRITE PREDICTIONS TO DATALAKE
        #self.write_predictions_to_datalake(df_preds, provas)
        
        
if __name__ == "__main__":
    pipeline = MakePredictionPipeline("lgbm_tcc")
    pipeline.run()

# COMMAND ----------

dbutils.fs.rm("dbfs:/FileStore/tables/base_futuro_tcc_csv.csv", True)

# COMMAND ----------

dbutils.fs.rm("dbfs:/FileStore/tables/base_tcc_csv.csv", True)

# COMMAND ----------

