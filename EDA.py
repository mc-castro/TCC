# Databricks notebook source
# MAGIC %pip install pandas_profiling

# COMMAND ----------

# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %pip install ipywidgets

# COMMAND ----------

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../src')
#from eda import make_subplots, get_correlations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn import FunctionSampler

# COMMAND ----------

pd.set_option('max_columns', None)
%matplotlib inline
sns.set_style('darkgrid')

# COMMAND ----------

df = spark.read.csv("dbfs:/FileStore/tables/base_tcc_csv.csv", header="true", inferSchema="true")
df = df.toPandas()
df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

profile = ProfileReport(df, title='Report', correlations={
                                                "pearson": {"calculate": False},
                                                "spearman": {"calculate": False},
                                                "kendall": {"calculate": False},
                                                "phi_k": {"calculate": False},
                                                "cramers": {"calculate": False}},
                                             missing_diagrams={
                                                "heatmap": False,
                                                "dendrogram": False,
                                                "matrix": False,
                                                "bar": False},
                                             interactions={
                                                "continuous": False})

# COMMAND ----------

profile.to_file("report.html")

# COMMAND ----------

df['resultado_processo'].value_counts()

# COMMAND ----------

# remove colunas com classes muito dominantes
def drop_low_variance(col):
    if df[col].nunique() == 1:
        return True
    elif df[col].nunique() == 2 and 'INV' in df[col].unique():
        return True
    elif df[df[col] != 'INV'][col].value_counts(1).iloc[0] > 0.9:
        return True
    elif 'SIM' in df[col].unique():
        if df[df[col] != 'INV'][col].value_counts().iloc[-1] < 5:
            return True
        else:
            return False
    else:
        return False

cols_to_remove = [col for col in df.columns if drop_low_variance(col)]
df.drop(cols_to_remove, axis=1, inplace=True)

# COMMAND ----------

cols_to_remove

# COMMAND ----------

display(df)

# COMMAND ----------

display(df[["prova_1","resultado_processo"]])

# COMMAND ----------

