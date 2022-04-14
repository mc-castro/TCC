# Databricks notebook source
# MAGIC %pip install boruta

# COMMAND ----------

# MAGIC %pip install scikit-learn==0.24.2

# COMMAND ----------

# MAGIC %pip install sklearn-genetic

# COMMAND ----------

# Databricks notebook source

import mlflow
import pandas as pd
import numpy as np
import logging
from pyspark.sql import DataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif, SelectKBest
from sklearn.metrics import classification_report, recall_score, precision_score, make_scorer, f1_score
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy
from genetic_selection import GeneticSelectionCV
from collections import Counter

# COMMAND ----------


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.cat_feats = None
        self.num_feats = None

    def read_data(self) -> pd.DataFrame:
        df = spark.read.csv("dbfs:/FileStore/tables/base_tcc_csv.csv", header="true", inferSchema="true")
        
        pandas_df = df.toPandas()
        
        return pandas_df
      
    def find_feature_types(self, df):
        '''
            this method selects the columns that will be features and then separates them into numeric and categorical
        '''
        df['vt'] = df['vt'].astype(str)
        feats = df.columns[1:-1]
        self.num_feats = [feat for feat in feats if df[feat].dtype != 'O']
        self.cat_feats = [feat for feat in feats if feat not in self.num_feats]
        return df
      
    def identify_low_variance_columns(self, df, col) -> bool:
        '''
            this method will identify columns with very dominant classes to be removed in drop_low_variance
        '''
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
          
    def drop_low_variance(self, df: pd.DataFrame):
        low_var_feats = [col for col in self.cat_feats if self.identify_low_variance_columns(df, col)]
        self.cat_feats = [feat for feat in self.cat_feats if feat not in low_var_feats]
        return df.drop(columns=low_var_feats)
      
      
    def boruta_selector (self, model, X, y, columns):  
        bor_selector = BorutaPy(model, n_estimators='auto', perc=100, max_iter=1000)
        bor_selector.fit(X.values, y.values)
        return columns[bor_selector.ranking_.argsort()[-(len(columns)-30):]]


    def sequential_selector (self, model, X, y, columns):
        ss_selector = SequentialFeatureSelector(model, n_features_to_select=30, direction='forward',
                                              scoring=make_scorer(f1_score, average='macro'), 
                                              cv=StratifiedKFold(shuffle=True, random_state=9), n_jobs=-1)
        ss_selector.fit(X, y)
        return columns[~ss_selector.get_support()]

    def mutual_information_selector (self, model, X, y, columns):
        mi_selector = SelectKBest(mutual_info_classif, k=30).fit(X, y)
        return columns[~mi_selector.get_support()]

    def genetic_algorithm_selector (self, model, X, y, columns):
        ga_selector = GeneticSelectionCV(model, cv=StratifiedKFold(shuffle=True, random_state=9), 
                                       scoring=make_scorer(f1_score, average='macro'), max_features=30, 
                                       n_population=100, n_generations=40, n_gen_no_change=10)
        ga_selector.fit(X, y)
        return columns[~ga_selector.get_support()]
        
    
    def feat_selection_pipe(self, df):

        df_ret = df.copy()
        for col in df[self.cat_feats].columns:
          df[col].fillna('Missing', inplace=True)
          le = LabelEncoder().fit(df[col])
          df[col] = pd.Series(le.transform(df[col]))

        #numeric feats:
        df[self.num_feats] = df[self.num_feats].fillna(df[self.num_feats].mean()) 

        X = df[self.cat_feats + self.num_feats].copy()

        #target:
        le = LabelEncoder().fit(df['resultado_processo'])
        y = pd.Series(le.transform(df['resultado_processo']))
        
        base_model = RandomForestClassifier(n_jobs=-1, max_depth=5, class_weight='balanced')
        columns = X.columns.values
        
        bottom_boruta = self.boruta_selector(clone(base_model), X, y, columns)
        bottom_ss = self.sequential_selector(clone(base_model), X, y, columns)
        bottom_mi = self.mutual_information_selector(clone(base_model), X, y, columns)
        bottom_ga = self.genetic_algorithm_selector(clone(base_model), X, y, columns)
        
        counts = pd.Series(dict(Counter(np.hstack([bottom_boruta, bottom_ss, bottom_mi, bottom_ga]))))
        #counts = pd.Series(dict(Counter(np.hstack([bottom_boruta, bottom_mi]))))
        
        cols_to_decide = counts[counts>2].index
        
        cg = [elem for elem in cols_to_decide if 'prova' not in elem if 'cargo' not in elem]
        tempo = [elem for elem in cols_to_decide if 'tempo_cargo' in elem]
        
        cols_to_remove = cg + tempo
        df_ret = df_ret.drop(columns=cols_to_remove)
        
        return df_ret
      
    def write_prepared_data(self, transformed_dataframe: DataFrame) -> None:
        dbutils.fs.rm("dbfs:/FileStore/tables/feats.csv", True)
        transformed_dataframe.write.csv("dbfs:/FileStore/tables/feats.csv", header="true")
        
        return

    def run(self):
        df = self.read_data()
        df = self.find_feature_types(df)
        df = self.drop_low_variance(df) 
        df = self.feat_selection_pipe(df)
        self.write_prepared_data(spark.createDataFrame(df))
        
if __name__ == "__main__":
     FeatureEngineeringPipeline().run()

# COMMAND ----------

feats = spark.read.csv("dbfs:/FileStore/tables/feats.csv", header="true", inferSchema="true")
display(feats)

# COMMAND ----------

pandas_df = feats.toPandas()
pandas_df['resultado_processo'].value_counts()

# COMMAND ----------

