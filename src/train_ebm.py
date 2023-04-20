# Databricks notebook source
# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %pip install mlflow==1.24.0

# COMMAND ----------

# MAGIC %pip install interpret

# COMMAND ----------

# Databricks notebook source
import pickle
import cloudpickle
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import recall_score, precision_score, make_scorer, f1_score, classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hyperopt import fmin, tpe, hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from functools import partial
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

import shap

class GBMFeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, high_card_feats=None, EBM=False):
        self.high_card_feats = high_card_feats
        self.EBM = EBM
        self.class_to_freq = dict()
        self.num_feats_means = dict()
        
    def fit(self, X, y=None):
        for col in self.high_card_feats:
            self.class_to_freq[col] = X[col].value_counts().to_dict()
        if self.EBM:
            for col in X.columns:
                if X[col].dtype != 'O':
                    self.num_feats_means[col] = X[col].mean()
        return self
            
    def transform(self, X, y=None):
        ret = X.copy()
        for col in self.high_card_feats:
            ret[col] = ret[col].replace(self.class_to_freq[col])
            ret[col] = ret[col].apply(lambda x: 0 if type(x) != int else x).astype(float)
        if not self.EBM:
            for col in ret.columns:
                if ret[col].dtype == 'O':
                    ret[col] = ret[col].astype('category')
        else:
            for col in self.num_feats_means:
                ret[col] = ret[col].fillna(self.num_feats_means[col])
        return ret

# COMMAND ----------

class PriceRangePredictor(mlflow.pyfunc.PythonModel):
  
    def load_context(self, context):
        with open(context.artifacts["estimator"], "rb") as f:
            self.model = pickle.load(f)

    def find_cat_features(self, df):
          df['vt'] = df['vt'].astype(str)
          feats = df.columns[:-1]
          cat_feats = [feat for feat in feats if df[feat].dtype == 'O']
          return cat_feats

    def is_high_card(self, col, df):
        if df[col].nunique()/len(df) > 0.3:
            return True
        else:
            return False

    def optimize_hyperparameters(self, pipe, X, y, max_evals=100):
        
        opt_space = {'ebm__learning_rate': hp.loguniform('ebm__learning_rate', np.log(0.001), np.log(0.5)),
                     #'ebm__validation_size': hp.uniform('ebm__validation_size', 0.05, 0.25),
                     #'ebm__early_stopping_rounds': scope.int(hp.quniform('early__stopping_rounds', 5, 100, 1)),
                     'ebm__max_rounds': scope.int(hp.quniform('ebm__max_rounds', 10, 3000, 1)),
                     'ebm__interactions': scope.int(hp.quniform('ebm__interactions', 0, 20, 1)),
                     'ebm__max_leaves': scope.int(hp.quniform('ebm__max_leaves', 2, 10, 1)),
                     'ebm__outer_bags': scope.int(hp.quniform('ebm__outer_bags', 8, 16, 1)),
                     #'ebm__inner_bags': scope.int(hp.quniform('ebm__inner_bags', 0, 5, 1)),
                     'ebm__max_bins': scope.int(hp.quniform('ebm__max_bins', 8, 128, 1)),
                     'ebm__max_interaction_bins': scope.int(hp.quniform('ebm__max_interaction_bins', 8, 64, 1)),
                     'ebm__min_samples_leaf': scope.int(hp.quniform('ebm__min_samples_leaf', 1, 30, 1))}

        def obj(x):
            
            model = clone(pipe).set_params(**x)
            preds = cross_val_predict(model, X, y, cv=10, n_jobs=-1)

            return -f1_score(y, preds, average='macro')


        best_hypers = fmin(obj, space=opt_space, algo=tpe.suggest, 
                        max_evals=max_evals, return_argmin=False)

        return best_hypers
      
    def track_results(self,y, y_pred, model, hypers):
        """Register the best hyperparameters, dev metrics and model with MLFlow."""
        recall0 = recall_score(y, y_pred, pos_label=0)
        recall1 = recall_score(y, y_pred, pos_label=1)
        precision0 = precision_score(y, y_pred, pos_label=0)
        precision1 = precision_score(y, y_pred, pos_label=1)
        f1_0 = f1_score(y, y_pred, pos_label=0)
        f1_1 = f1_score(y, y_pred, pos_label=1)
        accuracy = accuracy_score(y,y_pred)
        

        mlflow.log_metric("recall_class0", recall0)
        mlflow.log_metric("recall_class1", recall1)
        mlflow.log_metric("precision_class0", precision0)
        mlflow.log_metric("precision_class1", precision1)
        mlflow.log_metric("f1_class0", f1_0)
        mlflow.log_metric("f1_class1", f1_1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(hypers)
        mlflow.sklearn.log_model(model, "ebm")

        
    def fit(self, df):
       
        self.X = df.drop(['codigo_processo','resultado_processo'], axis=1)
        le = LabelEncoder().fit(df['resultado_processo'])
        y = pd.Series(le.transform(df['resultado_processo']))

        cat_feats = self.find_cat_features(self.X)
        high_card_feats = [col for col in cat_feats if self.is_high_card(col, self.X)]
        cat_feats = [col for col in cat_feats if col not in high_card_feats]

        self.ebm_pipe = Pipeline(
        [
            ('feat_trans', GBMFeatTransformer(high_card_feats, EBM=True)),
            #('over', RandomOverSampler(random_state=9)),
            ('ebm', ExplainableBoostingClassifier(n_jobs=1, validation_size=0))
        ]
        )


        best_hypers = self.optimize_hyperparameters(clone(self.ebm_pipe), self.X, y)
        self.model = clone(self.ebm_pipe).set_params(**best_hypers).fit(self.X, y)
        y_pred = cross_val_predict(clone(self.ebm_pipe).set_params(**best_hypers), self.X, y, cv=10)
  
        self.track_results(y, y_pred, self.model, best_hypers)

        
        with open("fitted_pipe.pkl", "wb") as f:
            cloudpickle.dump(self.model, f)
      
      
    def predict(self, context, df):
 
        df['vt'] = df['vt'].astype(str)
        proba = self.model.predict_proba(df)
        proba = pd.DataFrame(proba,columns=['prob_0','prob_1'])
        proba['predict'] = self.model.predict(df)
        proba['prob'] = proba.loc[:, ['prob_0', 'prob_1']].max(1)
        proba = proba.drop(['prob_0', 'prob_1'], axis=1)        
        
        return proba
      
      
def read_data():
    
        df = spark.read.csv("dbfs:/FileStore/tables/feats.csv", header="true", inferSchema="true")
        
        pandas_df = df.toPandas()
        
        return pandas_df
  
  
if __name__ == "__main__":

    mlflow.set_experiment("/TCC/train_processos")
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with mlflow.start_run(run_name=f"lightgbm_{date_now}") as run:

        # Load object and train model
        prp = PriceRangePredictor()
        
        # Read the data
        data = read_data()
        #ret[col].replace(["INV"], np.nan)
        #data['ultima_remuneracao'] = data['ultima_remuneracao'].fillna( data['ultima_remuneracao'].mean())
        data['prova_35'] = data['prova_35'].fillna( "NÂO")
        #data['ultima_remuneracao'] = data['ultima_remuneracao'].fillna( data['ultima_remuneracao'].mean())

        # Train the model
        train = prp.fit(data)
        
        conda_env = {
            'name': 'mlflow-env',
            'channels': [
                'defaults',
                'anaconda',
                'conda-forge'
            ],
            'dependencies': [
                'python=3.7.0',
                'cloudpickle==1.6.0',
                'scikit-learn==0.22.1',
                'imbalanced-learn'
            ]
        }
        
        artifacts = {
          'estimator': 'fitted_pipe.pkl'
        }
        
        mlflow.set_tag('model', 'GBMfeat+LGBM')
        
        X = data.drop(['codigo_processo','resultado_processo'], axis=1)
        y = prp.predict('context', X).predict
        
        model_signature = infer_signature(X, y.to_frame())
        
        mlflow.pyfunc.log_model(
            artifact_path='model',
            artifacts=artifacts,
            python_model = prp,
            signature = model_signature,
            conda_env = conda_env
        )
 
        mlflow.end_run()

# COMMAND ----------

prp.predict('context', data.drop(['codigo_processo', 'resultado_processo'], axis=1))

# COMMAND ----------

display(data.drop(['codigo_processo', 'resultado_processo'], axis=1))

# COMMAND ----------

y.value_counts()

# COMMAND ----------

ros = RandomOverSampler(random_state=9)
X_resampled, y_resampled = ros.fit_resample(X, y)
y_resampled.value_counts()

# COMMAND ----------

