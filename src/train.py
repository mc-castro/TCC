# Databricks notebook source
# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %pip install mlflow==1.24.0

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

import shap


class GBMFeatTransformer(BaseEstimator, TransformerMixin):
  
    def __init__(self, high_card_feats=None):
        self.high_card_feats = high_card_feats
        self.class_to_freq = dict()        

    def fit(self, X, y=None):
        for col in self.high_card_feats:
            self.class_to_freq[col] = X[col].value_counts().to_dict()
        return self
            
    def transform(self, X, y=None):
        ret = X.copy()
        for col in self.high_card_feats:
            ret[col] = ret[col].replace(self.class_to_freq[col])
            ret[col] = ret[col].apply(lambda x: 0 if type(x) != int else x).astype(float)
        for col in ret.columns:
            if ret[col].dtype == 'O':
                ret[col] = ret[col].astype('category')
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

    def optimize_hyperparameters(self, pipe, X, y, max_evals=1):
        
        opt_space = {'lgbm__learning_rate': hp.loguniform('lgbm__learning_rate', np.log(0.001), np.log(0.5)),
                        'lgbm__reg_alpha': hp.loguniform('lgbm__reg_alpha', np.log(0.001), np.log(1)),
                        'lgbm__reg_lambda': hp.loguniform('lgbm__reg_lambda', np.log(0.001), np.log(1)),
                        'lgbm__subsample': hp.uniform('lgbm__subsample', 0.2, 1),
                        'lgbm__colsample_bytree': hp.uniform('lgbm__colsample_bytree', 0.2, 1),
                        'lgbm__min_child_samples': scope.int(hp.quniform('lgbm__min_child_samples', 1, 100, 1)),
                        'lgbm__num_leaves': scope.int(hp.quniform('lgbm__num_leaves', 2, 50, 1)),
                        'lgbm__subsample_freq': scope.int(hp.quniform('lgbm__subsample_freq', 1, 10, 1)),
                        'lgbm__n_estimators': scope.int(hp.quniform('lgbm__n_estimators', 100, 5000, 1))}

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
        mlflow.sklearn.log_model(model, "lgbm")

        
    def fit(self, df):
       
        self.X = df.drop(['codigo_processo','resultado_processo'], axis=1)
        le = LabelEncoder().fit(df['resultado_processo'])
        y = pd.Series(le.transform(df['resultado_processo']))

        cat_feats = self.find_cat_features(self.X)
        high_card_feats = [col for col in cat_feats if self.is_high_card(col, self.X)]
        cat_feats = [col for col in cat_feats if col not in high_card_feats]

        self.lgbm_pipe = Pipeline(
        [
            ('feat_trans', GBMFeatTransformer(high_card_feats)),
            ('over', RandomOverSampler(random_state=9)),
            ('lgbm', LGBMClassifier())
        ]
        )

        best_hypers = self.optimize_hyperparameters(clone(self.lgbm_pipe), self.X, y)
        self.model = clone(self.lgbm_pipe).set_params(**best_hypers).fit(self.X, y)
        y_pred = cross_val_predict(clone(self.lgbm_pipe).set_params(**best_hypers), self.X, y, cv=10)
  
        self.track_results(y, y_pred, self.model, best_hypers)
    
        #ranking de features
        provas = [col for col in self.X.columns if 'prova' in col]
        explainer = shap.TreeExplainer(self.model["lgbm"])
        data_transformation = self.model['feat_trans'].fit_transform(self.X)

        shap_values = explainer.shap_values(data_transformation, approximate=False, check_additivity=False)

        cat_map = {}
        for col in data_transformation.columns:
            if data_transformation[col].dtype.name == 'category':
                cat_map[col] = dict(enumerate(data_transformation[col].cat.categories.tolist()))
                data_transformation[col] = data_transformation[col].cat.codes
            else:
                data_transformation[col].fillna(data_transformation[col].median(), inplace=True)
                
        vals= np.abs(shap_values)[1].mean(0)
        feature_importance = pd.DataFrame(list(zip(data_transformation.columns,vals)),columns=['Provas','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
                
        col_feat = data_transformation.columns.to_list()
        feat_result_favoravel = {}        
        for coluna_num, coluna in enumerate(col_feat):
            try:
              df_col = pd.DataFrame(shap_values[1][:,coluna_num], columns=['shap_value'])
              df_col.reset_index(level=0, inplace=True)
              df_col_neg = df_col.loc[df_col['shap_value'] < 0,]
              df_col_pos = df_col.loc[df_col['shap_value'] > 0,]

              list_neg = df_col_neg.index.to_list()
              list_pos = df_col_pos.index.to_list()

              value_neg = data_transformation.iloc[list_neg,coluna_num].mode()[0]
              value_pos = data_transformation.iloc[list_pos,coluna_num].mode()[0]

              feat_result_favoravel[coluna] = value_pos

            except:
              pass
        
        #seleciona as provas que possuem campos inválidos, pois elas classificam: 2-> impacto positivo e 1-> impacto negativo 
        feats_with_inv = [col for col in df.columns if 'INV' in df[col].unique()]
        
        feats_importance = pd.DataFrame(list(feat_result_favoravel.items()), columns = ['Provas', 'impacto'])
        prova = pd.DataFrame(provas, columns = ['Provas'])
        m = pd.merge(feats_importance, prova, how = 'inner', on = 'Provas')
        m_df = pd.merge(feature_importance, m, how = 'inner', on = 'Provas')
        m_df['impacto'] = m_df['impacto'].astype(str)

        m_df.loc[(m_df['impacto'] == '1.0') & (m_df['Provas'].isin(feats_with_inv).astype(int)), 'impacto'] = 'Negativo'
        m_df.loc[(m_df['impacto'] == '2.0'), 'impacto'] = 'Positivo'
        m_df.loc[(m_df['impacto'] == '0.0'), 'impacto'] = 'Negativo'
        m_df.loc[(m_df['impacto'] == '1.0') & ~(m_df['Provas'].isin(feats_with_inv).astype(int)), 'impacto'] = 'Positivo'
        
        #transforma o df para dict:
        provas_dict = m_df[m_df.columns[0]].to_dict()
        value_importance_dict = m_df[m_df.columns[1]].to_dict()
        impacto_dict = m_df[m_df.columns[2]].to_dict()
        
        #converte o dict para string
        self.provas_str = str(provas_dict) 
        self.value_importance_str = str(value_importance_dict) 
        self.impacto_str = str(impacto_dict)
        
        with open("fitted_pipe.pkl", "wb") as f:
            cloudpickle.dump(self.model, f)
      
      
    def predict(self, context, df):
 
        df['vt'] = df['vt'].astype(str)
        proba = self.model.predict_proba(df)
        proba = pd.DataFrame(proba,columns=['prob_0','prob_1'])
        proba['predict'] = self.model.predict(df)
        proba['prob'] = proba.loc[:, ['prob_0', 'prob_1']].max(1)
        proba = proba.drop(['prob_0', 'prob_1'], axis=1)        

        #ranking de features
        proba['provas'] = self.provas_str
        proba['value importance'] = self.value_importance_str
        proba['impacto'] = self.impacto_str
        
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
                'git+https://${SPARKPASSWORD}:x-oauth-basic@github.com/cervejaria-ambev/pyiris.git@v${VERSION_NUMBER}'
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

# Dummy
print(classification_report(y, DummyClassifier(strategy='stratified').fit(X, y).predict(X)))

# COMMAND ----------

y.value_counts()

# COMMAND ----------

ros = RandomOverSampler(random_state=9)
X_resampled, y_resampled = ros.fit_resample(X, y)
y_resampled.value_counts()

# COMMAND ----------

