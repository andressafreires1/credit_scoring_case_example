# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###### Tópicos que serão tratados neste notebook:
# MAGIC 
# MAGIC - 1) Leitura da base escorável após seleção de variáveis finais 
# MAGIC - 2) Aplicação do modelo final 
# MAGIC - 3) Gráfico de Shap
# MAGIC - 4) Demográfico das variáveis
# MAGIC - 5) Categorização das variáveis para monitoramento
# MAGIC 
# MAGIC ##### Equipe: DS Credit

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 01 . Bibliotecas ----------------------------------------------------------------------------

# COMMAND ----------

# Libraries

# ====================================================================================================================================================================
# -- BIBLIOTECAS -----------------------------------------------------------------------------------------------------------------------------------------------------


import pyspark.sql.functions as F
import pandas                as pd
import numpy                 as np
import datetime              as dt
import xgboost               as xgb
import matplotlib.pyplot     as plt
import seaborn               as sns

import mlflow
import mlflow.sklearn
import sklearn
import os
import warnings
import PythonShell

from pyspark.sql      import SQLContext, SparkSession, Window, Row
from pyspark.sql      import SparkSession 
from datetime         import date, datetime, timedelta


    ## Pandas

# Instalação de pacotes
!pip install boruta 
!pip install geneticalgorithm
!pip install progressbar
!pip install imblearn
!pip install category_encoders
!pip install scikit-plot
!pip install optuna
!pip install lightgbm
!pip install shap


from boruta                    import BorutaPy
from pytz                      import timezone
from datetime                  import datetime
from progressbar               import progressbar 
from geneticalgorithm          import geneticalgorithm as GA
from xgboost                   import XGBClassifier
from category_encoders.woe     import WOEEncoder
from lightgbm                  import LGBMClassifier


from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing     import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute            import SimpleImputer 
from sklearn.model_selection   import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel, RFECV, f_classif
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.compose           import ColumnTransformer
from sklearn.metrics           import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, log_loss, roc_curve, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.neural_network    import MLPClassifier

from scipy.stats               import chi2_contingency
from scipy.stats               import pointbiserialr

from imblearn.over_sampling    import RandomOverSampler
from imblearn.under_sampling   import RandomUnderSampler
import scikitplot as skplt
import optuna
import shap

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 600)

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 02 . Funções ----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Ordenação - Gains Table

# COMMAND ----------

def gains_table(baseline=None, base_ref=None,target=None, prob=None):
  
    baseline['decil'] = pd.qcut(baseline[prob], 10)
    rer, bins = pd.qcut(baseline[prob], 10, retbins=True, labels=False)
    
    faixas_inicio = []
    faixas_fim = []
    
    for i, j in enumerate(bins):
      if i != len(bins)-1:
        faixas_inicio.append(bins[i])
        faixas_fim.append(bins[i+1])
    if faixas_inicio[0] != 0:
      faixas_inicio[0] = 0
    if faixas_fim[-1] != 1000:
      faixas_fim[-1] = 1000

    marcacao_decil = []
    for i, j in enumerate(base_ref[prob]):
      for k, l in enumerate(faixas_inicio):
        if(j>faixas_inicio[k] and j<=faixas_fim[k]):
          marcacao_decil.append(k+1)
    print(len(marcacao_decil), len(base_ref)) 
    base_ref['marcacao_decil'] = marcacao_decil
    kstable = pd.crosstab(base_ref['marcacao_decil'], base_ref[target])
    
    ranges = []
    for i, j in enumerate(faixas_inicio):
      ranges.append('(' + str(round(j,2)) + '-' + str(round(faixas_fim[i],2)) + ')')
    kstable['Ranges'] = ranges
    kstable['Qtde_Bom'] = kstable[0]
    kstable['Qtde_Mau'] = kstable[1]
    kstable['Qtde_Total'] = (kstable.Qtde_Bom + kstable.Qtde_Mau)
    
    base_ref['target0'] = 1 - base_ref[target]
    kstable['Perc_Bom']   = (kstable.Qtde_Bom / base_ref['target0'].sum()).apply('{0:.2%}'.format)
    kstable['Perc_Mau']   = (kstable.Qtde_Mau / base_ref[target].sum()).apply('{0:.2%}'.format)
    kstable['Perc_Total'] = ((kstable.Qtde_Bom + kstable.Qtde_Mau) / (base_ref['target0'].sum() + base_ref[target].sum())).apply('{0:.2%}'.format)
    kstable['Perc_Acum_Bom']=(kstable.Qtde_Bom / base_ref['target0'].sum()).cumsum()
    kstable['Perc_Acum_Mau']=(kstable.Qtde_Mau / base_ref[target].sum()).cumsum()
    kstable['KS'] = np.round(kstable['Perc_Acum_Mau']-kstable['Perc_Acum_Bom'], 4) * 100
    
    #Formatando
    kstable['Perc_Acum_Mau']= kstable['Perc_Acum_Mau'].apply('{0:.2%}'.format)
    kstable['Perc_Acum_Bom']= kstable['Perc_Acum_Bom'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 12)
    
    return(kstable)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. GINI

# COMMAND ----------

def gini(predict, df, Target):
    # keep probabilities for the positive outcome only
    probs = predict
    # calculate AUC
    auc = roc_auc_score(np.array(df[Target]), np.array(probs))
    print('AUC: %.3f' % auc)
    print('Gini: %.3f' % (2*(auc-0.5)))
    # calculate roc curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(df[Target], probs)
    # plot no skill 
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    display(plt.show())
    return auc, (2*(auc-0.5))

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 03. Base final ------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Seleção das variáveis finais

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")

#Base  
selecao_final = spark.read.parquet('s3://picpay-datalake-sandbox/jeobara.zacheski/score_pf_cs/selecao_final_cs_carteira/')

# COMMAND ----------

# Removendo as variáveis que foram substituídas pelas criadas no item anterior
selecao_final2 = selecao_final.drop(
            #Correlaçao com outras variáveis que julgo ser mais importante
            'qtde_cartoes_cadastrados',
            'fracao_cartoes_ativos',
            'porcentagem_cartoes_ativos',
            'ct_min_pay_12m_trx',
            'am_min_payp2p_03m_trx',
            'am_min_payp2pbal_12m_trx',
            'am_min_payp2pcred_03m_trx',
            'ct_min_payp2pbal_12m_trx',
            'ct_min_pay_03m_trx',
            'am_min_paypavgood_06m_trx',
            'ct_min_paypav_06m_trx',
            'ct_min_payp2pbal_06m_trx',
            'ct_min_paypavgood_12m_trx',
            'ct_min_paypav_03m_trx',
            'ct_min_paypavbal_06m_trx',
            'ct_min_payp2pcred_06m_trx',
            'ct_min_paypavgood_06m_trx',
            'ct_min_payp2pcred_03m_trx',
            'am_tot_pay_01m_trx',
            'am_tot_payp2pcred_03m_trx',
            'am_avg_payp2pcred_03m_trx',
            'ct_avg_recp2p_03m_trx',
            'am_max_payp2p_03m_trx',
            'ct_avg_payp2p_12m_trx',
            'ct_tot_recp2p_03m_trx',
            'am_avg_payp2p_03m_trx',
            'am_max_paypav_06m_trx',
            'am_max_paypavcred_03m_trx',
            'ct_max_pay_03m_trx',
            'ct_avg_recp2p_06m_trx',
            'am_avg_paypavcred_03m_trx',
            'am_avg_paypav_06m_trx',
            'ct_tot_payp2p_03m_trx',
            'am_tot_pay_06m_trx',
            'am_max_recp2p_03m_trx',
            'ct_tot_pay_06m_trx',
            'ct_max_payp2p_06m_trx',
            'am_avg_paypavgood_06m_trx',
            'am_tot_paypavcred_03m_trx',
            'am_tot_paypav_06m_trx',
            'am_tot_paypav_03m_trx',
            'am_max_paypav_03m_trx',
            'ct_tot_pay_01m_trx',
            'ct_tot_paypavcred_12m_trx',
            'ct_avg_paypav_03m_trx',
            'ct_max_paypav_03m_trx',
            'ct_tot_paypav_03m_trx',
            'ct_min_payp2p_12m_trx',
            'ct_avg_paypav_06m_trx',
            'ct_tot_paypavbill_06m_trx',
            'ct_tot_payp2pbal_06m_trx',
            'ct_min_paypavcred_12m_trx',
            'ct_max_payp2pbal_12m_trx',
            'ct_avg_paypavbill_12m_trx',
            'ct_min_paypavcred_06m_trx',
            'am_max_paypavgood_06m_trx',
            'ct_tot_payp2pcred_06m_trx',
            'ct_tot_paypav_06m_trx',
            'ct_max_paypav_06m_trx',
            'ct_max_payp2pcred_06m_trx',
            'ct_avg_paypavgood_06m_trx',
            'ct_max_paypavgood_06m_trx',
            'ct_max_paypavbill_06m_trx',
  
            #Variáveis estranhas, não fazem sentido 
            'keyboard_type',
            'tipo_modelo',
            'changed_username_times',
            
            #Eliminado pois não é disponível em ambiente produtivo
            'H2PA',

            # Variáveis Eliminadas no Modelo 1 (maioria por correlação com outras variáveis de mesmo sentido e IV mais altos)
            'am_avg_paypav_12m_trx',
            'am_avg_paypavcred_12m_trx',
            'am_avg_paypavgood_12m_trx',
            'am_max_paypav_12m_trx',
            'am_max_paypavbill_12m_trx',
            'am_max_paypavcred_06m_trx',
            'am_min_paypavgood_12m_trx',
            'am_tot_paypav_12m_trx',
            'am_tot_paypavgood_06m_trx',
            'am_tot_paypavgood_12m_trx',
            'ct_avg_pay_03m_trx',
            'ct_avg_payp2p_06m_trx',
            'ct_avg_payp2pcred_03m_trx',
            'ct_min_paypavbill_12m_trx',
            'ct_tot_pay_03m_trx',
            'ct_tot_payp2p_06m_trx',
            'ct_tot_payp2pcred_03m_trx',
            'qtde_cartoes_da_bandeira_master',
            'ddd_region',
  
            # Variáveis Eliminadas no Modelo 2 (maioria por correlação com outras variáveis de mesmo sentido e IV mais altos)
            'ct_avg_pay_06m_trx',
            'am_tot_pay_12m_trx',
            'ct_max_paypavbill_12m_trx',
            'am_max_pay_06m_trx',
            'am_avg_pay_12m_trx',
            'ct_max_pay_12m_trx',
            'am_max_recp2pcred_12m_trx',
            'ct_min_paypavcred_03m_trx',
            'am_tot_payp2p_12m_trx',
            'am_avg_paypavmix_12m_trx',
            'ct_tot_paypavcred_03m_trx',
            'ct_avg_paypavcred_03m_trx',
            'am_min_paypav_06m_trx',
            'am_max_payp2p_12m_trx',
            'am_avg_paypavbill_12m_trx',
            'am_avg_pay_06m_trx',
            'am_avg_payp2pbal_12m_trx',
            'am_avg_paypavbal_06m_trx',
            'am_avg_payp2p_12m_trx',
            'am_tot_payp2pcred_12m_trx',
            'ct_min_recp2p_12m_trx',
            'ct_max_paypav_12m_trx',
            'ct_avg_recp2p_12m_trx',
            'ct_tot_pay_12m_trx',
            'ct_tot_paypavbill_12m_trx',
            'am_avg_payp2pcred_12m_trx',
            'ct_max_payp2pbal_06m_trx',
            'ct_max_payp2p_12m_trx',
            'ct_tot_paypav_12m_trx',
            'ct_avg_paypav_12m_trx',
            'ct_tot_paypavgood_06m_trx',
            'ct_avg_recp2pcred_06m_trx',
            'ct_tot_payp2p_12m_trx',
            'ct_max_recp2pbal_12m_trx',
            'qtde_bandeiras_distintas',
            'flag_consumer',
  
            # Variáveis Eliminadas no Modelo 3 (IV baixos e pouca importancia - ref. LGB)
            'loss_aversion_score',
            'qtde_cartoes_da_bandeira_nao_especificado',
            'ct_min_recp2pbal_12m_trx',
            'ct_avg_paypavgood_12m_trx',
            'ct_min_pay_06m_trx',
            'ct_min_payp2pcred_12m_trx',
            'am_avg_recp2p_03m_trx',
            'am_tot_recp2p_03m_trx',
            'qtde_cartoes_da_bandeira_visa',
  
            # Variáveis Eliminadas no Modelo 4 (período 3 meses, público behaviour)
            'am_max_payp2pcred_03m_trx',
            'ct_max_paypavcred_03m_trx',
            'ct_min_payp2p_03m_trx',
            'ct_max_payp2pcred_03m_trx',
  
            # Variáveis Eliminadas no Modelo 5 (mais fracas, menor importância)
            'ct_tot_payp2pbal_12m_trx',
            'ct_tot_paypavbal_06m_trx',
            'ct_min_paypavmix_12m_trx',
            
            # Variáveis Eliminadas no Modelo 6 (mais fracas, menor importância - pedido João) 
            'am_tot_paypavbill_12m_trx',
            'ct_min_paypavbal_12m_trx',
            'default_score',
            'is_business_account',
            'am_min_payp2pbal_06m_trx',
            'ct_max_pay_06m_trx',
            'ct_tot_recp2p_12m_trx',
            'device_ref',
            'ct_avg_paypavbill_06m_trx'

)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Bases (Dev + Val + Oot)

# COMMAND ----------

## Segmentação do grupo que será modelado
tmp = selecao_final2.toPandas() 

# Conversões Strings para numéricos 
tmp['am_max_pay_12m_trx']         = tmp['am_max_pay_12m_trx'].astype(float)
tmp['am_tot_payp2pbal_06m_trx']   = tmp['am_tot_payp2pbal_06m_trx'].astype(float)
tmp['am_max_payp2pbal_12m_trx']   = tmp['am_max_payp2pbal_12m_trx'].astype(float)
tmp['am_min_paypavcred_06m_trx']  = tmp['am_min_paypavcred_06m_trx'].astype(float)
tmp['am_tot_paypavcred_12m_trx']  = tmp['am_tot_paypavcred_12m_trx'].astype(float)
tmp['am_tot_paypavbal_06m_trx']   = tmp['am_tot_paypavbal_06m_trx'].astype(float)
tmp['am_tot_paypavmix_12m_trx']   = tmp['am_tot_paypavmix_12m_trx'].astype(float)
tmp['am_max_paypavgood_12m_trx']  = tmp['am_max_paypavgood_12m_trx'].astype(float)
tmp['am_max_recp2p_12m_trx']      = tmp['am_max_recp2p_12m_trx'].astype(float)


# Separação em Desenvolvimento (100%) e Out-of-Time
base_tot = tmp.where(tmp['ref_portfolio'] != '2020-04')
base_tot = base_tot.dropna(how='all') 

base_oot = tmp.where(tmp['ref_portfolio'] == '2020-04')
base_oot = base_oot.dropna(how='all') 

# Transformando as variáveis categoricas em dummies (k - 1 categoricas, one hot enconding)
#df_proc = pd.get_dummies(base_tot, drop_first = True, prefix_sep = ':')

# COMMAND ----------

## Segmentação do grupo que será modelado
aux = base_tot #df_proc

# Variáveis númericas - Substituir null por 0 (Zero)
numericas   = aux.select_dtypes(include=['int','float','float32','float64']).columns.tolist()

for col in numericas:
  aux[col] = pd.np.where(aux[col].isin([np.inf, -np.inf]), np.nan, aux[col])
  aux[col] = aux[col].fillna(0)
  
# Variáveis categóricas - Substituir null por "outros"
categoricas = aux.select_dtypes(['object']).columns.tolist()

for col in categoricas:
  aux[col] = pd.np.where(aux[col].isin([np.inf, -np.inf]), np.nan, aux[col])
  aux[col] = aux[col].fillna("others")
  aux[col] = aux[col].replace(True, 'True')
  aux[col] = aux[col].replace(False, 'False')
  
  
# Separação do período de desenvolvimento
  # DEV: 80%
  # VAL: 20%
  
 ## Removendo variáveis identificadoras e caracter (como transformar em DUMMY?)
features_final = [i for i in aux.columns if i not in ['cpf','consumer_id','ref_portfolio','ref_date','SConcEver60dP6_100','performance']]
resposta_final = 'SConcEver60dP6_100'
  
## Segmentação do grupo que será modelado
tot_final = aux
               
## Criando dataset temporário para testes - Somente com variáveis numéricas")
x, y = tot_final[features_final], tot_final[resposta_final]
  
## Separação da base de dados em treino e teste (desenvolvimento e validação)
  ## Necessário verificar qual sera o percentual de separação, gosto de trabalhar com 80/20 ou 70/30
x_base_dev, x_base_val, y_base_dev, y_base_val = train_test_split(x, y, train_size = 0.8, random_state = 123)

base_dev = pd.concat([x_base_dev, y_base_dev], axis = 1)
base_val = pd.concat([x_base_val, y_base_val], axis = 1)
base_dev.reset_index(drop = True, inplace = True)
base_val.reset_index(drop = True, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 04.  Modelo final -------------------------------------------------------------

# COMMAND ----------

# Total Desenvolvimento - 100%
tot_x = base_tot.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA','consumer_id', 'ref_portfolio', 'ref_date','cpf'])
tot_y = base_tot.filter(['SConcEver60dP6_100'])

# Desenvolvimento - 80%
dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

# Validação - 20%
val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Out-of-Time
oot_x = base_oot.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'consumer_id', 'ref_portfolio', 'ref_date','cpf'])
oot_y = base_oot.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

cat_columns

# COMMAND ----------

num_columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##### LGB - Light Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAGIC 
# MAGIC HSPA + INTERNAS

# COMMAND ----------

# Treinando com os parâmetros escolhidos
LGB = Pipeline(steps = [('preprocessor_lgb', preprocessor),
                        ('model_lgb',LGBMClassifier(random_state = 123, boosting_type = 'dart', max_depth = 3, n_estimators = 175, learning_rate = 0.4940814646340456,
                                                    class_weight = 'balanced', reg_alpha = 0.02399020943974248, reg_lambda = 0.014865002953056582, subsample = 0.75))]) 
LGB.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = LGB.predict_proba(dev_x)[:,1]
y_predict_val = LGB.predict_proba(val_x)[:,1]
y_predict_tot = LGB.predict_proba(tot_x)[:,1]
y_predict_oot = LGB.predict_proba(oot_x)[:,1]

proba_dev = LGB.predict_proba(dev_x)
proba_val = LGB.predict_proba(val_x)
proba_tot = LGB.predict_proba(tot_x)
proba_oot = LGB.predict_proba(oot_x)


# COMMAND ----------

#Features Importance (Gráfico colorido)
features = num_columns+cat_columns
importances = LGB.named_steps['model_lgb'].feature_importances_
indices = np.argsort(importances)

xx = importances[indices]
yy = [features[i] for i in indices]
df_importances = pd.DataFrame({'Columns':yy, 'Importances':xx})

df_importances.columns = ['Columns','Importances']
df_importances = df_importances.sort_values(by = 'Importances', ascending = False)

sns.set(rc={'figure.figsize':(5,6)})
sns.barplot(y="Columns", x="Importances", data=df_importances)

# COMMAND ----------

#Tabela com a importância das variáveis
df_importances

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 05. Ordenação

# COMMAND ----------

# Bases de entrada

dev = base_dev # Base referência para cálculo dos ranges de score
tot = base_tot # Apenas aplicar range
val = base_val # Apenas aplicar range
oot = base_oot # Apenas aplicar range

# COMMAND ----------

#Escoragem em toda a base

## Score 0 = Probabilidade bom
dev['score_cs_carteira_0'] = proba_dev[:, 0]*1000
tot['score_cs_carteira_0'] = proba_tot[:, 0]*1000
val['score_cs_carteira_0'] = proba_val[:, 0]*1000
oot['score_cs_carteira_0'] = proba_oot[:, 0]*1000

## Score 1 = Probabilidade Mau
dev['score_cs_carteira_1'] = proba_dev[:, 1]*1000
tot['score_cs_carteira_1'] = proba_tot[:, 1]*1000
val['score_cs_carteira_1'] = proba_val[:, 1]*1000
oot['score_cs_carteira_1'] = proba_oot[:, 1]*1000

# COMMAND ----------

pp_gains_dev = gains_table(baseline=dev,base_ref=dev, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot = gains_table(baseline=dev,base_ref=tot, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val = gains_table(baseline=dev,base_ref=val, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot = gains_table(baseline=dev,base_ref=oot, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev

# COMMAND ----------

pp_gains_tot

# COMMAND ----------

pp_gains_val

# COMMAND ----------

pp_gains_oot

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 06. Gráfico de Shap

# COMMAND ----------

X_shap = pd.DataFrame(LGB['preprocessor_lgb'].transform(dev_x), columns = num_columns+cat_columns)
explainer_rf = shap.TreeExplainer(LGB['model_lgb'])
shap_values = explainer_rf.shap_values(X_shap, check_additivity=False)

# COMMAND ----------

display(shap.summary_plot(shap_values[1], X_shap, plot_type='bar')) #barras
display(shap.summary_plot(shap_values[1], X_shap)) #shap 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 07. Demografico das variáveis finais do modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1. Categóricas

# COMMAND ----------

cat = dev.select_dtypes(['object', 'category']).columns.tolist()
cat.append('ref_portfolio')
analise_categoricas = selecao_final2.toPandas().filter(items=cat)

# COMMAND ----------

analise_categoricas.columns

# COMMAND ----------

ddd_state_group_by = analise_categoricas.groupby(['ref_portfolio', 'ddd_state']).ddd_state.agg(['count']).rename(columns={'count': 'perc'})
ddd_state_group_by  = ddd_state_group_by.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
ddd_state_group_by.reset_index().pivot('ref_portfolio','ddd_state','perc').plot(kind='bar', title='Estados', stacked = True, figsize=(10, 10),colormap="gist_rainbow")
plt.legend(bbox_to_anchor=(1.0, 0.9))
display(plt.show())

# COMMAND ----------

digital_account_status_group_by = analise_categoricas.groupby(['ref_portfolio', 'digital_account_status']).digital_account_status.agg(['count']).rename(columns={'count': 'perc'})
digital_account_status_group_by  = digital_account_status_group_by.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
digital_account_status_group_by.reset_index().pivot('ref_portfolio','digital_account_status','perc').plot(kind='bar', title='digital_account_status', stacked = True, figsize=(10, 10),colormap="Greens")
plt.legend(bbox_to_anchor=(1.0, 0.5))
display(plt.show())

# COMMAND ----------

bandeira_group_by = analise_categoricas.groupby(['ref_portfolio', 'bandeira_cartao_ref']).bandeira_cartao_ref.agg(['count']).rename(columns={'count': 'perc'})
bandeira_group_by  = bandeira_group_by.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
bandeira_group_by.reset_index().pivot('ref_portfolio','bandeira_cartao_ref','perc').plot(kind='bar', title='bandeira_cartao_ref', stacked = True, figsize=(10, 10),colormap="Greens")
plt.legend(bbox_to_anchor=(1.0, 0.5))
display(plt.show())

# COMMAND ----------

email_group_by = analise_categoricas.groupby(['ref_portfolio', 'email_ajuste']).email_ajuste.agg(['count']).rename(columns={'count': 'perc'})
email_group_by  = email_group_by.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
email_group_by.reset_index().pivot('ref_portfolio','email_ajuste','perc').plot(kind='bar', title='email_ajuste', stacked = True, figsize=(10, 10),colormap="gist_rainbow")
plt.legend(bbox_to_anchor=(1.0, 0.5))
display(plt.show())


# COMMAND ----------

modelo_celular_group_by = analise_categoricas.groupby(['ref_portfolio', 'modelo_celular_ref']).modelo_celular_ref.agg(['count']).rename(columns={'count': 'perc'})
modelo_celular_group_by  = modelo_celular_group_by.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
modelo_celular_group_by.reset_index().pivot('ref_portfolio','modelo_celular_ref','perc').plot(kind='bar', title='modelo_celular_ref', stacked = True, figsize=(10, 10),colormap="gist_rainbow")
plt.legend(bbox_to_anchor=(1.0, 0.9))
display(plt.show())

# COMMAND ----------

instituicao_group_by = analise_categoricas.groupby(['ref_portfolio', 'agrupamento_instituicao_cartao_ref']).agrupamento_instituicao_cartao_ref.agg(['count']).rename(columns={'count': 'perc'})
instituicao_group_by  = instituicao_group_by.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
instituicao_group_by.reset_index().pivot('ref_portfolio','agrupamento_instituicao_cartao_ref','perc').plot(kind='bar', title='agrupamento_instituicao_cartao_ref', stacked = True, figsize=(10, 10),colormap="gist_rainbow")
plt.legend(bbox_to_anchor=(1.0, 0.9))
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2. Númericas - Boxplot

# COMMAND ----------

num = dev.select_dtypes(['int', 'float']).columns.tolist()
num.append('ref_portfolio')
num.remove('SCORE_C2BA')
analise_numericas = selecao_final2.toPandas().filter(items=num)

# COMMAND ----------

analise_numericas.columns

# COMMAND ----------

numericas  = ['days_since_last_transaction', 'idade_em_anos',
       'tempo_registro_em_meses', 'qtde_cartoes_ativos_cadastrados',
       'qtde_emissores_distintos', 'HSPA', 'peer_effect_score',
       'mental_accounting_score', 'am_max_pay_12m_trx',
       'ct_min_paypav_12m_trx', 'ct_tot_payp2pcred_12m_trx',
       'am_tot_payp2pbal_06m_trx', 'am_max_payp2pbal_12m_trx',
       'am_min_paypavcred_06m_trx', 'am_tot_paypavcred_12m_trx',
       'am_tot_paypavbal_06m_trx', 'am_tot_paypavmix_12m_trx',
       'am_max_paypavgood_12m_trx', 'am_max_recp2p_12m_trx',
       'SConcEver60dP6_100']
for col in numericas:
  analise_numericas[col] = pd.to_numeric(analise_numericas[col])

# COMMAND ----------

for i, j in enumerate(numericas):
  analise_numericas.boxplot(column=[j], by=['ref_portfolio'])
  display(plt.show())
