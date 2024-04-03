# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###### Tópicos que serão tratados neste notebook:
# MAGIC 
# MAGIC - 1) Leitura da base escorável após seleção de variáveis finais 
# MAGIC - 2) Aplicação do modelo final 
# MAGIC - 3) Salvs modelo objeto no MLFlow
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

dev_x.columns

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
# MAGIC # ** 06. Score para input em produção
# MAGIC 
# MAGIC Para os casos que temos HSPA null em produção vamos fazer o input da mediana do score durante a base de desenvolvimento (80%), para o público cujo C2BA <= 360

# COMMAND ----------

dev_menor360 = dev.where(dev['SCORE_C2BA'] <= 360) # Base referência para cálculo dos ranges de score
dev_menor360 = dev_menor360.dropna(how='all') 

# COMMAND ----------

print(dev_menor360['HSPA'].median())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 07. Salva objeto modelo no MLFlow

# COMMAND ----------

import os
import mlflow
import mlflow.sklearn

MODEL_NAME='score_pf_cs_carteira'
MLFLOW_FOLDER = '/mlflow-qa'
FULL_EXPERIMENT_ID = '{}/experiments/{}'.format(MLFLOW_FOLDER, MODEL_NAME)
artifact_location = 'dbfs:{}/experiments/{}'.format(MLFLOW_FOLDER, MODEL_NAME)

try:
  mlflow.create_experiment(
  FULL_EXPERIMENT_ID, artifact_location=artifact_location)
  print('CREATING')
except:
  print('ALREADY EXISTS')


# COMMAND ----------

mlflow.set_experiment(FULL_EXPERIMENT_ID)
mlflow.start_run()

# COMMAND ----------

mlflow.sklearn.log_model(LGB['model_lgb'], artifact_path='main_model')
mlflow.sklearn.log_model(LGB['preprocessor_lgb'], artifact_path='preprocessor_lgb')

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 08. Homologação MLFlow

# COMMAND ----------

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Lendo os arquivos do MLFLOW
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

MLFLOW_FOLDER = os.getenv('MLFLOW_FOLDER')

MODEL_NAME = 'score_pf_cs_carteira'
FULL_EXPERIMENT_ID = '{}/experiments/{}'.format(MLFLOW_FOLDER, MODEL_NAME)
mlflow.set_experiment(FULL_EXPERIMENT_ID)
latest_run = dict(mlflow.search_runs().sort_values(by='start_time', ascending=False).iloc[0])
artifact_uri = latest_run['artifact_uri']

model_lgb = mlflow.sklearn.load_model(artifact_uri+ '/main_model')
preprocessor_lgb = mlflow.sklearn.load_model(artifact_uri+ '/preprocessor_lgb')

model_mlflow = Pipeline(steps = [('preprocessor_lgb', preprocessor_lgb),
                        ('model_lgb', model_lgb)]) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Definindo colunas categóricas e numéricas
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

cat_columns_mlflow = ['ddd_state',
 'digital_account_status',
 'bandeira_cartao_ref',
 'email_ajuste',
 'modelo_celular_ref',
 'agrupamento_instituicao_cartao_ref']


num_columns_mlflow = ['days_since_last_transaction',
 'idade_em_anos',
 'tempo_registro_em_meses',
 'qtde_cartoes_ativos_cadastrados',
 'qtde_emissores_distintos',
 'HSPA',
 'peer_effect_score',
 'mental_accounting_score',
 'am_max_pay_12m_trx',
 'ct_min_paypav_12m_trx',
 'ct_tot_payp2pcred_12m_trx',
 'am_tot_payp2pbal_06m_trx',
 'am_max_payp2pbal_12m_trx',
 'am_min_paypavcred_06m_trx',
 'am_tot_paypavcred_12m_trx',
 'am_tot_paypavbal_06m_trx',
 'am_tot_paypavmix_12m_trx',
 'am_max_paypavgood_12m_trx',
 'am_max_recp2p_12m_trx']

# COMMAND ----------

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Escoragem das bases
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

## Score para homologação = inteiro
dev['score_notebook'] = dev['score_cs_carteira_0'].astype(int) 
tot['score_notebook'] = tot['score_cs_carteira_0'].astype(int) 
val['score_notebook'] = val['score_cs_carteira_0'].astype(int) 
oot['score_notebook'] = oot['score_cs_carteira_0'].astype(int) 


# COMMAND ----------

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Escoragem para homologação
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
proba_dev_mlflow = model_mlflow.predict_proba(dev[dev_x.columns])
proba_tot_mlflow = model_mlflow.predict_proba(tot[dev_x.columns])
proba_val_mlflow = model_mlflow.predict_proba(val[dev_x.columns])
proba_oot_mlflow = model_mlflow.predict_proba(oot[dev_x.columns])

dev['score_mlflow'] = proba_dev_mlflow[:,0]*1000
dev['score_mlflow'] = dev['score_mlflow'].astype(int)

tot['score_mlflow'] = proba_tot_mlflow[:,0]*1000
tot['score_mlflow'] = tot['score_mlflow'].astype(int)

val['score_mlflow'] = proba_val_mlflow[:,0]*1000
val['score_mlflow'] = val['score_mlflow'].astype(int)

oot['score_mlflow'] = proba_oot_mlflow[:,0]*1000
oot['score_mlflow'] = oot['score_mlflow'].astype(int)

# COMMAND ----------

dev['dif'] = dev['score_notebook']-dev['score_mlflow']
tot['dif'] = tot['score_notebook']-tot['score_mlflow']
val['dif'] = val['score_notebook']-val['score_mlflow']
oot['dif'] = oot['score_notebook']-oot['score_mlflow']

# COMMAND ----------

dev['dif'].max(), tot['dif'].max(), tot['dif'].max(), oot['dif'].max()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 09. Escoragem base full

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")

#Base total 
base_entrada = spark.read.parquet('s3://picpay-datalake-sandbox/jeobara.zacheski/score_pf_cs/abt_final_full/')\
                                            .filter(F.col('flag_consumer').isin('without_relation','with_relation'))

#Target e mau H0 Serasa (APP/BHV): 
target_serasa = spark.read.parquet('s3://picpay-datalake-sandbox/joao.pinto/score_pf/datasets/features_serasa_target')

#Scores Serasa (APP/BHV): 
scores_serasa = spark.read.parquet('s3://picpay-datalake-sandbox/joao.pinto/score_pf/datasets/features_serasa_scores')

#Features Book Transactions
book_transactions = spark.read.parquet('s3://picpay-datalake-sandbox/joao.pinto/score_pf/datasets/features_book_transactions')

#Features Behavioural Segmentation
bhv_segmentation = spark.read.parquet('s3://picpay-datalake-sandbox/joao.pinto/score_pf/datasets/features_bhv_segmentation')


# Join 1 - Base entrada + informações de target em D0 e 6m
join1 = base_entrada.join(target_serasa, on=['consumer_id','ref_portfolio','flag_consumer'], how='left')\
              .drop(target_serasa.cpf)\
              .drop(target_serasa.ref_date)\
              .drop(target_serasa.ref_serasa)

# Join 2 - Base join1 + informações de scores
join2 = join1.join(scores_serasa, on=['consumer_id','ref_portfolio','flag_consumer'], how='left')\
              .drop(scores_serasa.cpf)\
              .drop(scores_serasa.ref_date)


# Join 3 - informações de book bhv
join3 = join2.join(bhv_segmentation, on=['consumer_id','ref_portfolio','flag_consumer', 'cpf'], how='left')\
              .drop(bhv_segmentation.cpf)\
              .drop(bhv_segmentation.consumer_id)\
              .drop(bhv_segmentation.ref_portfolio)\
              .drop(bhv_segmentation.ref_date)

# Join 3 - informações de book transacional
join4 = join3.join(book_transactions, on=['consumer_id','ref_portfolio','flag_consumer', 'cpf'], how='left')\
              .drop(book_transactions.cpf)\
              .drop(book_transactions.consumer_id)\
              .drop(book_transactions.ref_portfolio)\
              .drop(book_transactions.ref_date)

# Público do behavior:
  ## Cliente com pelo menos 3 meses de relacionamento e com pelo menos 1 transação nos últimos 3 meses, ou seja, devemos excluir desse modelo pois já é escopo do Behavior.

join4 = join4.withColumn('escopo_behavior',F.when((F.col('tempo_registro_em_meses') >= 3)&(F.col('ct_tot_pay_03m_trx') >= 1), 'Sim'))

filtro1 = join4.filter(F.col('escopo_behavior').isNull())

#Score C2BA Serasa Serasa
score_atual = spark.read.parquet('s3://picpay-datalake-sandbox/joao.pinto/poc_bureaux/serasa/pessoa_fisica/pf_c2ba')
score_atual = score_atual.withColumn('cpf', F.col('cpf_hash'))
score_atual = score_atual.drop_duplicates(subset = ['cpf', 'ref_serasa'])
               
# Join 3 - join2 + C2BA 
join5 = filtro1.join(score_atual, on=['cpf','ref_serasa'], how='left')\
              .drop(score_atual.ref_date)\
              .drop(score_atual.cpf_hash)\
              .drop(score_atual.MENSAGEM_SCORE)\
              .drop(score_atual.ref_date)

## Separação do tipo de modelo/desenvolvimento
join6 = join5.withColumn('exclusao',F.when(F.col('SConcEver60dP0_100') == 1, '01. Mau H0')\
                                       .otherwise(F.when((F.col('H2PA') == 0) | (F.col('HSPA') == 0) | (F.col('SCORE_C2BA') == 0), '02. Score Serasa = 0')\
                                          .otherwise(F.when((F.col('H2PA') == 1) | (F.col('HSPA') == 1) | (F.col('SCORE_C2BA') == 1), '03. No-hit Serasa')\
                                             .otherwise(F.when((F.col('SConcEver60dP6_100').isNull()),'04. Sem perf. 6M Serasa')\
                                                 .otherwise(F.when(F.col('has_deactivated') == 'true', '05. Cliente Inativo')\
                                                     .otherwise(F.when((F.col('idade_em_anos') < 18) | (F.col('idade_em_anos') > 90) , '06. Idade < 18 ou > 90')))))))

# COMMAND ----------

#===========================================================================================================================================================================
  #Preprocessing
#===========================================================================================================================================================================

#Ajuste variavel de domínio do e-mail
base = join6.withColumn('email_ajuste',F.when(F.lower(F.col('email_domain')) == 'gmail', 'gmail')\
                                       .otherwise(F.when(F.lower(F.col('email_domain')) == 'hotmail', 'hotmail')\
                                           .otherwise(F.when(F.lower(F.col('email_domain')) == 'yahoo', 'yahoo')\
                                                 .otherwise(F.when(F.lower(F.col('email_domain')) == 'outlook', 'outlook')\
                                                     .otherwise(F.when(F.lower(F.col('email_domain')) == 'icloud', 'icloud')\
                                                          .otherwise(F.when(F.lower(F.col('email_domain')) == 'live', 'live')\
                                                              .otherwise(F.when(F.lower(F.col('email_domain')) == 'uol', 'uol')\
                                                                  .otherwise(F.when(F.lower(F.col('email_domain')) == 'terra', 'terra')\
                                                                      .otherwise(F.when(F.lower(F.col('email_domain')) == 'msn', 'msn')\
                                                                          .otherwise(F.when(F.lower(F.col('email_domain')) == 'globo', 'globo')\
                                                                              .otherwise('Outros/Erro')))))))))))

base = base.withColumn('device_modelo_ref_aj', F.lower(F.col('device_modelo_ref')))

base = base.withColumn('modelo_celular_ref',F.when(base.device_modelo_ref_aj.contains('samsung'),'samsung')\
                           .otherwise(F.when(base.device_modelo_ref_aj.contains('motorola'),'motorola')\
                               .otherwise(F.when(base.device_modelo_ref_aj.contains('iphone'),'iphone')\
                                   .otherwise(F.when(base.device_modelo_ref_aj.contains('xiaomi'),'xiaomi')\
                                       .otherwise(F.when(base.device_modelo_ref_aj.contains('lg'),'lg')\
                                           .otherwise(F.when(base.device_modelo_ref_aj.contains('asus'),'asus')\
                                               .otherwise(F.when(base.device_modelo_ref_aj.contains('lenovo'),'lenovo')\
                                                   .otherwise(F.when(base.device_modelo_ref_aj.contains('tcl'),'tcl')\
                                                       .otherwise(F.when(base.device_modelo_ref_aj.contains('positivo'),'positivo')\
                                                           .otherwise(F.when(base.device_modelo_ref_aj.contains('multilaser'),'multilaser')\
                                                               .otherwise(F.when(base.device_modelo_ref_aj.contains('ipad'),'ipad')\
                                                                   .otherwise(F.when(base.device_modelo_ref_aj.contains('quantum'),'quantum')\
                                                                       .otherwise(F.when(base.device_modelo_ref_aj.contains('sony'),'sony')\
                                                                           .otherwise(F.when(base.device_modelo_ref_aj.contains('nokia'),'nokia')\
                                                                               .otherwise(F.when(base.device_modelo_ref_aj.contains('huawei'),'huawei')\
                                                                                   .otherwise(F.when(base.device_modelo_ref_aj.contains('oneplus'),'oneplus')\
                                                                                       .otherwise('outros')))))))))))))))))

base = base.withColumn('instituicao_cartao_ref_aj', F.lower(F.col('instituicao_cartao_ref')))

base = base.withColumn('agrupamento_instituicao_cartao_ref',\
                               F.when(F.col('instituicao_cartao_ref_aj').isin('bradesco',\
                                                                              'ibibank',\
                                                                              'hsbc',\
                                                                              'leader',\
                                                                              'hsbc bank brasil s.a.  banco multiplo',\
                                                                              'banco bankpar s.a.'),'bradesco')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('banco csf s/a',\
                                                                                        'bco carrefour'),'banco carrefour')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('itau',\
                                                                                         'citibank',\
                                                                                         'citi',\
                                                                                         'banco citibank, s.a.',\
                                                                                         'unibanco',\
                                                                                         'redecard, s.a.',\
                                                                                         'citicard (991)'),'itau')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('santander',\
                                                                                         'santanderserfin',\
                                                                                         'santander rio'),'santander')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('banco do brasil',\
                                                                                         'bancodobrasil',\
                                                                                         'banco do brasil s.a.'),'banco do brasil')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('hub pagamentos s.a.',\
                                                                                         'credz'),'fintechs')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('avista sa credito financiamento e investimento',\
                                                                                 'portoseg cfi',\
                                                                                 'midway',\
                                                                                 'midway s.a. - credito, financiamento e investimento',\
                                                                                 'realize credito financiamento e investimento sa',\
                                                                                 'portoseg s.a. credito financiamento e investimento',\
                                                                                 'renner',\
                                                                                 'pernambucanas financiadora s a - cfi',\
                                                                                 'cred-system administradora de cartoes de credito ltda',\
                                                                                 'caruana s.a. - sociedade de credito, financiamento e investimento',\
                                                                                 'crefisa sa credito financiamento e investimentos',\
                                                                                 'sicredi',\
                                                                                 'omni',\
                                                                                 'cetelem',\
                                                                                 'bv financeira',\
                                                                                 'unik',\
                                                                                 'calcard',\
                                                                                 'credifar',\
                                                                                 'fortbrasil administradora de cartoes de credito sa',\
                                                                                 'uniprime norte do parana',\
                                                                                 'airfox servicos e intermediacoes ltda',\
                                                                                 'dmcard cartoes de credito s.a.',\
                                                                                 'caruana'),'financeiras')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('bank of america',\
                                                                                             'abn amro bank',\
                                                                                             'capital one bank',\
                                                                                             'credit one',\
                                                                                             'capital one',\
                                                                                             'la banque postale',\
                                                                                             'allied irish banks plc',\
                                                                                             'barclays bank plc',\
                                                                                             'keybank national association',\
                                                                                             'u.s. bank, n.a.',\
                                                                                             'royal bank of canada',\
                                                                                             'bank of ireland',\
                                                                                             'bank simpanan nasional',\
                                                                                             'u.s. bank n.a. nd',\
                                                                                             'usbank',\
                                                                                             'wells fargo bank nevada, n.a.',\
                                                                                             'bank hapoalim',\
                                                                                             'confidence',\
                                                                                             'poste italiane s.p.a. (banca posta)',\
                                                                                             'american express',\
                                                                                             'galicia',\
                                                                                             'americanexpress',\
                                                                                             'caisse nationale des caisses depargne (cnce)',\
                                                                                             'caja de ahorros y monte piedad de segovia (caja segovia)',\
                                                                                             'aval card, s.a. de c.v.',\
                                                                                             'barclaycard',\
                                                                                             'davivienda',\
                                                                                             'high plains',\
                                                                                             'suntrust',\
                                                                                             'halifax',\
                                                                                             'bnp paribas',\
                                                                                             'travelex card services limited',\
                                                                                             'chase',\
                                                                                             'wells fargo',\
                                                                                             'cibc'),'bancos e fin. internacionais')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('banco cooperativo do brasil s/a',\
                                                                                                         'banco csf s/a',\
                                                                                                         'banco votorantim s/a',\
                                                                                                         'banco cetelem s.a',\
                                                                                                         'banco pan sa',\
                                                                                                         'banco cooperativo sicredi sa',\
                                                                                                         'banco bmg s/a',\
                                                                                                         'banco c6 sa',\
                                                                                                         'banco agibank s.a.',\
                                                                                                         'bancoob',\
                                                                                                         'banco triangulo s/a',\
                                                                                                         'banco cbss sa',\
                                                                                                         'banco daycoval s.a.',\
                                                                                                         'banco rodobens s.a.',\
                                                                                                         'banco safra s/a',\
                                                                                                         'banco triangulo',\
                                                                                                         'banco bs2 s.a.',\
                                                                                                         'banco rendimento s.a.',\
                                                                                                         'novo banco continental s.a. - banco multiplo',\
                                                                                                         'pottencial',\
                                                                                                         'panamericano',\
                                                                                                         'bco bonsucesso',\
                                                                                                         'bmb',\
                                                                                                         'safra',\
                                                                                                         'rendimento',\
                                                                                                         'votorantim',\
                                                                                                         'cbss',\
                                                                                                         'bco. ind.brasil',\
                                                                                                         'bco industrial'),'peq.bancos nac. privados')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('pagseguro internet ltda',\
                                                                                         'acesso solucoes de pagamento s.a.',\
                                                                                         'super pagamentos e administracao de meios eletronicos ltda',\
                                                                                         'pagseguro',\
                                                                                         'edenred solucoes de pagamentos hyla s.a.',\
                                                                                         'brasil pre',\
                                                                                         'acg administradora de cartoes sa',\
                                                                                         'cdt(conductor)',\
                                                                                         'qui! card brasil solucoes de pagamento s.a.',\
                                                                                         'paypal'),'solucoes de pagamento')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('banestes',\
                                                                                         'banco do estado do rio grande do sul s.a. (banrisul s.a.)',\
                                                                                         'banrisul',\
                                                                                         'brb',\
                                                                                         'banco de brasilia s.a.',\
                                                                                         'bnb',\
                                                                                         'banco do estado do para s.a.'),'bancos estatais')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('n/a') | F.col('instituicao_cartao_ref_aj').isNull() ,'vazio')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('nu pagamentos sa'),'nu pagamentos sa')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('caixa economica federal'),'caixa economica federal')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('banco inter s.a.'),'banco inter s.a.')\
                               .otherwise(F.when(F.col('instituicao_cartao_ref_aj').isin('banco original'),'banco original')\
                               .otherwise('Outros')))))))))))))))))  

base_final = base.drop('email_domain','device_modelo_ref_aj','device_modelo_ref','instituicao_cartao_ref_aj', 'instituicao_cartao_ref')

# COMMAND ----------

print('-----------------------------------------------------')
print('Tamanho da Base Visão Carteira (Linhas/Colunas): ', (base_final.count(), len(base_final.columns)))
print('-----------------------------------------------------')

# COMMAND ----------

col_list = ['cpf','consumer_id','ref_portfolio','flag_consumer','ref_date',
           #Variáveis adicionais
            'SConcEver60dP6_100','SCORE_C2BA','exclusao',
           # Variáveis categoricas do modelo
            'ddd_state','digital_account_status','bandeira_cartao_ref','email_ajuste','modelo_celular_ref','agrupamento_instituicao_cartao_ref',
           # Variáveis categoricas do modelo
            'days_since_last_transaction','idade_em_anos','tempo_registro_em_meses','qtde_cartoes_ativos_cadastrados','qtde_emissores_distintos','HSPA','peer_effect_score',
            'mental_accounting_score','am_max_pay_12m_trx','ct_min_paypav_12m_trx','ct_tot_payp2pcred_12m_trx','am_tot_payp2pbal_06m_trx','am_max_payp2pbal_12m_trx',
            'am_min_paypavcred_06m_trx','am_tot_paypavcred_12m_trx','am_tot_paypavbal_06m_trx','am_tot_paypavmix_12m_trx','am_max_paypavgood_12m_trx','am_max_recp2p_12m_trx']

base_full = base_final[col_list]

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")
publico = base_full.toPandas()

# Conversões Strings para numéricos 
publico['am_max_pay_12m_trx']         = publico['am_max_pay_12m_trx'].astype(float)
publico['am_tot_payp2pbal_06m_trx']   = publico['am_tot_payp2pbal_06m_trx'].astype(float)
publico['am_max_payp2pbal_12m_trx']   = publico['am_max_payp2pbal_12m_trx'].astype(float)
publico['am_min_paypavcred_06m_trx']  = publico['am_min_paypavcred_06m_trx'].astype(float)
publico['am_tot_paypavcred_12m_trx']  = publico['am_tot_paypavcred_12m_trx'].astype(float)
publico['am_tot_paypavbal_06m_trx']   = publico['am_tot_paypavbal_06m_trx'].astype(float)
publico['am_tot_paypavmix_12m_trx']   = publico['am_tot_paypavmix_12m_trx'].astype(float)
publico['am_max_paypavgood_12m_trx']  = publico['am_max_paypavgood_12m_trx'].astype(float)
publico['am_max_recp2p_12m_trx']      = publico['am_max_recp2p_12m_trx'].astype(float)

# COMMAND ----------

# Variáveis númericas - Substituir null por 0 (Zero)
numericas   = publico.select_dtypes(include=['int','float']).columns.tolist()

for col in numericas:
  publico[col] = pd.to_numeric(publico[col])
  publico[col] = pd.np.where(publico[col].isin([np.inf, -np.inf]), np.nan, publico[col])
  publico[col] = publico[col].fillna(0)
  
# Variáveis categóricas - Substituir null por "outros"
categoricas = publico.select_dtypes(['object']).columns.tolist()

for col in categoricas:
  publico[col] = pd.np.where(publico[col].isin([np.inf, -np.inf]), np.nan, publico[col])
  publico[col] = publico[col].fillna("others")
  publico[col] = publico[col].replace(True, 'True')
  publico[col] = publico[col].replace(False, 'False')
  

# COMMAND ----------

publico.dtypes

# COMMAND ----------

escorada = model_mlflow.predict_proba(publico[dev_x.columns])

publico['score_cs_carteira'] = escorada[:,0]*1000
publico['score_cs_carteira'] = publico['score_cs_carteira'].astype(int)

# COMMAND ----------

publico

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.1. Homologação caso 1

# COMMAND ----------

# Filtros na base full REESCORADA, usando o MLFlow

homologacao_caso1 = publico.where(publico['HSPA'] == 866) 
homologacao_caso1 = homologacao_caso1.dropna(how='all') 

homologacao_caso1 = homologacao_caso1.where(homologacao_caso1['ref_portfolio'] == '2019-11') 
homologacao_caso1 = homologacao_caso1.dropna(how='all') 

homologacao_caso1 = homologacao_caso1.where(homologacao_caso1['idade_em_anos'] == 21) 
homologacao_caso1 = homologacao_caso1.dropna(how='all')

homologacao_caso1 = homologacao_caso1.where(homologacao_caso1['ddd_state'] == 'CE') 
homologacao_caso1 = homologacao_caso1.dropna(how='all')

homologacao_caso1 = homologacao_caso1.where(homologacao_caso1['modelo_celular_ref'] == 'xiaomi') 
homologacao_caso1 = homologacao_caso1.dropna(how='all')

# COMMAND ----------

display(homologacao_caso1)

# COMMAND ----------

homologacao_caso1['score_cs_carteira']

# COMMAND ----------

#Base escorada no notebook inicial
bat_caso1 = tot.where(tot['HSPA'] == 866) 
bat_caso1 = bat_caso1.dropna(how='all') 

bat_caso1 = bat_caso1.where(bat_caso1['idade_em_anos'] == 21) 
bat_caso1 = bat_caso1.dropna(how='all')

bat_caso1 = bat_caso1.where(bat_caso1['ddd_state'] == 'CE') 
bat_caso1 = bat_caso1.dropna(how='all')

bat_caso1 = bat_caso1.where(bat_caso1['modelo_celular_ref'] == 'xiaomi') 
bat_caso1 = bat_caso1.dropna(how='all')

# COMMAND ----------

bat_caso1

# COMMAND ----------

bat_caso1['score_mlflow']

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.1. Homologação caso 2

# COMMAND ----------

# Filtros na base full REESCORADA, usando o MLFlow

homologacao_caso2 = publico.where(publico['HSPA'] == 653) 
homologacao_caso2 = homologacao_caso2.dropna(how='all') 

homologacao_caso2 = homologacao_caso2.where(homologacao_caso2['ref_portfolio'] == '2020-04') 
homologacao_caso2 = homologacao_caso2.dropna(how='all') 

homologacao_caso2 = homologacao_caso2.where(homologacao_caso2['idade_em_anos'] == 28) 
homologacao_caso2 = homologacao_caso2.dropna(how='all')

homologacao_caso2 = homologacao_caso2.where(homologacao_caso2['ddd_state'] == 'MG') 
homologacao_caso2 = homologacao_caso2.dropna(how='all')

homologacao_caso2 = homologacao_caso2.where(homologacao_caso2['modelo_celular_ref'] == 'motorola') 
homologacao_caso2 = homologacao_caso2.dropna(how='all')

# COMMAND ----------

display(homologacao_caso2)

# COMMAND ----------

homologacao_caso2['score_cs_carteira']

# COMMAND ----------

#Base escorada no notebook inicial
bat_caso2 = oot.where(oot['HSPA'] == 653) 
bat_caso2 = bat_caso2.dropna(how='all') 

bat_caso2 = bat_caso2.where(bat_caso2['idade_em_anos'] == 28) 
bat_caso2 = bat_caso2.dropna(how='all')

bat_caso2 = bat_caso2.where(bat_caso2['ddd_state'] == 'MG') 
bat_caso2 = bat_caso2.dropna(how='all')

bat_caso2 = bat_caso2.where(bat_caso2['modelo_celular_ref'] == 'motorola') 
bat_caso2 = bat_caso2.dropna(how='all')

# COMMAND ----------

bat_caso2

# COMMAND ----------

bat_caso2['score_mlflow']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 10. Salva base full escorada no meu S3

# COMMAND ----------

display(publico)

# COMMAND ----------

tabela = pd.DataFrame(publico)
tabela_full = spark.createDataFrame(tabela)

# COMMAND ----------

display(tabela_full)

# COMMAND ----------

tabela_full.count()

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")

tabela_full.write.parquet('s3://picpay-datalake-sandbox/jeobara.zacheski/score_pf_cs/base_full_escorada_cs_carteira/', mode='overwrite')
