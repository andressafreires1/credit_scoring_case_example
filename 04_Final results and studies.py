# Databricks notebook source
# MAGIC %md
# MAGIC ### Projeto: Score PF - Application Visão Carteira
# MAGIC 
# MAGIC ###### Tópicos que serão tratados neste notebook:
# MAGIC 
# MAGIC - 1) Leitura da base escorável após seleção de variáveis finais 
# MAGIC - 2) Aplicação do modelo final 
# MAGIC - 3) Avaliação dos resultados em toda a base (tot, dev, val e oot)
# MAGIC - 4) Avaliação da ordenação em todos os períodos de avaliação
# MAGIC - 5) Avaliação de resultados segmentados por público:
# MAGIC       a. População COM/SEM cartão cadastrado;
# MAGIC       b. Score C2BA <= 360 e C2BA > 360 = Corte de produção, devido a compra da base Serasa pelo Banco Original.
# MAGIC       c. Idade <= 25 anos e > 25 anos
# MAGIC 
# MAGIC ##### Owners: @Jeobara e @João Dias
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

display(selecao_final2.groupBy('am_max_pay_12m_trx','ref_portfolio').count().orderBy(F.desc('count')))

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
# MAGIC # ** 05. Comparativo de ordenação

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

# MAGIC %md
# MAGIC ### 5.1. C2BA

# COMMAND ----------

C2BA_gains_dev = gains_table(baseline=dev,base_ref=dev, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot = gains_table(baseline=dev,base_ref=tot, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val = gains_table(baseline=dev,base_ref=val, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot = gains_table(baseline=dev,base_ref=oot, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev

# COMMAND ----------

C2BA_gains_tot

# COMMAND ----------

C2BA_gains_val

# COMMAND ----------

C2BA_gains_oot

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2. HSPA

# COMMAND ----------

HSPA_gains_dev = gains_table(baseline=dev,base_ref=dev, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot = gains_table(baseline=dev,base_ref=tot, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val = gains_table(baseline=dev,base_ref=val, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot = gains_table(baseline=dev,base_ref=oot, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev

# COMMAND ----------

HSPA_gains_tot

# COMMAND ----------

HSPA_gains_val

# COMMAND ----------

HSPA_gains_oot

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3. Modelo CS Carteira

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
# MAGIC # ** 06. Estudos Adicionais

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1. Público Sem Cartão cadastrado

# COMMAND ----------

# Bases Sem Cartao cadastrado

dev_semcartao = dev.where(dev['qtde_cartoes_ativos_cadastrados'] == 0.0) # Base referência para cálculo dos ranges de score
dev_semcartao = dev_semcartao.dropna(how='all') 

tot_semcartao = tot.where(tot['qtde_cartoes_ativos_cadastrados'] == 0.0) # Apenas aplicar range
tot_semcartao = tot_semcartao.dropna(how='all') 

val_semcartao = val.where(val['qtde_cartoes_ativos_cadastrados'] == 0.0) # Apenas aplicar range
val_semcartao = val_semcartao.dropna(how='all') 

oot_semcartao = oot.where(oot['qtde_cartoes_ativos_cadastrados'] == 0.0) # Apenas aplicar range
oot_semcartao = oot_semcartao.dropna(how='all') 


# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.1.1. C2BA

# COMMAND ----------

C2BA_gains_dev_semcartao = gains_table(baseline=dev_semcartao,base_ref=dev_semcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_semcartao = gains_table(baseline=dev_semcartao,base_ref=tot_semcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_semcartao = gains_table(baseline=dev_semcartao,base_ref=val_semcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_semcartao = gains_table(baseline=dev_semcartao,base_ref=oot_semcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_semcartao

# COMMAND ----------

C2BA_gains_tot_semcartao

# COMMAND ----------

C2BA_gains_val_semcartao

# COMMAND ----------

C2BA_gains_oot_semcartao

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.1.2. HSPA

# COMMAND ----------

HSPA_gains_dev_semcartao = gains_table(baseline=dev_semcartao,base_ref=dev_semcartao, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_semcartao = gains_table(baseline=dev_semcartao,base_ref=tot_semcartao, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_semcartao = gains_table(baseline=dev_semcartao,base_ref=val_semcartao, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_semcartao = gains_table(baseline=dev_semcartao,base_ref=oot_semcartao, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_semcartao

# COMMAND ----------

HSPA_gains_tot_semcartao

# COMMAND ----------

HSPA_gains_val_semcartao

# COMMAND ----------

HSPA_gains_oot_semcartao

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.1.3. CS Carteira

# COMMAND ----------

pp_gains_dev_semcartao = gains_table(baseline=dev_semcartao,base_ref=dev_semcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_semcartao = gains_table(baseline=dev_semcartao,base_ref=tot_semcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_semcartao = gains_table(baseline=dev_semcartao,base_ref=val_semcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_semcartao = gains_table(baseline=dev_semcartao,base_ref=oot_semcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_semcartao

# COMMAND ----------

pp_gains_tot_semcartao

# COMMAND ----------

pp_gains_val_semcartao

# COMMAND ----------

pp_gains_oot_semcartao

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2. Público Com Cartão cadastrado

# COMMAND ----------

# Bases Com Cartao cadastrado

dev_comcartao = dev.where(dev['qtde_cartoes_ativos_cadastrados'] != 0.0) # Base referência para cálculo dos ranges de score
dev_comcartao = dev_comcartao.dropna(how='all') 

tot_comcartao = tot.where(tot['qtde_cartoes_ativos_cadastrados'] != 0.0) # Apenas aplicar range
tot_comcartao = tot_comcartao.dropna(how='all') 

val_comcartao = val.where(val['qtde_cartoes_ativos_cadastrados'] != 0.0) # Apenas aplicar range
val_comcartao = val_comcartao.dropna(how='all') 

oot_comcartao = oot.where(oot['qtde_cartoes_ativos_cadastrados'] != 0.0) # Apenas aplicar range
oot_comcartao = oot_comcartao.dropna(how='all') 


# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2.1. C2BA

# COMMAND ----------

C2BA_gains_dev_comcartao = gains_table(baseline=dev_comcartao,base_ref=dev_comcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_comcartao = gains_table(baseline=dev_comcartao,base_ref=tot_comcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_comcartao = gains_table(baseline=dev_comcartao,base_ref=val_comcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_comcartao = gains_table(baseline=dev_comcartao,base_ref=oot_comcartao, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_comcartao

# COMMAND ----------

C2BA_gains_tot_comcartao

# COMMAND ----------

C2BA_gains_val_comcartao

# COMMAND ----------

C2BA_gains_oot_comcartao

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2.2. HSPA

# COMMAND ----------

HSPA_gains_dev_comcartao = gains_table(baseline=dev_comcartao,base_ref=dev_comcartao, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_comcartao = gains_table(baseline=dev_comcartao,base_ref=tot_comcartao, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_comcartao = gains_table(baseline=dev_comcartao,base_ref=val_comcartao, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_comcartao = gains_table(baseline=dev_comcartao,base_ref=oot_comcartao, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_comcartao

# COMMAND ----------

HSPA_gains_tot_comcartao

# COMMAND ----------

HSPA_gains_val_comcartao

# COMMAND ----------

HSPA_gains_oot_comcartao

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2.3. CS Carteira

# COMMAND ----------

pp_gains_dev_comcartao = gains_table(baseline=dev_comcartao,base_ref=dev_comcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_comcartao = gains_table(baseline=dev_comcartao,base_ref=tot_comcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_comcartao = gains_table(baseline=dev_comcartao,base_ref=val_comcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_comcartao = gains_table(baseline=dev_comcartao,base_ref=oot_comcartao, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_comcartao

# COMMAND ----------

pp_gains_tot_comcartao

# COMMAND ----------

pp_gains_val_comcartao

# COMMAND ----------

pp_gains_oot_comcartao

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3. Score C2BA <= 360

# COMMAND ----------

# Bases Com filtro de Score do C2BA atualmente em produção.
  # Ou seja, scores abaixo desse corte não são encontrados na base de score da Serasa. Só são "comprados" cliente cujo Score do C2BA seja > .

dev_menor360 = dev.where(dev['SCORE_C2BA'] <= 360) # Base referência para cálculo dos ranges de score
dev_menor360 = dev_menor360.dropna(how='all') 

tot_menor360 = tot.where(tot['SCORE_C2BA'] <= 360) # Apenas aplicar range
tot_menor360 = tot_menor360.dropna(how='all') 

val_menor360 = val.where(val['SCORE_C2BA'] <= 360) # Apenas aplicar range
val_menor360 = val_menor360.dropna(how='all') 

oot_menor360 = oot.where(oot['SCORE_C2BA'] <= 360) # Apenas aplicar range
oot_menor360 = oot_menor360.dropna(how='all') 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.3.1. C2BA

# COMMAND ----------

C2BA_gains_dev_menor360 = gains_table(baseline=dev_menor360,base_ref=dev_menor360, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_menor360 = gains_table(baseline=dev_menor360,base_ref=tot_menor360, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_menor360 = gains_table(baseline=dev_menor360,base_ref=val_menor360, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_menor360 = gains_table(baseline=dev_menor360,base_ref=oot_menor360, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_menor360

# COMMAND ----------

C2BA_gains_tot_menor360

# COMMAND ----------

C2BA_gains_val_menor360

# COMMAND ----------

C2BA_gains_oot_menor360

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.3.2. HSPA

# COMMAND ----------

HSPA_gains_dev_menor360 = gains_table(baseline=dev_menor360,base_ref=dev_menor360, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_menor360 = gains_table(baseline=dev_menor360,base_ref=tot_menor360, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_menor360 = gains_table(baseline=dev_menor360,base_ref=val_menor360, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_menor360 = gains_table(baseline=dev_menor360,base_ref=oot_menor360, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_menor360

# COMMAND ----------

HSPA_gains_tot_menor360

# COMMAND ----------

HSPA_gains_val_menor360

# COMMAND ----------

HSPA_gains_oot_menor360

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.3.3. CS Carteira

# COMMAND ----------

pp_gains_dev_menor360 = gains_table(baseline=dev_menor360,base_ref=dev_menor360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_menor360 = gains_table(baseline=dev_menor360,base_ref=tot_menor360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_menor360 = gains_table(baseline=dev_menor360,base_ref=val_menor360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_menor360 = gains_table(baseline=dev_menor360,base_ref=oot_menor360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_menor360

# COMMAND ----------

pp_gains_tot_menor360

# COMMAND ----------

pp_gains_val_menor360

# COMMAND ----------

pp_gains_oot_menor360

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4. Score C2BA > 360

# COMMAND ----------

# Bases Com filtro de Score do C2BA atualmente em produção.
  # Ou seja, scores abaixo desse corte não são encontrados na base de score da Serasa. Só são "comprados" cliente cujo Score do C2BA seja > .

dev_maior360 = dev.where(dev['SCORE_C2BA'] > 360) # Base referência para cálculo dos ranges de score
dev_maior360 = dev_maior360.dropna(how='all') 

tot_maior360 = tot.where(tot['SCORE_C2BA'] > 360) # Apenas aplicar range
tot_maior360 = tot_maior360.dropna(how='all') 

val_maior360 = val.where(val['SCORE_C2BA'] > 360) # Apenas aplicar range
val_maior360 = val_maior360.dropna(how='all') 

oot_maior360 = oot.where(oot['SCORE_C2BA'] > 360) # Apenas aplicar range
oot_maior360 = oot_maior360.dropna(how='all') 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.4.1. C2BA

# COMMAND ----------

C2BA_gains_dev_maior360 = gains_table(baseline=dev_maior360,base_ref=dev_maior360, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_maior360 = gains_table(baseline=dev_maior360,base_ref=tot_maior360, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_maior360 = gains_table(baseline=dev_maior360,base_ref=val_maior360, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_maior360 = gains_table(baseline=dev_maior360,base_ref=oot_maior360, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_maior360

# COMMAND ----------

C2BA_gains_tot_maior360

# COMMAND ----------

C2BA_gains_val_maior360

# COMMAND ----------

C2BA_gains_oot_maior360

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.4.2. HSPA

# COMMAND ----------

HSPA_gains_dev_maior360 = gains_table(baseline=dev_maior360,base_ref=dev_maior360, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_maior360 = gains_table(baseline=dev_maior360,base_ref=tot_maior360, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_maior360 = gains_table(baseline=dev_maior360,base_ref=val_maior360, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_maior360 = gains_table(baseline=dev_maior360,base_ref=oot_maior360, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_maior360

# COMMAND ----------

HSPA_gains_tot_maior360

# COMMAND ----------

HSPA_gains_val_maior360

# COMMAND ----------

HSPA_gains_oot_maior360

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.4.3. CS Carteira

# COMMAND ----------

pp_gains_dev_maior360 = gains_table(baseline=dev_maior360,base_ref=dev_maior360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_maior360 = gains_table(baseline=dev_maior360,base_ref=tot_maior360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_maior360 = gains_table(baseline=dev_maior360,base_ref=val_maior360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_maior360 = gains_table(baseline=dev_maior360,base_ref=oot_maior360, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_maior360

# COMMAND ----------

pp_gains_tot_maior360

# COMMAND ----------

pp_gains_val_maior360

# COMMAND ----------

pp_gains_oot_maior360

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5. Idade <= 25 anos

# COMMAND ----------

dev_idademenor25 = dev.where(dev['idade_em_anos'] <= 25) # Base referência para cálculo dos ranges de score
dev_idademenor25 = dev_idademenor25.dropna(how='all') 

tot_idademenor25 = tot.where(tot['idade_em_anos'] <= 25) # Apenas aplicar range
tot_idademenor25 = tot_idademenor25.dropna(how='all') 

val_idademenor25 = val.where(val['idade_em_anos'] <= 25) # Apenas aplicar range
val_idademenor25 = val_idademenor25.dropna(how='all') 

oot_idademenor25 = oot.where(oot['idade_em_anos'] <= 25) # Apenas aplicar range
oot_idademenor25 = oot_idademenor25.dropna(how='all')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.5.1. C2BA

# COMMAND ----------

C2BA_gains_dev_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=dev_idademenor25, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=tot_idademenor25, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=val_idademenor25, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=oot_idademenor25, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_idademenor25

# COMMAND ----------

C2BA_gains_tot_idademenor25

# COMMAND ----------

C2BA_gains_val_idademenor25

# COMMAND ----------

C2BA_gains_oot_idademenor25

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.5.2. HSPA

# COMMAND ----------

HSPA_gains_dev_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=dev_idademenor25, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=tot_idademenor25, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=val_idademenor25, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=oot_idademenor25, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_idademenor25

# COMMAND ----------

HSPA_gains_tot_idademenor25

# COMMAND ----------

HSPA_gains_val_idademenor25

# COMMAND ----------

HSPA_gains_oot_idademenor25

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.5.3. CS Carteira

# COMMAND ----------

pp_gains_dev_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=dev_idademenor25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=tot_idademenor25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=val_idademenor25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_idademenor25 = gains_table(baseline=dev_idademenor25,base_ref=oot_idademenor25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_idademenor25

# COMMAND ----------

pp_gains_tot_idademenor25

# COMMAND ----------

pp_gains_val_idademenor25

# COMMAND ----------

pp_gains_oot_idademenor25

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.6. Idade > 25 anos

# COMMAND ----------

dev_idademaior25 = dev.where(dev['idade_em_anos'] > 25) # Base referência para cálculo dos ranges de score
dev_idademaior25 = dev_idademaior25.dropna(how='all') 

tot_idademaior25 = tot.where(tot['idade_em_anos'] > 25) # Apenas aplicar range
tot_idademaior25 = tot_idademaior25.dropna(how='all') 

val_idademaior25 = val.where(val['idade_em_anos'] > 25) # Apenas aplicar range
val_idademaior25 = val_idademaior25.dropna(how='all') 

oot_idademaior25 = oot.where(oot['idade_em_anos'] > 25) # Apenas aplicar range
oot_idademaior25 = oot_idademaior25.dropna(how='all')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.6.1. C2BA

# COMMAND ----------

C2BA_gains_dev_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=dev_idademaior25, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=tot_idademaior25, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=val_idademaior25, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=oot_idademaior25, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_idademaior25

# COMMAND ----------

C2BA_gains_tot_idademaior25

# COMMAND ----------

C2BA_gains_val_idademaior25

# COMMAND ----------

C2BA_gains_oot_idademaior25

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.6.2. HSPA

# COMMAND ----------

HSPA_gains_dev_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=dev_idademaior25, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=tot_idademaior25, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=val_idademaior25, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=oot_idademaior25, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_idademaior25

# COMMAND ----------

HSPA_gains_tot_idademaior25

# COMMAND ----------

HSPA_gains_val_idademaior25

# COMMAND ----------

HSPA_gains_oot_idademaior25

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.6.3. CS Carteira

# COMMAND ----------

pp_gains_dev_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=dev_idademaior25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=tot_idademaior25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=val_idademaior25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_idademaior25 = gains_table(baseline=dev_idademaior25,base_ref=oot_idademaior25, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_idademaior25

# COMMAND ----------

pp_gains_tot_idademaior25

# COMMAND ----------

pp_gains_val_idademaior25

# COMMAND ----------

pp_gains_oot_idademaior25

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.7. Sem transação nos últimos 12 meses

# COMMAND ----------

dev_semtrx = dev.where(dev['am_max_pay_12m_trx'].isin([np.inf, -np.inf,np.nan,0])) # Base referência para cálculo dos ranges de score
dev_semtrx = dev_semtrx.dropna(how='all') 

tot_semtrx = tot.where(tot['am_max_pay_12m_trx'].isin([np.inf, -np.inf,np.nan,0])) # Apenas aplicar range
tot_semtrx = tot_semtrx.dropna(how='all') 

val_semtrx = val.where(val['am_max_pay_12m_trx'].isin([np.inf, -np.inf,np.nan,0])) # Apenas aplicar range
val_semtrx = val_semtrx.dropna(how='all') 

oot_semtrx = oot.where(oot['am_max_pay_12m_trx'].isin([np.inf, -np.inf,np.nan,0])) # Apenas aplicar range
oot_semtrx = oot_semtrx.dropna(how='all')


# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.7.1. C2BA

# COMMAND ----------

C2BA_gains_dev_semtrx = gains_table(baseline=dev_semtrx,base_ref=dev_semtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_semtrx = gains_table(baseline=dev_semtrx,base_ref=tot_semtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_semtrx = gains_table(baseline=dev_semtrx,base_ref=val_semtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_semtrx = gains_table(baseline=dev_semtrx,base_ref=oot_semtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_semtrx

# COMMAND ----------

C2BA_gains_tot_semtrx

# COMMAND ----------

C2BA_gains_val_semtrx

# COMMAND ----------

C2BA_gains_oot_semtrx

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.7.2. HSPA

# COMMAND ----------

HSPA_gains_dev_semtrx = gains_table(baseline=dev_semtrx,base_ref=dev_semtrx, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_semtrx = gains_table(baseline=dev_semtrx,base_ref=tot_semtrx, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_semtrx = gains_table(baseline=dev_semtrx,base_ref=val_semtrx, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_semtrx = gains_table(baseline=dev_semtrx,base_ref=oot_semtrx, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_semtrx

# COMMAND ----------

HSPA_gains_tot_semtrx

# COMMAND ----------

HSPA_gains_val_semtrx

# COMMAND ----------

HSPA_gains_oot_semtrx

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.7.3. CS Carteira

# COMMAND ----------

pp_gains_dev_semtrx = gains_table(baseline=dev_semtrx,base_ref=dev_semtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_semtrx = gains_table(baseline=dev_semtrx,base_ref=tot_semtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_semtrx = gains_table(baseline=dev_semtrx,base_ref=val_semtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_semtrx = gains_table(baseline=dev_semtrx,base_ref=oot_semtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_semtrx

# COMMAND ----------

pp_gains_tot_semtrx

# COMMAND ----------

pp_gains_val_semtrx

# COMMAND ----------

pp_gains_oot_semtrx

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.8. Com transação nos últimos 12 meses

# COMMAND ----------

dev_comtrx = dev.where(dev['am_max_pay_12m_trx'] > 0) # Base referência para cálculo dos ranges de score
dev_comtrx = dev_comtrx.dropna(how='all') 

tot_comtrx = tot.where(tot['am_max_pay_12m_trx'] > 0) # Apenas aplicar range
tot_comtrx = tot_comtrx.dropna(how='all') 

val_comtrx = val.where(val['am_max_pay_12m_trx'] > 0) # Apenas aplicar range
val_comtrx = val_comtrx.dropna(how='all') 

oot_comtrx = oot.where(oot['am_max_pay_12m_trx'] > 0) # Apenas aplicar range
oot_comtrx = oot_comtrx.dropna(how='all')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.8.1. C2BA

# COMMAND ----------

C2BA_gains_dev_comtrx = gains_table(baseline=dev_comtrx,base_ref=dev_comtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_tot_comtrx = gains_table(baseline=dev_comtrx,base_ref=tot_comtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val_comtrx = gains_table(baseline=dev_comtrx,base_ref=val_comtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_oot_comtrx = gains_table(baseline=dev_comtrx,base_ref=oot_comtrx, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev_comtrx

# COMMAND ----------

C2BA_gains_tot_comtrx

# COMMAND ----------

C2BA_gains_val_comtrx

# COMMAND ----------

C2BA_gains_oot_comtrx

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.8.2. HSPA

# COMMAND ----------

HSPA_gains_dev_comtrx = gains_table(baseline=dev_comtrx,base_ref=dev_comtrx, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_tot_comtrx = gains_table(baseline=dev_comtrx,base_ref=tot_comtrx, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val_comtrx = gains_table(baseline=dev_comtrx,base_ref=val_comtrx, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_oot_comtrx = gains_table(baseline=dev_comtrx,base_ref=oot_comtrx, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev_comtrx

# COMMAND ----------

HSPA_gains_tot_comtrx

# COMMAND ----------

HSPA_gains_val_comtrx

# COMMAND ----------

HSPA_gains_oot_comtrx

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.8.3. CS Carteira

# COMMAND ----------

pp_gains_dev_comtrx = gains_table(baseline=dev_comtrx,base_ref=dev_comtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_tot_comtrx = gains_table(baseline=dev_comtrx,base_ref=tot_comtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_val_comtrx = gains_table(baseline=dev_comtrx,base_ref=val_comtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")
pp_gains_oot_comtrx = gains_table(baseline=dev_comtrx,base_ref=oot_comtrx, target="SConcEver60dP6_100", prob="score_cs_carteira_0")

# COMMAND ----------

pp_gains_dev_comtrx

# COMMAND ----------

pp_gains_tot_comtrx

# COMMAND ----------

pp_gains_val_comtrx

# COMMAND ----------

pp_gains_oot_comtrx