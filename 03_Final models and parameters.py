# Databricks notebook source
# MAGIC %md
# MAGIC ### Projeto: Score PF - Application Visão Carteira
# MAGIC 
# MAGIC ###### Tópicos que serão tratados neste notebook:
# MAGIC 
# MAGIC - 1) Leitura da base escorável após seleção de variáveis finais 
# MAGIC - 2) Aplicação dos melhores modelos, obtidos após aplicação do método optuna. 
# MAGIC - 3) Avaliação dos resultados em toda a base (tot, dev, val e oot)
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

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 600)

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 02 . Funções ----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Ordenação e KS

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
# MAGIC ### 2.3. IV

# COMMAND ----------

def get_IV(df, feature, target, cat = True, q = 10):
  if cat == True:
      lst = []
    # optional
    # df[feature] = df[feature].fillna("NULL")
      unique_values = df[feature].unique()
      for val in unique_values:
          lst.append([feature,                                                        # Feature name
                      val,                                                            # Value of a feature (unique)
                      df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (Fraud == 0)
                      df[(df[feature] == val) & (df[target] == 1)].count()[feature]   # Bad  (Fraud == 1)
                     ])
      data = pd.DataFrame(lst, columns=['Variable', 'Value', 'Good', 'Bad'])
      total_bad = df[df[target] == 1].count()[feature]
      total_good = df.shape[0] - total_bad
      data['Distribution Good'] = data['Good']/ total_good
      data['Distribution Bad'] = data['Bad'] / total_bad
      data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
      data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
      data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
      data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
      data.index = range(len(data.index))
      iv = data['IV'].sum()
  else:
      data = pd.crosstab(pd.cut(df[feature], q), df[target])
      data.columns = ['0', '1']
      total_bad = data['1'].sum()
      total_good = data['0'].sum()
      data['Distribution Good'] = data['0']/ total_good
      data['Distribution Bad'] = data['1'] / total_bad
      data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
      data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
      data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
      iv = data['IV'].sum()
  return iv

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 03. Base Application Carteira ------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Seleção das variáveis finais

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")

#Base total de new consumers = : 
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

#display(selecao_final2.groupBy('modelo_celular_D0').count().orderBy(F.desc('count')))
#modelo_celular_D0
#device_D0

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

# MAGIC %md
# MAGIC ### 3.2. Bases (Dev + Val + Oot)

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

## Cria a matriz de correlação
corr = x_base_dev.corr().abs()
corr

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 04.  Modelos finalistas -------------------------------------------------------------

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
# MAGIC ##### 4.1. XGB - Extreme Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAGIC 
# MAGIC HSPA + INTERNAS

# COMMAND ----------

# Treinando com os parâmetros escolhidos
XGB = Pipeline(steps = [('preprocessor_xgb', preprocessor),
                        ('model_xgb',XGBClassifier(random_state = 123, n_estimators = 300, max_depth = 3, reg_alpha = 0.9780178933771676,
                                                  reg_lambda = 0.003254424511620696, gamma = 5, learning_rate = 0.2002553424989827, 
                                                  objective = 'binary:logistic', subsample = 0.9))]) 
XGB.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = XGB.predict_proba(dev_x)[:,1]
y_predict_val = XGB.predict_proba(val_x)[:,1]
y_predict_tot = XGB.predict_proba(tot_x)[:,1]
y_predict_oot = XGB.predict_proba(oot_x)[:,1]

proba_dev = XGB.predict_proba(dev_x)
proba_val = XGB.predict_proba(val_x)
proba_tot = XGB.predict_proba(tot_x)
proba_oot = XGB.predict_proba(oot_x)

# COMMAND ----------

# Total Desenvolvimento - 100%
skplt.metrics.plot_ks_statistic(tot_y['SConcEver60dP6_100'], proba_tot)
display(plt.show())

# COMMAND ----------

# Desenvolvimento - 80%
skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

# Validação 20%
skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

# Out-of-Time
skplt.metrics.plot_ks_statistic(oot_y['SConcEver60dP6_100'], proba_oot)
display(plt.show())

# COMMAND ----------

# Total Desenvolvimento - 100%
gini_tot = (2*roc_auc_score(tot_y['SConcEver60dP6_100'],y_predict_tot)) - 1

# Desenvolvimento - 80%
gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1

# Validação - 20%
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

# Out-of-Time
gini_oot = (2*roc_auc_score(oot_y['SConcEver60dP6_100'],y_predict_oot)) - 1

# COMMAND ----------

gini_tot, gini_dev, gini_val, gini_oot

# COMMAND ----------

#Features Importance (Gráfico em Azul)
features = num_columns+cat_columns
importances = XGB.named_steps['model_xgb'].feature_importances_
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
# MAGIC ##### 4.2. LGB - Light Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

# Total Desenvolvimento - 100%
skplt.metrics.plot_ks_statistic(tot_y['SConcEver60dP6_100'], proba_tot)
display(plt.show())

# COMMAND ----------

# Desenvolvimento - 80%
skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

# Validação 20%
skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

# Out-of-Time
skplt.metrics.plot_ks_statistic(oot_y['SConcEver60dP6_100'], proba_oot)
display(plt.show())

# COMMAND ----------

# Total Desenvolvimento - 100%
gini_tot = (2*roc_auc_score(tot_y['SConcEver60dP6_100'],y_predict_tot)) - 1

# Desenvolvimento - 80%
gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1

# Validação - 20%
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

# Out-of-Time
gini_oot = (2*roc_auc_score(oot_y['SConcEver60dP6_100'],y_predict_oot)) - 1

# COMMAND ----------

gini_tot, gini_dev, gini_val, gini_oot

# COMMAND ----------

#Features Importance (Gráfico colorido)
#Features Importance (Gráfico em Azul)
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
# MAGIC ##### 4.2. GBM - Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAGIC 
# MAGIC HSPA + INTERNAS

# COMMAND ----------

# Treinando com os parâmetros escolhidos
GBM = Pipeline(steps = [('preprocessor_gbm', preprocessor),
                        ('model_gbm',GradientBoostingClassifier(random_state = 123, loss = 'exponential', n_estimators = 450, max_depth = 3, learning_rate = 0.09897056270235999, subsample = 0.8500000000000001,validation_fraction = 0.1))]) 
GBM.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = GBM.predict_proba(dev_x)[:,1]
y_predict_val = GBM.predict_proba(val_x)[:,1]
y_predict_tot = GBM.predict_proba(tot_x)[:,1]
y_predict_oot = GBM.predict_proba(oot_x)[:,1]

proba_dev = GBM.predict_proba(dev_x)
proba_val = GBM.predict_proba(val_x)
proba_tot = GBM.predict_proba(tot_x)
proba_oot = GBM.predict_proba(oot_x)

# COMMAND ----------

# Total Desenvolvimento - 100%
skplt.metrics.plot_ks_statistic(tot_y['SConcEver60dP6_100'], proba_tot)
display(plt.show())

# COMMAND ----------

# Desenvolvimento - 80%
skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

# Validação 20%
skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

# Out-of-Time
skplt.metrics.plot_ks_statistic(oot_y['SConcEver60dP6_100'], proba_oot)
display(plt.show())

# COMMAND ----------

# Total Desenvolvimento - 100%
gini_tot = (2*roc_auc_score(tot_y['SConcEver60dP6_100'],y_predict_tot)) - 1

# Desenvolvimento - 80%
gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1

# Validação - 20%
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

# Out-of-Time
gini_oot = (2*roc_auc_score(oot_y['SConcEver60dP6_100'],y_predict_oot)) - 1

# COMMAND ----------

gini_tot, gini_dev, gini_val, gini_oot

# COMMAND ----------

#Features Importance (Gráfico colorido)
#Features Importance (Gráfico em Azul)
features = num_columns+cat_columns
importances = GBM.named_steps['model_gbm'].feature_importances_
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