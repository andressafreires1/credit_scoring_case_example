# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###### Tópicos que serão tratados neste notebook:
# MAGIC 
# MAGIC - 1) Leitura da base escorável com a seleção de variáveis finais 
# MAGIC - 2) Testes de modelos iniciais
# MAGIC - 3) Grid Search
# MAGIC - 4) Optuna
# MAGIC - 5) Escolha do modelo final
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
from sklearn.metrics           import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, log_loss, roc_curve
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
# MAGIC # ** 03. Base variáveis New Consumer ------------------------------------------------------

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
            'H2PA')

# COMMAND ----------

## Segmentação do grupo que será modelado
tmp = selecao_final2.toPandas() 

# Conversões Strings para numéricos 
tmp['am_max_pay_06m_trx']         = tmp['am_max_pay_06m_trx'].astype(float)
tmp['am_avg_pay_06m_trx']         = tmp['am_avg_pay_06m_trx'].astype(float)
tmp['am_max_pay_12m_trx']         = tmp['am_max_pay_12m_trx'].astype(float)
tmp['am_tot_pay_12m_trx']         = tmp['am_tot_pay_12m_trx'].astype(float)
tmp['am_avg_pay_12m_trx']         = tmp['am_avg_pay_12m_trx'].astype(float)
tmp['am_max_payp2p_12m_trx']      = tmp['am_max_payp2p_12m_trx'].astype(float)
tmp['am_tot_payp2p_12m_trx']      = tmp['am_tot_payp2p_12m_trx'].astype(float)
tmp['am_avg_payp2p_12m_trx']      = tmp['am_avg_payp2p_12m_trx'].astype(float)
tmp['am_min_paypav_06m_trx']      = tmp['am_min_paypav_06m_trx'].astype(float)
tmp['am_max_paypav_12m_trx']      = tmp['am_max_paypav_12m_trx'].astype(float)
tmp['am_tot_paypav_12m_trx']      = tmp['am_tot_paypav_12m_trx'].astype(float)
tmp['am_avg_paypav_12m_trx']      = tmp['am_avg_paypav_12m_trx'].astype(float)
tmp['am_max_payp2pcred_03m_trx']  = tmp['am_max_payp2pcred_03m_trx'].astype(float)
tmp['am_tot_payp2pcred_12m_trx']  = tmp['am_tot_payp2pcred_12m_trx'].astype(float)
tmp['am_avg_payp2pcred_12m_trx']  = tmp['am_avg_payp2pcred_12m_trx'].astype(float)
tmp['am_min_payp2pbal_06m_trx']   = tmp['am_min_payp2pbal_06m_trx'].astype(float)
tmp['am_tot_payp2pbal_06m_trx']   = tmp['am_tot_payp2pbal_06m_trx'].astype(float)
tmp['am_max_payp2pbal_12m_trx']   = tmp['am_max_payp2pbal_12m_trx'].astype(float)
tmp['am_avg_payp2pbal_12m_trx']   = tmp['am_avg_payp2pbal_12m_trx'].astype(float)
tmp['am_min_paypavcred_06m_trx']  = tmp['am_min_paypavcred_06m_trx'].astype(float)
tmp['am_max_paypavcred_06m_trx']  = tmp['am_max_paypavcred_06m_trx'].astype(float)
tmp['am_tot_paypavcred_12m_trx']  = tmp['am_tot_paypavcred_12m_trx'].astype(float)
tmp['am_avg_paypavcred_12m_trx']  = tmp['am_avg_paypavcred_12m_trx'].astype(float)
tmp['am_tot_paypavbal_06m_trx']   = tmp['am_tot_paypavbal_06m_trx'].astype(float)
tmp['am_avg_paypavbal_06m_trx']   = tmp['am_avg_paypavbal_06m_trx'].astype(float)
tmp['am_tot_paypavmix_12m_trx']   = tmp['am_tot_paypavmix_12m_trx'].astype(float)
tmp['am_avg_paypavmix_12m_trx']   = tmp['am_avg_paypavmix_12m_trx'].astype(float)
tmp['am_max_paypavbill_12m_trx']  = tmp['am_max_paypavbill_12m_trx'].astype(float)
tmp['am_tot_paypavbill_12m_trx']  = tmp['am_tot_paypavbill_12m_trx'].astype(float)
tmp['am_avg_paypavbill_12m_trx']  = tmp['am_avg_paypavbill_12m_trx'].astype(float)
tmp['am_tot_paypavgood_06m_trx']  = tmp['am_tot_paypavgood_06m_trx'].astype(float)
tmp['am_min_paypavgood_12m_trx']  = tmp['am_min_paypavgood_12m_trx'].astype(float)
tmp['am_max_paypavgood_12m_trx']  = tmp['am_max_paypavgood_12m_trx'].astype(float)
tmp['am_tot_paypavgood_12m_trx']  = tmp['am_tot_paypavgood_12m_trx'].astype(float)
tmp['am_avg_paypavgood_12m_trx']  = tmp['am_avg_paypavgood_12m_trx'].astype(float)
tmp['am_tot_recp2p_03m_trx']      = tmp['am_tot_recp2p_03m_trx'].astype(float)
tmp['am_avg_recp2p_03m_trx']      = tmp['am_avg_recp2p_03m_trx'].astype(float)
tmp['am_max_recp2p_12m_trx']      = tmp['am_max_recp2p_12m_trx'].astype(float)
tmp['am_max_recp2pcred_12m_trx']  = tmp['am_max_recp2pcred_12m_trx'].astype(float)


# Separação em Desenvolvimento (100%) e Out-of-Time
base_tot = tmp.where(tmp['ref_portfolio'] != '2020-04')
base_tot = base_tot.dropna(how='all') 

base_oot = tmp.where(tmp['ref_portfolio'] == '2020-04')
base_oot = base_oot.dropna(how='all') 

# Transformando as variáveis categoricas em dummies (k - 1 categoricas, one hot enconding)
#df_proc = pd.get_dummies(base_tot, drop_first = True, prefix_sep = ':')

# COMMAND ----------

#display(selecao_final2.groupBy('changed_username_times').count().orderBy(F.desc('count'))) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 04. Testes Modelos -------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1. Bases (Dev + Val + Oot) - Pós Seleção

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

print (base_dev.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 4.2. Modelo Quick Dirty

# COMMAND ----------

cat_columns = x_base_dev.select_dtypes(include=['object']).columns.tolist()
num_columns = x_base_dev.select_dtypes(include=['int','float','float32','float64']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
model = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(n_jobs=-1, random_state=123, class_weight='balanced', max_depth=7))])
model.fit(x_base_dev, y_base_dev)

# COMMAND ----------

cat_columns

# COMMAND ----------

num_columns

# COMMAND ----------

y_proba_dev = model.predict_proba(x_base_dev)
y_proba_val = model.predict_proba(base_val.drop(resposta_final, axis=1))

# COMMAND ----------

from scipy.stats import ks_2samp

df_results = pd.DataFrame({'class':y_base_dev, 'proba0': y_proba_dev[:,0], 'proba1': y_proba_dev[:,1]})
proba1 = df_results[df_results['class']==0]['proba1']
proba0 = df_results[df_results['class']==1]['proba1']

# COMMAND ----------

ks_2samp(proba1, proba0)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(y_base_dev, y_proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(y_base_val, y_proba_val)
display(plt.show())

# COMMAND ----------

features = num_columns+cat_columns
importances = model.named_steps['model'].feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(15,35))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
display(plt.show())

# COMMAND ----------

xx = importances[indices]
yy = [features[i] for i in indices]
df_importances = pd.DataFrame({'Columns':yy, 'Importances':xx})

df_importances.columns = ['Columns','Importances']
df_importances = df_importances.sort_values(by = 'Importances', ascending = False)

sns.set(rc={'figure.figsize':(5,6)})
sns.barplot(y="Columns", x="Importances", data=df_importances)

# COMMAND ----------

df_importances

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 4.3. Matriz de correlação

# COMMAND ----------

## Cria a matriz de correlação
corr = x_base_dev.corr().abs()
corr

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 05. Modelos Serasa -------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.1. C2BA

# COMMAND ----------

C2BA_gains_dev = gains_table(baseline=base_dev,base_ref=base_dev, target="SConcEver60dP6_100", prob="SCORE_C2BA")
C2BA_gains_val = gains_table(baseline=base_dev,base_ref=base_val, target="SConcEver60dP6_100", prob="SCORE_C2BA")

# COMMAND ----------

C2BA_gains_dev

# COMMAND ----------

C2BA_gains_val

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.2. H2PA

# COMMAND ----------

H2PA_gains_dev = gains_table(baseline=base_dev,base_ref=base_dev, target="SConcEver60dP6_100", prob="H2PA")
H2PA_gains_val = gains_table(baseline=base_dev,base_ref=base_val, target="SConcEver60dP6_100", prob="H2PA")

# COMMAND ----------

H2PA_gains_dev

# COMMAND ----------

H2PA_gains_val

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.3. HSPA

# COMMAND ----------

HSPA_gains_dev = gains_table(baseline=base_dev,base_ref=base_dev, target="SConcEver60dP6_100", prob="HSPA")
HSPA_gains_val = gains_table(baseline=base_dev,base_ref=base_val, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

HSPA_gains_dev

# COMMAND ----------

HSPA_gains_val

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 05.  Optuna (Grid Search) ------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.1. XGB - Extreme Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.1.1. INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: 
def fitXGB(trial):
  "Train XGBOOST Model"
  
  objective_list_reg = ['binary:logistic']
  params ={'model_xgb__n_estimators': trial.suggest_int('n_estimators', 50, 500, 25), 
           'model_xgb__max_depth':trial.suggest_int('max_depth', 1, 3, 1),   
           'model_xgb__reg_alpha':trial.suggest_loguniform('reg_alpha', 0.001, 1),
           'model_xgb__reg_lambda':trial.suggest_loguniform('reg_lambda', 0.001, 1),
           'model_xgb__gamma':trial.suggest_int('gamma', 0, 8),
           'model_xgb__learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
           'model_xgb__objective':trial.suggest_categorical('objective', objective_list_reg),
           'model_xgb__subsample':trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.05),
  }
  
  XGB = Pipeline(steps = [('preprocessor_xgb', preprocessor),('model_xgb', XGBClassifier(random_state = 123))])  
  XGB.set_params(**params)
 
  XGB.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = XGB.predict_proba(val_x)[:,1]
  # roc_auc = roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  # print(f'SCORE DE AREA UNDER THE CURVE: {roc_auc}')
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc

# COMMAND ----------

# Fazendo a otimização
study = optuna.create_study(direction = 'maximize')
study.optimize(fitXGB, n_trials = 50) 

# COMMAND ----------

print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
XGB = Pipeline(steps = [('preprocessor_xgb', preprocessor),
                        ('model_xgb',XGBClassifier(random_state = 123, n_estimators = 400, max_depth = 3, reg_alpha = 0.0955837617400097,
                                                  reg_lambda = 0.0011334336756423263, gamma = 1, learning_rate = 0.15790883742640568, 
                                                  objective = 'binary:logistic', subsample = 1.0))]) 
XGB.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = XGB.predict_proba(dev_x)[:,1]
y_predict_val = XGB.predict_proba(val_x)[:,1]

proba_dev = XGB.predict_proba(dev_x)
proba_val = XGB.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.1.2. HSPA + INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: 
def fitXGB(trial):
  "Train XGBOOST Model"
  
  objective_list_reg = ['binary:logistic']
  params ={'model_xgb__n_estimators': trial.suggest_int('n_estimators', 50, 500, 25), 
           'model_xgb__max_depth':trial.suggest_int('max_depth', 1, 3, 1),   
           'model_xgb__reg_alpha':trial.suggest_loguniform('reg_alpha', 0.001, 1),
           'model_xgb__reg_lambda':trial.suggest_loguniform('reg_lambda', 0.001, 1),
           'model_xgb__gamma':trial.suggest_int('gamma', 0, 8),
           'model_xgb__learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
           'model_xgb__objective':trial.suggest_categorical('objective', objective_list_reg),
           'model_xgb__subsample':trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.05)
  }
  
  XGB = Pipeline(steps = [('preprocessor_xgb', preprocessor),('model_xgb', XGBClassifier(random_state = 123))])  
  XGB.set_params(**params)
 
  XGB.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = XGB.predict_proba(val_x)[:,1]
  # roc_auc = roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  # print(f'SCORE DE AREA UNDER THE CURVE: {roc_auc}')
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
study = optuna.create_study(direction = 'maximize')
study.optimize(fitXGB, n_trials = 50)

# COMMAND ----------

print(study.best_trial)

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

proba_dev = XGB.predict_proba(dev_x)
proba_val = XGB.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.2. RF - Random Forest ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.2.1. INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])


# COMMAND ----------

# Função de treino de parâmetros: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
def fitRF(trial):
  "Train RF Model"
  
  criterion_list = ['gini', 'entropy']
  class_weight_list = ['balanced', 'balanced_subsample']  
  params = {'model_rf__criterion': trial.suggest_categorical('criterion',criterion_list),
            'model_rf__n_estimators': trial.suggest_int('n_estimators', 50, 500, 25),
            'model_rf__max_features': trial.suggest_int('max_features', 2, 10, 2),
            'model_rf__min_samples_split': trial.suggest_int('min_samples_split', 300, 900, 50),
            'model_rf__max_samples': trial.suggest_discrete_uniform('max_samples', 0.5, 0.95, 0.05),
            'model_rf__class_weight': trial.suggest_categorical('class_weight',class_weight_list)          
  }
  
  RF = Pipeline(steps = [('preprocessor_rf', preprocessor),('model_rf', RandomForestClassifier(random_state = 123))])  
  RF.set_params(**params)
 
  RF.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = RF.predict_proba(val_x)[:,1]
  # roc_auc = roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  # print(f'SCORE DE AREA UNDER THE CURVE: {roc_auc}')
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
study = optuna.create_study(direction = 'maximize')
study.optimize(fitRF, n_trials = 50) 

# COMMAND ----------

print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
RF = Pipeline(steps = [('preprocessor_rf', preprocessor),
                       ('model_rf', RandomForestClassifier(random_state = 123, criterion = 'gini', n_estimators = 475, max_features = 10, min_samples_split = 600, 
                                                            max_samples = 0.95, class_weight = 'balanced'))]) 
RF.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = RF.predict_proba(dev_x)[:,1]
y_predict_val = RF.predict_proba(val_x)[:,1]

proba_dev = RF.predict_proba(dev_x)
proba_val = RF.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.2.2. HSPA + INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
def fitRF(trial):
  "Train RF Model"
  
  criterion_list = ['gini', 'entropy']
  class_weight_list = ['balanced', 'balanced_subsample']  
  params = {'model_rf__criterion': trial.suggest_categorical('criterion',criterion_list),
            'model_rf__n_estimators': trial.suggest_int('n_estimators', 50, 500, 25),
            'model_rf__max_features': trial.suggest_int('max_features', 2, 10, 2),
            'model_rf__min_samples_split': trial.suggest_int('min_samples_split', 300, 900, 50),
            'model_rf__max_samples': trial.suggest_discrete_uniform('max_samples', 0.5, 0.95, 0.05),
            'model_rf__class_weight': trial.suggest_categorical('class_weight',class_weight_list)          
  }
  
  RF = Pipeline(steps = [('preprocessor_rf', preprocessor),('model_rf', RandomForestClassifier(random_state = 123))])  
  RF.set_params(**params)
 
  RF.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = RF.predict_proba(val_x)[:,1]
  # roc_auc = roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  # print(f'SCORE DE AREA UNDER THE CURVE: {roc_auc}')
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
study = optuna.create_study(direction = 'maximize')
study.optimize(fitRF, n_trials = 50) 

# COMMAND ----------

print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
RF = Pipeline(steps = [('preprocessor_rf', preprocessor),
                        ('model_rf', RandomForestClassifier(random_state = 123, criterion = 'gini', n_estimators = 450, max_features = 10, min_samples_split = 300, 
                                                            max_samples = 0.75, class_weight = 'balanced_subsample'))]) 
RF.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = RF.predict_proba(dev_x)[:,1]
y_predict_val = RF.predict_proba(val_x)[:,1]

proba_dev = RF.predict_proba(dev_x)
proba_val = RF.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.3. LGB - Light Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.3.1. INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
def fitLGB(trial):
  "Train LightGBM Model"
   
  boosting_type_list = ['gbdt', 'dart', 'goss']  
  class_weight_list = ['balanced']
  params = {'model_lgb__boosting_type':trial.suggest_categorical('boosting_type',boosting_type_list),
            'model_lgb__n_estimators':trial.suggest_int('n_estimators', 50, 500, 25),
            'model_lgb__max_depth':trial.suggest_int('max_depth', 1, 3, 1), 
            'model_lgb__learning_rate':trial.suggest_loguniform('learning_rate',0.005, 0.5),
            'model_lgb__class_weight':trial.suggest_categorical('class_weight',class_weight_list),
            'model_lgb__reg_alpha':trial.suggest_loguniform('reg_alpha', 0.001, 1),
            'model_lgb__reg_lambda':trial.suggest_loguniform('reg_lambda', 0.001, 1),                    
            'model_lgb__subsample': trial.suggest_discrete_uniform('subsample', 0.5, 0.95, 0.05)
  }
   
  LGB = Pipeline(steps = [('preprocessor_lgb', preprocessor),('model_lgb', LGBMClassifier(random_state = 123))])  
  LGB.set_params(**params)
 
  LGB.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = LGB.predict_proba(val_x)[:,1]
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  # print(f'SCORE DE AREA UNDER THE CURVE: {roc_auc}')
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
  # RODEI EM UM NOTEBOOK PARALELO, TEMPO DE PROCESSAMENTO = 17.12 minutes
#study = optuna.create_study(direction = 'maximize')
#study.optimize(fitLGB, n_trials = 50)  

# COMMAND ----------

#print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
LGB = Pipeline(steps = [('preprocessor_lgb', preprocessor),
                        ('model_lgb',LGBMClassifier(random_state = 123, boosting_type = 'gbdt', max_depth = 2, n_estimators = 500, learning_rate = 0.3395729230943446,
                                                    class_weight = 'balanced', reg_alpha = 0.048031391687675194, reg_lambda = 0.013998523657456236, subsample = 0.8500000000000001))]) 
LGB.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = LGB.predict_proba(dev_x)[:,1]
y_predict_val = LGB.predict_proba(val_x)[:,1]

proba_dev = LGB.predict_proba(dev_x)
proba_val = LGB.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.3.2. HSPA + INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
def fitLGB(trial):
  "Train LightGBM Model"
   
  boosting_type_list = ['gbdt', 'dart', 'goss']  
  class_weight_list = ['balanced']
  params = {'model_lgb__boosting_type':trial.suggest_categorical('boosting_type',boosting_type_list),
            'model_lgb__n_estimators':trial.suggest_int('n_estimators', 50, 500, 25),
            'model_lgb__max_depth':trial.suggest_int('max_depth', 1, 3, 1), 
            'model_lgb__learning_rate':trial.suggest_loguniform('learning_rate',0.005, 0.5),
            'model_lgb__class_weight':trial.suggest_categorical('class_weight',class_weight_list),
            'model_lgb__reg_alpha':trial.suggest_loguniform('reg_alpha', 0.001, 1),
            'model_lgb__reg_lambda':trial.suggest_loguniform('reg_lambda', 0.001, 1),                    
            'model_lgb__subsample': trial.suggest_discrete_uniform('subsample', 0.5, 0.95, 0.05)
  }
   
  LGB = Pipeline(steps = [('preprocessor_lgb', preprocessor),('model_lgb', LGBMClassifier(random_state = 123))])  
  LGB.set_params(**params)
 
  LGB.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = LGB.predict_proba(val_x)[:,1]
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  # print(f'SCORE DE AREA UNDER THE CURVE: {roc_auc}')
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
study = optuna.create_study(direction = 'maximize')
study.optimize(fitLGB, n_trials = 50)  

# COMMAND ----------

print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
LGB = Pipeline(steps = [('preprocessor_lgb', preprocessor),
                        ('model_lgb',LGBMClassifier(random_state = 123, boosting_type = 'dart', max_depth = 3, n_estimators = 175, learning_rate = 0.4940814646340456,
                                                    class_weight = 'balanced', reg_alpha = 0.02399020943974248, reg_lambda = 0.014865002953056582, subsample = 0.75))]) 
LGB.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = LGB.predict_proba(dev_x)[:,1]
y_predict_val = LGB.predict_proba(val_x)[:,1]

proba_dev = LGB.predict_proba(dev_x)
proba_val = LGB.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.4. GBM - Gradient Boosting ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.4.1. INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA', 'HSPA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
def fitGBM(trial):
  "Train GBM Model"
 
  loss_list = ['deviance','exponential']                                                       
  params = {'model_gbm__loss':trial.suggest_categorical('loss', loss_list),                     
            'model_gbm__n_estimators':trial.suggest_int('n_estimators', 50, 500, 25),
            'model_gbm__max_depth':trial.suggest_int('max_depth', 1, 3, 1), 
            'model_gbm__learning_rate':trial.suggest_loguniform('learning_rate',0.005, 0.5),   
            'model_gbm__subsample':trial.suggest_discrete_uniform('subsample', 0.5, 0.95, 0.05),
            'model_gbm__validation_fraction':trial.suggest_discrete_uniform('validation_fraction', 0.05, 0.5, 0.05)       
  }
   
  GBM = Pipeline(steps = [('preprocessor_gbm', preprocessor),('model_gbm', GradientBoostingClassifier(random_state = 123))])  
  GBM.set_params(**params)
 
  GBM.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = GBM.predict_proba(val_x)[:,1]
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
  # RODEI ESSE PROCESSO EM OUTRO NOTEBOOK PARALELO, TEMPO DE PROCESSAMENTO = 11.36 hours
#study = optuna.create_study(direction = 'maximize')
#study.optimize(fitGBM, n_trials = 50)  

# COMMAND ----------

#print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
GBM = Pipeline(steps = [('preprocessor_gbm', preprocessor),
                        ('model_gbm',GradientBoostingClassifier(random_state = 123, loss = 'exponential', n_estimators = 475, max_depth = 3, learning_rate = 0.21352725304039114, subsample = 0.95,validation_fraction = 0.4))]) 
GBM.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = GBM.predict_proba(dev_x)[:,1]
y_predict_val = GBM.predict_proba(val_x)[:,1]

proba_dev = GBM.predict_proba(dev_x)
proba_val = GBM.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 5.4.2. HSPA + INTERNAS

# COMMAND ----------

dev_x = base_dev.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
dev_y = base_dev.filter(['SConcEver60dP6_100'])

val_x = base_val.drop(columns=['SConcEver60dP6_100', 'SCORE_C2BA'])
val_y = base_val.filter(['SConcEver60dP6_100'])

# Pipeline de pre-processing para as variáveis
cat_columns = dev_x.select_dtypes(include=['object']).columns.tolist()
num_columns = dev_x.select_dtypes(include=['int','float']).columns.tolist() 

num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', WOEEncoder())])

preprocessor = ColumnTransformer( transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])

# COMMAND ----------

# Função de treino de parâmetros: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
def fitGBM(trial):
  "Train GBM Model"
 
  loss_list = ['deviance','exponential']                                                       
  params = {'model_gbm__loss':trial.suggest_categorical('loss', loss_list),                     
            'model_gbm__n_estimators':trial.suggest_int('n_estimators', 50, 500, 25),
            'model_gbm__max_depth':trial.suggest_int('max_depth', 1, 3, 1), 
            'model_gbm__learning_rate':trial.suggest_loguniform('learning_rate',0.005, 0.5),   
            'model_gbm__subsample':trial.suggest_discrete_uniform('subsample', 0.5, 0.95, 0.05),
            'model_gbm__validation_fraction':trial.suggest_discrete_uniform('validation_fraction', 0.05, 0.5, 0.05)       
  }
   
  GBM = Pipeline(steps = [('preprocessor_gbm', preprocessor),('model_gbm', GradientBoostingClassifier(random_state = 123))])  
  GBM.set_params(**params)
 
  GBM.fit(dev_x, dev_y['SConcEver60dP6_100'])
  y_predict_val = GBM.predict_proba(val_x)[:,1]
  gini = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1
  
  print(f'GINI INDEX: {gini}')
  return gini # roc_auc


# Fazendo a otimização
study = optuna.create_study(direction = 'maximize')
study.optimize(fitGBM, n_trials = 50)   

# COMMAND ----------

print(study.best_trial)

# COMMAND ----------

# Treinando com os parâmetros escolhidos
GBM = Pipeline(steps = [('preprocessor_gbm', preprocessor),
                        ('model_gbm',GradientBoostingClassifier(random_state = 123, loss = 'exponential', n_estimators = 450, max_depth = 3, learning_rate = 0.09897056270235999, subsample = 0.8500000000000001,validation_fraction = 0.1))]) 
GBM.fit(dev_x, dev_y['SConcEver60dP6_100'])

# COMMAND ----------

y_predict_dev = GBM.predict_proba(dev_x)[:,1]
y_predict_val = GBM.predict_proba(val_x)[:,1]

proba_dev = GBM.predict_proba(dev_x)
proba_val = GBM.predict_proba(val_x)

# COMMAND ----------

skplt.metrics.plot_ks_statistic(dev_y['SConcEver60dP6_100'], proba_dev)
display(plt.show())

# COMMAND ----------

skplt.metrics.plot_ks_statistic(val_y['SConcEver60dP6_100'], proba_val)
display(plt.show())

# COMMAND ----------

gini_dev = (2*roc_auc_score(dev_y['SConcEver60dP6_100'],y_predict_dev)) - 1
gini_val = (2*roc_auc_score(val_y['SConcEver60dP6_100'],y_predict_val)) - 1

gini_dev, gini_val
