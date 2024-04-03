# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###### Baseline para Monitoramento em Produção
# MAGIC 
# MAGIC ##### Equipe: DS Credit

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 01 . Bibliotecas ---------------------------------------------------------------------------

# COMMAND ----------

from pyspark.sql.functions import regexp_replace,min, max, month, split, year, concat,lower, substring, sqrt, lit, col, mean, stddev, when, count, sum, months_between, to_date, countDistinct, round, datediff, first, month, ceil, asc, year, isnan, date_format, abs, trunc, desc, row_number
import copy
from pyspark.sql.window import Window 
from datetime import datetime
from pytz import timezone
import pandas                as pd
import numpy                 as np
import datetime              as dt
import xgboost               as xgb
import matplotlib.pyplot     as plt
import seaborn               as sns
import mlflow
import mlflow.sklearn
import os
import sklearn
import warnings
import PythonShell
from pyspark.sql import SQLContext, SparkSession, Window, Row
from pyspark.sql import SparkSession 
from datetime import date, datetime, timedelta
!pip install scikit-plot
import scikitplot as skplt

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 02 . Funções ----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Ordenação - Gains Table

# COMMAND ----------

def gains_table(baseline=None, base_ref=None,target=None, prob=None):
    
    baseline.reset_index(inplace=True, drop=True)
    
    base_ref.reset_index(inplace=True, drop=True)
    
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
        if faixas_inicio[k] == 0 and j == 0:
          marcacao_decil.append(1)
        elif(j>faixas_inicio[k] and j<=faixas_fim[k]):
          marcacao_decil.append(k+1)

    print(len(base_ref), len(marcacao_decil))
    base_ref['marcacao_decil'] = marcacao_decil
    kstable = pd.crosstab(base_ref['marcacao_decil'], base_ref[target])
    
    ranges = []
    for i, j in enumerate(faixas_inicio):
      ranges.append('(' + str(j) + '-' + str(faixas_fim[i]) + ')')
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
    
    if 'marcacao_decil' in base_ref.columns:
      base_ref.drop(columns = ['marcacao_decil'], inplace = True)
    if 'target0' in base_ref.columns:
      base_ref.drop(columns = ['target0'], inplace = True) 
    if 'decil' in base_ref.columns:
      base_ref.drop(columns = ['decil'], inplace = True) 
    if 'marcacao_decil' in baseline.columns:
      baseline.drop(columns = ['marcacao_decil'], inplace = True)
    if 'target0' in baseline.columns:
      baseline.drop(columns = ['target0'], inplace = True) 
    if 'decil' in baseline.columns:
      baseline.drop(columns = ['decil'], inplace = True) 
      
    return(kstable)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Ordenação - Vars Númericas

# COMMAND ----------

def gains_table_num(baseline=None, base_ref=None,target=None, prob=None, safra = '', id_modelo = '', q=None, bins = None):
    
    baseline.reset_index(inplace=True, drop=True)
    
    base_ref.reset_index(inplace=True, drop=True)
    
    if q ==None:
      baseline['faixas'] = pd.cut(baseline[prob], bins)
      rer, bins = pd.cut(baseline[prob], bins, retbins=True, labels=False)
    else:
      baseline['faixas'] = pd.cut(baseline[prob], q, bins)
      rer, bins = pd.cut(baseline[prob], q, bins, retbins=True, labels=False)

    
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

    marcacao_faixas = []
    for i, j in enumerate(base_ref[prob]):
      for k, l in enumerate(faixas_inicio):
        if faixas_inicio[k] == 0 and j == 0:
          marcacao_faixas.append(1)
        elif(j>faixas_inicio[k] and j<=faixas_fim[k]):
          marcacao_faixas.append(k+1)

    base_ref['marcacao_faixas'] = marcacao_faixas
    
    if target!=None:
      kstable = pd.crosstab(base_ref['marcacao_faixas'], base_ref[target])
    else:
      kstable['Qtde_Total'] = base_ref['marcacao_faixas'].value_counts(dropna=False).sort_index()
      
      
    ranges = []
    for i, j in enumerate(faixas_inicio):
      ranges.append('(' + str(j) + '-' + str(faixas_fim[i]) + ')')
    
    if target!=None:
    
      kstable['Safra'] = [safra] * len(kstable)
      kstable['ID Modelo'] = [id_modelo] * len(kstable)
      kstable['Variável'] = [prob] * len(kstable)
      kstable['Ranges'] = ranges
      kstable['Qtde_Bom'] = kstable[0]
      kstable['Qtde_Mau'] = kstable[1]
      kstable['Qtde_Total'] = (kstable.Qtde_Bom + kstable.Qtde_Mau)
    
      base_ref['target0'] = 1 - base_ref[target]
      kstable['Perc_Bom']   = (kstable.Qtde_Bom / base_ref['target0'].sum()).apply('{0:.2%}'.format)
      kstable['Perc_Mau']   = (kstable.Qtde_Mau / base_ref[target].sum()).apply('{0:.2%}'.format)
      kstable['Perc_Total'] = ((kstable.Qtde_Bom + kstable.Qtde_Mau) / (base_ref['target0'].sum() + base_ref[target].sum())).apply('{0:.2%}'.format)
   
    else:
      
      kstable['Safra'] = [safra] * len(kstable)
      kstable['ID Modelo'] = [id_modelo] * len(kstable)
      kstable['Variável'] = [prob] * len(kstable)
      kstable['Ranges'] = ranges
      kstable['Qtde_Bom'] = 0
      kstable['Qtde_Mau'] = 0
    
      base_ref['target0'] = 0
      kstable['Perc_Bom']   = 0
      kstable['Perc_Mau']   = 0
      kstable['Perc_Total'] = kstable['Qtde_Total']/len(base_ref)     
    
    #Formatando
    if q != None:
      kstable.index = range(1,q+1)
    else:
      kstable.index = range(1,len(bins))
    kstable.index.rename('faixas', inplace=True)
    pd.set_option('display.max_columns', 12)
    
    if 'marcacao_faixas' in base_ref.columns:
      base_ref.drop(columns = ['marcacao_faixas'], inplace = True)
    if 'target0' in base_ref.columns:
      base_ref.drop(columns = ['target0'], inplace = True) 
    if 'faixas' in base_ref.columns:
      base_ref.drop(columns = ['faixas'], inplace = True) 
    if 'marcacao_faixas' in baseline.columns:
      baseline.drop(columns = ['marcacao_faixas'], inplace = True)
    if 'target0' in baseline.columns:
      baseline.drop(columns = ['target0'], inplace = True) 
    if 'faixas' in baseline.columns:
      baseline.drop(columns = ['faixas'], inplace = True) 
      
    return(kstable)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. Gráficos de Volume

# COMMAND ----------

def volume(df, col_groupby, color):
  cat = df.select_dtypes(['object', 'category']).columns.tolist()
  for i in cat:
    ax= pd.crosstab(df[col_groupby], df[i]).apply(lambda r: r/r.sum()*100, axis=1)
    ax_1 = ax.plot.bar(figsize=(16,8), stacked=True, rot=0, colormap = color)

    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlabel(i)
    plt.ylabel('Percent Distribution')
    
    display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 03. Base escorada ------------------------------------------------------

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")

#Base 
# 1.527.592
tabela_full = spark.read.parquet('s3://picpay-datalake-sandbox/jeobara.zacheski/score_pf_cs/base_full_escorada_cs_carteira/')

#Filtros produção
tabela1 = tabela_full.withColumn('exclusao_producao',when(col('SCORE_C2BA') <= 360, '01. C2BA <= 360')\
                                          .otherwise(when((col('idade_em_anos') < 18) | (col('idade_em_anos') > 90) , '02. Idade < 18 ou > 90')))

baseline = tabela1.filter(col('exclusao_producao').isNull())
# 731.219
baseline = baseline.withColumn('safra',when(col('ref_portfolio') == '2020-04', 'Out-of-Time')\
                                          .otherwise(when(col('ref_portfolio') != '2020-04' , 'Baseline')))

# COMMAND ----------

tabela_full.count()

# COMMAND ----------

baseline.count()

# COMMAND ----------

## Segmentação do grupo que será modelado
tmp = baseline.toPandas() 

# Separação em Desenvolvimento (100%) e Out-of-Time
base_tot = tmp.where(tmp['ref_portfolio'] != '2020-04')
base_tot = base_tot.dropna(how='all') 

base_oot = tmp.where(tmp['ref_portfolio'] == '2020-04')
base_oot = base_oot.dropna(how='all') 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Faixas Picpay Score

# COMMAND ----------

pp_gains_tot = gains_table(baseline=base_tot,base_ref=base_tot, target="SConcEver60dP6_100", prob="score_cs_carteira")
pp_gains_oot = gains_table(baseline=base_tot,base_ref=base_oot, target="SConcEver60dP6_100", prob="score_cs_carteira")

# COMMAND ----------

pp_gains_tot

# COMMAND ----------

pp_gains_oot

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Faixas HSPA

# COMMAND ----------

hspa_tot = gains_table(baseline=base_tot,base_ref=base_tot, target="SConcEver60dP6_100", prob="HSPA")
hspa_oot = gains_table(baseline=base_tot,base_ref=base_oot, target="SConcEver60dP6_100", prob="HSPA")

# COMMAND ----------

hspa_tot

# COMMAND ----------

hspa_oot

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Faixas Idade

# COMMAND ----------

idade_tot = gains_table(baseline=base_tot,base_ref=base_tot, target="SConcEver60dP6_100", prob="idade_em_anos")
idade_oot = gains_table(baseline=base_tot,base_ref=base_oot, target="SConcEver60dP6_100", prob="idade_em_anos")

# COMMAND ----------

idade_tot

# COMMAND ----------

idade_oot

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. Faixas tempo_registro_em_meses

# COMMAND ----------

tempo_tot = gains_table(baseline=base_tot,base_ref=base_tot, target="SConcEver60dP6_100", prob="tempo_registro_em_meses")
tempo_oot = gains_table(baseline=base_tot,base_ref=base_oot, target="SConcEver60dP6_100", prob="tempo_registro_em_meses")

# COMMAND ----------

tempo_tot

# COMMAND ----------

tempo_oot

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5. Faixas days_since_last_transaction

# COMMAND ----------

base_tot['days_since_last_transaction'].describe()

# COMMAND ----------

base_oot['days_since_last_transaction'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6. Faixas am_max_pay_12m_trx

# COMMAND ----------

base_tot['am_max_pay_12m_trx'].describe()

# COMMAND ----------

base_oot['am_max_pay_12m_trx'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7. Faixas mental_accounting_score

# COMMAND ----------

base_tot['mental_accounting_score'].describe()

# COMMAND ----------

base_oot['mental_accounting_score'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8. Faixas peer_effect_score

# COMMAND ----------

base_tot['peer_effect_score'].describe()

# COMMAND ----------

base_oot['peer_effect_score'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.9. Faixas am_max_recp2p_12m_trx

# COMMAND ----------

base_tot['am_max_recp2p_12m_trx'].describe()

# COMMAND ----------

base_oot['am_max_recp2p_12m_trx'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.10. Faixas qtde_emissores_distintos

# COMMAND ----------

base_tot['qtde_emissores_distintos'].describe()

# COMMAND ----------

base_oot['qtde_emissores_distintos'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.11. Faixas am_max_payp2pbal_12m_trx

# COMMAND ----------

base_tot['am_max_payp2pbal_12m_trx'].describe()

# COMMAND ----------

base_oot['am_max_payp2pbal_12m_trx'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 04. Marcação das classes na bade DEV ----------------------------------------------

# COMMAND ----------

#======================================================================================================================================================================
# Score Picpay
variaveis = baseline.withColumn('classe_PPScore',when((col('score_cs_carteira') <= 280), '01. 0 - 280')\
                            .otherwise(when((col('score_cs_carteira') > 280) & (col('score_cs_carteira') <= 413), '02. 281 - 413')\
                                .otherwise(when((col('score_cs_carteira') > 413) & (col('score_cs_carteira') <= 501), '03. 414 - 501')\
                                    .otherwise(when((col('score_cs_carteira') > 501) & (col('score_cs_carteira') <= 589), '04. 502 - 589')\
                                        .otherwise(when((col('score_cs_carteira') > 589) & (col('score_cs_carteira') <= 674), '05. 590 - 674')\
                                            .otherwise(when((col('score_cs_carteira') > 674) & (col('score_cs_carteira') <= 741), '06. 675 - 741')\
                                                .otherwise(when((col('score_cs_carteira') > 741) & (col('score_cs_carteira') <= 794), '07. 742 - 794')\
                                                    .otherwise(when((col('score_cs_carteira') > 794) & (col('score_cs_carteira') <= 841), '08. 795 - 841')\
                                                        .otherwise(when((col('score_cs_carteira') > 841) & (col('score_cs_carteira') <= 895), '09. 842 - 895')\
                                                            .otherwise(when((col('score_cs_carteira') > 895) & (col('score_cs_carteira') <= 1000), '10. 896 - 1000')\
                                                                .otherwise('99. Outros')))))))))))

#======================================================================================================================================================================
# Score HSPA
variaveis = variaveis.withColumn('classe_HSPA',when((col('HSPA') <= 431), '01. 0 - 431')\
                            .otherwise(when((col('HSPA') > 431) & (col('HSPA') <= 517), '02. 432 - 517')\
                                .otherwise(when((col('HSPA') > 517) & (col('HSPA') <= 587), '03. 518 - 587')\
                                    .otherwise(when((col('HSPA') > 587) & (col('HSPA') <= 653), '04. 588 - 653')\
                                        .otherwise(when((col('HSPA') > 653) & (col('HSPA') <= 710), '05. 654 - 710')\
                                            .otherwise(when((col('HSPA') > 710) & (col('HSPA') <= 765), '06. 711 - 765')\
                                                .otherwise(when((col('HSPA') > 765) & (col('HSPA') <= 829), '07. 766 - 829')\
                                                    .otherwise(when((col('HSPA') > 829) & (col('HSPA') <= 898), '08. 830 - 898')\
                                                        .otherwise(when((col('HSPA') > 898) & (col('HSPA') <= 940), '09. 899 - 940')\
                                                            .otherwise(when((col('HSPA') > 940) & (col('HSPA') <= 1000), '10. 941 - 1000')\
                                                                .otherwise('99. Outros')))))))))))
 
#======================================================================================================================================================================
# idade_em_anos
variaveis = variaveis.withColumn('classe_idade',when((col('idade_em_anos') < 18) | (col('idade_em_anos') > 90), '99. Idade < 18 ou > 90')\
                                  .otherwise(when((col('idade_em_anos') >= 18) & (col('idade_em_anos') <= 20), '01. 18 - 20')\
                                      .otherwise(when((col('idade_em_anos') > 20) & (col('idade_em_anos') <= 25), '02. 21 - 25')\
                                          .otherwise(when((col('idade_em_anos') > 25) & (col('idade_em_anos') <= 30), '03. 26 - 30')\
                                              .otherwise(when((col('idade_em_anos') > 30) & (col('idade_em_anos') <= 35), '04. 31 - 35')\
                                                  .otherwise(when((col('idade_em_anos') > 35) & (col('idade_em_anos') <= 40), '05. 36 - 40')\
                                                      .otherwise(when((col('idade_em_anos') > 40) & (col('idade_em_anos') <= 50), '06. 41 - 50')\
                                                          .otherwise(when((col('idade_em_anos') > 50) & (col('idade_em_anos') <= 90), '07. 51 - 90')\
                                                              .otherwise('999. Outros')))))))))

#======================================================================================================================================================================
# ddd_state
variaveis = variaveis.withColumn('classe_ddd_state',when(col('ddd_state').isin('RO','AC','AM','RR'),'Noroeste')\
                                 .otherwise(when(col('ddd_state').isin('PA','AP','TO','MA'),'Norte')\
                                     .otherwise(when(col('ddd_state').isin('PI','CE','RN','PB','PE','AL','SE','BA'),'Nordeste')\
                                         .otherwise(when(col('ddd_state').isin('MG','ES','RJ'),'Sudeste -SP')\
                                             .otherwise(when(col('ddd_state').isin('PR','SC','RS'),'Sul')\
                                                 .otherwise(when(col('ddd_state').isin('MS','MT','GO','DF'),'Centro-Oeste')\
                                                     .otherwise(col('ddd_state'))))))))

 

#====================================================================================================================================================================== 
#tempo_registro_em_meses
variaveis = variaveis.withColumn('classe_tempo_relacionamento',when((col('tempo_registro_em_meses').isNull()), '01. Relacionamento <= 1 mês')\
                           .otherwise(when((col('tempo_registro_em_meses') >= 0) & (col('tempo_registro_em_meses') <= 1), '01. Relacionamento <= 1 mês')\
                               .otherwise(when((col('tempo_registro_em_meses') > 1) & (col('tempo_registro_em_meses') <= 3), '02. 1 < Relacionamento <= 3 meses')\
                                   .otherwise(when((col('tempo_registro_em_meses') > 3) & (col('tempo_registro_em_meses') <= 6), '03. 3 < Relacionamento <= 6 meses')\
                                       .otherwise(when((col('tempo_registro_em_meses') > 6) & (col('tempo_registro_em_meses') <= 12), '04. 6 < Relacionamento <= 12 meses')\
                                           .otherwise(when((col('tempo_registro_em_meses') > 12) & (col('tempo_registro_em_meses') <= 24), '05. 12 < Relacionamento <= 24 meses')\
                                               .otherwise(when((col('tempo_registro_em_meses') > 24) & (col('tempo_registro_em_meses') <= 36), '06. 24 < Relacionamento <= 36 meses')\
                                                  .otherwise(when((col('tempo_registro_em_meses') > 36), '07. Relacionamento > 36 meses')\
                                                    .otherwise('99. Outros')))))))))

#====================================================================================================================================================================== 
#days_since_last_transaction
variaveis = variaveis.withColumn('classe_last_transaction',when((col('days_since_last_transaction').isNull()), '01. 0 dias')\
                          .otherwise(when((col('days_since_last_transaction') == 0), '01. 0 dias')\
                            .otherwise(when((col('days_since_last_transaction') > 0) & (col('days_since_last_transaction') <= 10), '02. 1 - 10 dias')\
                                .otherwise(when((col('days_since_last_transaction') > 10) & (col('days_since_last_transaction') <= 30), '03. 11 - 30 dias')\
                                  .otherwise(when((col('days_since_last_transaction') > 30) & (col('days_since_last_transaction') <= 60), '04. 31 - 60 dias')\
                                    .otherwise(when((col('days_since_last_transaction') > 60) & (col('days_since_last_transaction') <= 180), '05. 61 - 180 dias')\
                                        .otherwise(when((col('days_since_last_transaction') > 180) & (col('days_since_last_transaction') <= 360), '06. 181 - 360 dias')\
                                          .otherwise(when((col('days_since_last_transaction') > 360) & (col('days_since_last_transaction') <= 720), '07. 361 - 720 dias')\
                                            .otherwise(when((col('days_since_last_transaction') > 720), '08. >= 721 dias')\
                                              .otherwise('99. Outros'))))))))))

#======================================================================================================================================================================
#agrupamento_instituicao_cartao_ref
variaveis = variaveis.withColumn('classe_instituicao_cartao',when((col('agrupamento_instituicao_cartao_ref') == 'caixa economica federal'), 'caixa')\
                                     .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'bradesco'), 'bradesco')\
                                         .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'nu pagamentos sa'), 'nubank')\
                                             .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'itau'), 'itau')\
                                                 .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'santander'), 'santander')\
                                                     .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'banco do brasil'), 'banco do brasil')\
                                                         .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'banco original'), 'banco original')\
                                                             .otherwise(when(col('agrupamento_instituicao_cartao_ref').isin('vazio','Outros') | col('agrupamento_instituicao_cartao_ref').isNull(),'vazio')\
                                                                 .otherwise('outras ifs')))))))))
           
#======================================================================================================================================================================
#am_max_pay_12m_trx
variaveis = variaveis.withColumn('classe_am_max_pay_12m_trx',when((col('am_max_pay_12m_trx').isNull()), '01. R$ 0')\
                                 .otherwise(when((col('am_max_pay_12m_trx') == 0), '01. R$ 0')\
                                     .otherwise(when((col('am_max_pay_12m_trx') > 0) & (col('am_max_pay_12m_trx') <= 10), '02. R$ 0 - 10')\
                                         .otherwise(when((col('am_max_pay_12m_trx') > 10) & (col('am_max_pay_12m_trx') <= 50), '03. R$ 10 - 50')\
                                             .otherwise(when((col('am_max_pay_12m_trx') > 50) & (col('am_max_pay_12m_trx') <= 100), '04. R$ 50 - 100')\
                                                 .otherwise(when((col('am_max_pay_12m_trx') > 100), '05. > R$ 100')\
                                                         .otherwise('99. Outros')))))))


#======================================================================================================================================================================
#modelo_celular_ref
variaveis = variaveis.withColumn('classe_modelo_celular',when((col('modelo_celular_ref') == 'samsung'), 'samsung')\
                                     .otherwise(when((col('modelo_celular_ref') == 'motorola'), 'motorola')\
                                         .otherwise(when((col('modelo_celular_ref') == 'iphone'), 'iphone')\
                                             .otherwise(when((col('modelo_celular_ref') == 'xiaomi'), 'xiaomi')\
                                                 .otherwise(when((col('modelo_celular_ref') == 'lg'), 'lg')\
                                                     .otherwise('Outros'))))))
                                           #======================================================================================================================================================================
#mental_accounting_score
variaveis = variaveis.withColumn('classe_mental_accounting_score',when((col('mental_accounting_score').isNull()), '01. 0')\
                                 .otherwise(when((col('mental_accounting_score') == 0), '01. 0')\
                                     .otherwise(when((col('mental_accounting_score') > 0) & (col('mental_accounting_score') <= 10), '02. 0 - 10')\
                                         .otherwise(when((col('mental_accounting_score') > 10) & (col('mental_accounting_score') <= 50), '03. 10 - 50')\
                                             .otherwise(when((col('mental_accounting_score') > 50), '04. > 50')\
                                                         .otherwise('99. Outros'))))))
                                
#======================================================================================================================================================================
#peer_effect_score
variaveis = variaveis.withColumn('classe_peer_effect_score',when((col('peer_effect_score').isNull()), '01. 0')\
                                 .otherwise(when((col('peer_effect_score') == 0), '01. 0')\
                                     .otherwise(when((col('peer_effect_score') > 0) & (col('peer_effect_score') <= 20), '02. 0 - 20')\
                                         .otherwise(when((col('peer_effect_score') > 20) & (col('peer_effect_score') <= 50), '03. 20 - 50')\
                                             .otherwise(when((col('peer_effect_score') > 50), '04. > 50')\
                                                         .otherwise('99. Outros'))))))


#======================================================================================================================================================================
#email_ajuste
variaveis = variaveis.withColumn('classe_email_ajuste',when((col('email_ajuste') == 'gmail'), 'gmail')\
                                     .otherwise(when((col('email_ajuste') == 'hotmail'), 'hotmail')\
                                         .otherwise(when((col('email_ajuste') == 'outlook'), 'outlook')\
                                             .otherwise(when((col('email_ajuste') == 'yahoo'), 'yahoo')\
                                                 .otherwise('Outros')))))

#======================================================================================================================================================================               #am_max_recp2p_12m_trx
variaveis = variaveis.withColumn('classe_am_max_recp2p_12m_trx',when((col('am_max_recp2p_12m_trx').isNull()), '01. R$ 0')\
                                 .otherwise(when((col('am_max_recp2p_12m_trx') == 0), '01. R$ 0')\
                                     .otherwise(when((col('am_max_recp2p_12m_trx') > 0) & (col('am_max_recp2p_12m_trx') <= 10), '02. R$ 0 - 10')\
                                         .otherwise(when((col('am_max_recp2p_12m_trx') > 10) & (col('am_max_recp2p_12m_trx') <= 50), '03. R$ 10 - 50')\
                                             .otherwise(when((col('am_max_recp2p_12m_trx') > 50) & (col('am_max_recp2p_12m_trx') <= 100), '04. R$ 50 - 100')\
                                                 .otherwise(when((col('am_max_recp2p_12m_trx') > 100), '05. > R$ 100')\
                                                         .otherwise('99. Outros')))))))

#======================================================================================================================================================================
#qtde_emissores_distintos
variaveis = variaveis.withColumn('classe_emissores_distintos',when((col('qtde_emissores_distintos').isNull()), '01. Qtde = 0')\
                                     .otherwise(when((col('qtde_emissores_distintos') == 0), '01. Qtde = 0')\
                                         .otherwise(when((col('qtde_emissores_distintos') == 1), '02. Qtde = 1')\
                                             .otherwise(when((col('qtde_emissores_distintos') > 1), '03. Qtde >= 2')\
                                                 .otherwise('99. Outros')))))

#======================================================================================================================================================================
#am_max_payp2pbal_12m_trx
variaveis = variaveis.withColumn('classe_am_max_payp2pbal_12m_trx',when((col('am_max_payp2pbal_12m_trx').isNull()), '01. R$ 0')\
                                 .otherwise(when((col('am_max_payp2pbal_12m_trx') == 0), '01. R$ 0')\
                                     .otherwise(when((col('am_max_payp2pbal_12m_trx') > 0) & (col('am_max_payp2pbal_12m_trx') <= 10), '02. R$ 0 - 10')\
                                         .otherwise(when((col('am_max_payp2pbal_12m_trx') > 10) & (col('am_max_payp2pbal_12m_trx') <= 50), '03. R$ 10 - 50')\
                                             .otherwise(when((col('am_max_payp2pbal_12m_trx') > 50) & (col('am_max_payp2pbal_12m_trx') <= 100), '04. R$ 50 - 100')\
                                                 .otherwise(when((col('am_max_payp2pbal_12m_trx') > 100), '05. > R$ 100')\
                                                         .otherwise('99. Outros')))))))

# COMMAND ----------

classes = ['classe_PPScore',\
              'classe_HSPA',\
              'classe_idade',\
              'classe_ddd_state',\
              'classe_tempo_relacionamento',\
              'classe_last_transaction',\
              'classe_instituicao_cartao',\
              'classe_am_max_pay_12m_trx',\
              'classe_modelo_celular',\
              'classe_mental_accounting_score',\
              'classe_peer_effect_score',\
              'classe_email_ajuste',\
              'classe_am_max_recp2p_12m_trx',\
              'classe_emissores_distintos',\
              'digital_account_status',\
              'classe_am_max_payp2pbal_12m_trx']

classes.append('safra')
analise_classes_dev = variaveis.toPandas().filter(items=classes)

# COMMAND ----------

cubo1 = variaveis.groupBy('safra',\
                          'classe_am_max_payp2pbal_12m_trx').count().orderBy(desc('count'))
display(cubo1)

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 05. Marcação das classes em produção ----------------------------------------------

# COMMAND ----------

# Variáveis
vars_pf_cs_consumer_m_v1       = spark.read.parquet('s3://picpay-datalake-projects/temp_models/vars_pf_cs_consumer_m_v1')       # CS CONSUMER

# Score
score_pf_cs_consumer_m_v1 = spark.table('models.score_pf_cs_consumer_m_v1')     # CS CONSUMER
score_pf_cs_consumer_m_v1 = score_pf_cs_consumer_m_v1.select('cpf','consumer_id','referenced_at','score_pf_cs_carteira','score_pf_cs_carteira_final')

# Cruzando os scores das logs apartadas com as bases mães
cs_carteira = vars_pf_cs_consumer_m_v1.join(score_pf_cs_consumer_m_v1,on = ['cpf', 'consumer_id', 'referenced_at'], how = 'left') # CS CONSUMER

# Adicionando uma coluna extra de referencia
cs_carteira = cs_carteira.withColumn('safra', substring('referenced_at',1,7))

# COMMAND ----------

score_pf_cs = cs_carteira.filter(col('score_pf_cs_carteira_final') > 1)
score_pf_cs = score_pf_cs.filter(col('referenced_at') >= '2021-01-01')

#======================================================================================================================================================================
# Score Picpay
vars_prod = score_pf_cs.withColumn('classe_PPScore',when((col('score_pf_cs_carteira_final') <= 280), '01. 0 - 280')\
                           .otherwise(when((col('score_pf_cs_carteira_final') > 280) & (col('score_pf_cs_carteira_final') <= 413), '02. 281 - 413')\
                               .otherwise(when((col('score_pf_cs_carteira_final') > 413) & (col('score_pf_cs_carteira_final') <= 501), '03. 414 - 501')\
                                   .otherwise(when((col('score_pf_cs_carteira_final') > 501) & (col('score_pf_cs_carteira_final') <= 589), '04. 502 - 589')\
                                       .otherwise(when((col('score_pf_cs_carteira_final') > 589) & (col('score_pf_cs_carteira_final') <= 674), '05. 590 - 674')\
                                           .otherwise(when((col('score_pf_cs_carteira_final') > 674) & (col('score_pf_cs_carteira_final') <= 741), '06. 675 - 741')\
                                               .otherwise(when((col('score_pf_cs_carteira_final') > 741) & (col('score_pf_cs_carteira_final') <= 794), '07. 742 - 794')\
                                                   .otherwise(when((col('score_pf_cs_carteira_final') > 794) & (col('score_pf_cs_carteira_final') <= 841), '08. 795 - 841')\
                                                       .otherwise(when((col('score_pf_cs_carteira_final') > 841) & (col('score_pf_cs_carteira_final') <= 895), '09. 842 - 895')\
                                                           .otherwise(when((col('score_pf_cs_carteira_final') > 895) & (col('score_pf_cs_carteira_final') <= 1000), '10. 896 - 1000')\
                                                                .otherwise('99. Outros')))))))))))

#======================================================================================================================================================================
# Score HSPA
vars_prod = vars_prod.withColumn('classe_HSPA',when((col('HSPA') <= 431), '01. 0 - 431')\
                            .otherwise(when((col('HSPA') > 431) & (col('HSPA') <= 517), '02. 432 - 517')\
                                .otherwise(when((col('HSPA') > 517) & (col('HSPA') <= 587), '03. 518 - 587')\
                                    .otherwise(when((col('HSPA') > 587) & (col('HSPA') <= 653), '04. 588 - 653')\
                                        .otherwise(when((col('HSPA') > 653) & (col('HSPA') <= 710), '05. 654 - 710')\
                                            .otherwise(when((col('HSPA') > 710) & (col('HSPA') <= 765), '06. 711 - 765')\
                                                .otherwise(when((col('HSPA') > 765) & (col('HSPA') <= 829), '07. 766 - 829')\
                                                    .otherwise(when((col('HSPA') > 829) & (col('HSPA') <= 898), '08. 830 - 898')\
                                                        .otherwise(when((col('HSPA') > 898) & (col('HSPA') <= 940), '09. 899 - 940')\
                                                            .otherwise(when((col('HSPA') > 940) & (col('HSPA') <= 1000), '10. 941 - 1000')\
                                                                .otherwise('99. Outros')))))))))))
 
#======================================================================================================================================================================
# idade_em_anos
vars_prod = vars_prod.withColumn('classe_idade',when((col('idade_em_anos') < 18) | (col('idade_em_anos') > 90), '99. Idade < 18 ou > 90')\
                                  .otherwise(when((col('idade_em_anos') >= 18) & (col('idade_em_anos') <= 20), '01. 18 - 20')\
                                      .otherwise(when((col('idade_em_anos') > 20) & (col('idade_em_anos') <= 25), '02. 21 - 25')\
                                          .otherwise(when((col('idade_em_anos') > 25) & (col('idade_em_anos') <= 30), '03. 26 - 30')\
                                              .otherwise(when((col('idade_em_anos') > 30) & (col('idade_em_anos') <= 35), '04. 31 - 35')\
                                                  .otherwise(when((col('idade_em_anos') > 35) & (col('idade_em_anos') <= 40), '05. 36 - 40')\
                                                      .otherwise(when((col('idade_em_anos') > 40) & (col('idade_em_anos') <= 50), '06. 41 - 50')\
                                                          .otherwise(when((col('idade_em_anos') > 50) & (col('idade_em_anos') <= 90), '07. 51 - 90')\
                                                              .otherwise('999. Outros')))))))))

#======================================================================================================================================================================
# ddd_state
vars_prod = vars_prod.withColumn('classe_ddd_state',when((col('ddd_state').isNull()), 'others')\
                                 .otherwise(when(col('ddd_state').isin('RO','AC','AM','RR'),'Noroeste')\
                                 .otherwise(when(col('ddd_state').isin('PA','AP','TO','MA'),'Norte')\
                                     .otherwise(when(col('ddd_state').isin('PI','CE','RN','PB','PE','AL','SE','BA'),'Nordeste')\
                                         .otherwise(when(col('ddd_state').isin('MG','ES','RJ'),'Sudeste -SP')\
                                             .otherwise(when(col('ddd_state').isin('PR','SC','RS'),'Sul')\
                                                 .otherwise(when(col('ddd_state').isin('MS','MT','GO','DF'),'Centro-Oeste')\
                                                     .otherwise(col('ddd_state')))))))))

#====================================================================================================================================================================== 
#tempo_registro_em_meses
vars_prod = vars_prod.withColumn('classe_tempo_relacionamento',when((col('tempo_registro_em_meses').isNull()), '01. Relacionamento <= 1 mês')\
                           .otherwise(when((col('tempo_registro_em_meses') >= 0) & (col('tempo_registro_em_meses') <= 1), '01. Relacionamento <= 1 mês')\
                               .otherwise(when((col('tempo_registro_em_meses') > 1) & (col('tempo_registro_em_meses') <= 3), '02. 1 < Relacionamento <= 3 meses')\
                                   .otherwise(when((col('tempo_registro_em_meses') > 3) & (col('tempo_registro_em_meses') <= 6), '03. 3 < Relacionamento <= 6 meses')\
                                       .otherwise(when((col('tempo_registro_em_meses') > 6) & (col('tempo_registro_em_meses') <= 12), '04. 6 < Relacionamento <= 12 meses')\
                                           .otherwise(when((col('tempo_registro_em_meses') > 12) & (col('tempo_registro_em_meses') <= 24), '05. 12 < Relacionamento <= 24 meses')\
                                               .otherwise(when((col('tempo_registro_em_meses') > 24) & (col('tempo_registro_em_meses') <= 36), '06. 24 < Relacionamento <= 36 meses')\
                                                  .otherwise(when((col('tempo_registro_em_meses') > 36), '07. Relacionamento > 36 meses')\
                                                    .otherwise('99. Outros')))))))))

#====================================================================================================================================================================== 
#days_since_last_transaction
vars_prod = vars_prod.withColumn('classe_last_transaction',when((col('days_since_last_transaction').isNull()), '01. 0 dias')\
                          .otherwise(when((col('days_since_last_transaction') == 0), '01. 0 dias')\
                            .otherwise(when((col('days_since_last_transaction') > 0) & (col('days_since_last_transaction') <= 10), '02. 1 - 10 dias')\
                                .otherwise(when((col('days_since_last_transaction') > 10) & (col('days_since_last_transaction') <= 30), '03. 11 - 30 dias')\
                                  .otherwise(when((col('days_since_last_transaction') > 30) & (col('days_since_last_transaction') <= 60), '04. 31 - 60 dias')\
                                    .otherwise(when((col('days_since_last_transaction') > 60) & (col('days_since_last_transaction') <= 180), '05. 61 - 180 dias')\
                                        .otherwise(when((col('days_since_last_transaction') > 180) & (col('days_since_last_transaction') <= 360), '06. 181 - 360 dias')\
                                          .otherwise(when((col('days_since_last_transaction') > 360) & (col('days_since_last_transaction') <= 720), '07. 361 - 720 dias')\
                                            .otherwise(when((col('days_since_last_transaction') > 720), '08. >= 721 dias')\
                                              .otherwise('99. Outros'))))))))))
             
  
#======================================================================================================================================================================
#agrupamento_instituicao_cartao_ref
vars_prod = vars_prod.withColumn('classe_instituicao_cartao',when((col('agrupamento_instituicao_cartao_ref') == 'caixa economica federal'), 'caixa')\
                                     .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'bradesco'), 'bradesco')\
                                         .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'nu pagamentos sa'), 'nubank')\
                                             .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'itau'), 'itau')\
                                                 .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'santander'), 'santander')\
                                                     .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'banco do brasil'), 'banco do brasil')\
                                                         .otherwise(when((col('agrupamento_instituicao_cartao_ref') == 'banco original'), 'banco original')\
                                                             .otherwise(when(col('agrupamento_instituicao_cartao_ref').isin('vazio','Outros') | col('agrupamento_instituicao_cartao_ref').isNull(),'vazio')\
                                                                 .otherwise('outras ifs')))))))))


#======================================================================================================================================================================
#am_max_pay_12m_trx
vars_prod = vars_prod.withColumn('classe_am_max_pay_12m_trx',when((col('am_max_pay_12m_trx').isNull()), '01. R$ 0')\
                                 .otherwise(when((col('am_max_pay_12m_trx') == 0), '01. R$ 0')\
                                     .otherwise(when((col('am_max_pay_12m_trx') > 0) & (col('am_max_pay_12m_trx') <= 10), '02. R$ 0 - 10')\
                                         .otherwise(when((col('am_max_pay_12m_trx') > 10) & (col('am_max_pay_12m_trx') <= 50), '03. R$ 10 - 50')\
                                             .otherwise(when((col('am_max_pay_12m_trx') > 50) & (col('am_max_pay_12m_trx') <= 100), '04. R$ 50 - 100')\
                                                 .otherwise(when((col('am_max_pay_12m_trx') > 100), '05. > R$ 100')\
                                                         .otherwise('99. Outros')))))))

#======================================================================================================================================================================
#modelo_celular_ref
vars_prod = vars_prod.withColumn('classe_modelo_celular',when((col('modelo_celular_ref') == 'samsung'), 'samsung')\
                                     .otherwise(when((col('modelo_celular_ref') == 'motorola'), 'motorola')\
                                         .otherwise(when((col('modelo_celular_ref') == 'iphone'), 'iphone')\
                                             .otherwise(when((col('modelo_celular_ref') == 'xiaomi'), 'xiaomi')\
                                                 .otherwise(when((col('modelo_celular_ref') == 'lg'), 'lg')\
                                                     .otherwise('Outros'))))))


#======================================================================================================================================================================
#mental_accounting_score
vars_prod = vars_prod.withColumn('classe_mental_accounting_score',when((col('mental_accounting_score').isNull()), '01. 0')\
                                 .otherwise(when((col('mental_accounting_score') == 0), '01. 0')\
                                     .otherwise(when((col('mental_accounting_score') > 0) & (col('mental_accounting_score') <= 10), '02. 0 - 10')\
                                         .otherwise(when((col('mental_accounting_score') > 10) & (col('mental_accounting_score') <= 50), '03. 10 - 50')\
                                             .otherwise(when((col('mental_accounting_score') > 50), '04. > 50')\
                                                         .otherwise('99. Outros'))))))
                                
#======================================================================================================================================================================
#peer_effect_score
vars_prod = vars_prod.withColumn('classe_peer_effect_score',when((col('peer_effect_score').isNull()), '01. 0')\
                                 .otherwise(when((col('peer_effect_score') == 0), '01. 0')\
                                     .otherwise(when((col('peer_effect_score') > 0) & (col('peer_effect_score') <= 20), '02. 0 - 20')\
                                         .otherwise(when((col('peer_effect_score') > 20) & (col('peer_effect_score') <= 50), '03. 20 - 50')\
                                             .otherwise(when((col('peer_effect_score') > 50), '04. > 50')\
                                                         .otherwise('99. Outros'))))))

#======================================================================================================================================================================
#email_ajuste
vars_prod = vars_prod.withColumn('classe_email_ajuste',when((col('email_ajuste') == 'gmail'), 'gmail')\
                                     .otherwise(when((col('email_ajuste') == 'hotmail'), 'hotmail')\
                                         .otherwise(when((col('email_ajuste') == 'outlook'), 'outlook')\
                                             .otherwise(when((col('email_ajuste') == 'yahoo'), 'yahoo')\
                                                 .otherwise('Outros')))))

#======================================================================================================================================================================               #am_max_recp2p_12m_trx
vars_prod = vars_prod.withColumn('classe_am_max_recp2p_12m_trx',when((col('am_max_recp2p_12m_trx').isNull()), '01. R$ 0')\
                                 .otherwise(when((col('am_max_recp2p_12m_trx') == 0), '01. R$ 0')\
                                     .otherwise(when((col('am_max_recp2p_12m_trx') > 0) & (col('am_max_recp2p_12m_trx') <= 10), '02. R$ 0 - 10')\
                                         .otherwise(when((col('am_max_recp2p_12m_trx') > 10) & (col('am_max_recp2p_12m_trx') <= 50), '03. R$ 10 - 50')\
                                             .otherwise(when((col('am_max_recp2p_12m_trx') > 50) & (col('am_max_recp2p_12m_trx') <= 100), '04. R$ 50 - 100')\
                                                 .otherwise(when((col('am_max_recp2p_12m_trx') > 100), '05. > R$ 100')\
                                                         .otherwise('99. Outros')))))))

#======================================================================================================================================================================
#qtde_emissores_distintos
vars_prod = vars_prod.withColumn('classe_emissores_distintos',when((col('qtde_emissores_distintos').isNull()), '01. Qtde = 0')\
                                     .otherwise(when((col('qtde_emissores_distintos') == 0), '01. Qtde = 0')\
                                         .otherwise(when((col('qtde_emissores_distintos') == 1), '02. Qtde = 1')\
                                             .otherwise(when((col('qtde_emissores_distintos') > 1), '03. Qtde >= 2')\
                                                 .otherwise('99. Outros')))))

#======================================================================================================================================================================
#am_max_payp2pbal_12m_trx
vars_prod = vars_prod.withColumn('classe_am_max_payp2pbal_12m_trx',when((col('am_max_payp2pbal_12m_trx').isNull()), '01. R$ 0')\
                                 .otherwise(when((col('am_max_payp2pbal_12m_trx') == 0), '01. R$ 0')\
                                     .otherwise(when((col('am_max_payp2pbal_12m_trx') > 0) & (col('am_max_payp2pbal_12m_trx') <= 10), '02. R$ 0 - 10')\
                                         .otherwise(when((col('am_max_payp2pbal_12m_trx') > 10) & (col('am_max_payp2pbal_12m_trx') <= 50), '03. R$ 10 - 50')\
                                             .otherwise(when((col('am_max_payp2pbal_12m_trx') > 50) & (col('am_max_payp2pbal_12m_trx') <= 100), '04. R$ 50 - 100')\
                                                 .otherwise(when((col('am_max_payp2pbal_12m_trx') > 100), '05. > R$ 100')\
                                                         .otherwise('99. Outros')))))))

# COMMAND ----------

classes = ['classe_PPScore',\
              'classe_HSPA',\
              'classe_idade',\
              'classe_ddd_state',\
              'classe_tempo_relacionamento',\
              'classe_last_transaction',\
              'classe_instituicao_cartao',\
              'classe_am_max_pay_12m_trx',\
              'classe_modelo_celular',\
              'classe_mental_accounting_score',\
              'classe_peer_effect_score',\
              'classe_email_ajuste',\
              'classe_am_max_recp2p_12m_trx',\
              'classe_emissores_distintos',\
              'digital_account_status',\
              'classe_am_max_payp2pbal_12m_trx']

classes.append('safra')
analise_prod = vars_prod.toPandas().filter(items=classes)

# COMMAND ----------

prod1 = vars_prod.groupBy('safra',\
                          'classe_instituicao_cartao').count().orderBy(desc('count'))
display(prod1)

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 06. Append de Bases - Baseline + Prod  ----------------------------------------------

# COMMAND ----------

classes_final = analise_classes_dev.append(analise_prod, ignore_index=True)

# COMMAND ----------

volume(classes_final, 'safra', 'gist_rainbow')

# COMMAND ----------

dfCartoesCadastrados = spark.table('consumers.credit_cards')\
               .filter(col('consumer_id').isNotNull())\
               .filter(col('bin').isNotNull())\
               .filter(col('registered_at').isNotNull())

dfVariaveisCartoesVisaoCliente2 = dfCartoesCadastrados.select('consumer_id', 'registered_at', 'issuer', 'scheme')
dfVariaveisCartoesVisaoCliente2  = dfVariaveisCartoesVisaoCliente2.withColumn('Last',row_number().over(Window.partitionBy('consumer_id').orderBy(col('registered_at').desc())))
  
dfVariaveisCartoesVisaoCliente_ref = dfVariaveisCartoesVisaoCliente2.groupBy('consumer_id').agg(max(when(col('Last') == 1, col('issuer'))).alias('instituicao_cartao_ref'), max(when(col('Last') == 1, col('scheme'))).alias('bandeira_cartao_ref'))

# COMMAND ----------

display(dfVariaveisCartoesVisaoCliente_ref.groupBy('instituicao_cartao_ref').count().orderBy(desc('count')))

# COMMAND ----------

t2 = dfVariaveisCartoesVisaoCliente_ref.filter(col('consumer_id').isin('25373109','21279803','5510916','27469811','24402342',
                                                '2384827','3084135','3228384','6407684','6931282'))

# COMMAND ----------

display(t2)

# COMMAND ----------

