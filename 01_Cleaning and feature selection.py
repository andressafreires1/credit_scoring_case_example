# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###### Tópicos que serão tratados neste notebook:
# MAGIC 
# MAGIC - 1) Leitura da base de público final e joins com bases de variáveis
# MAGIC - 2) Exclusão de variáveis conceituais
# MAGIC - 3) Pré-Seleção de variáveis: Missing + 0 (Zero) > 95%
# MAGIC - 4) Seleção de variáveis: Boruta, Correlação, Chi2 e IV
# MAGIC - 5) Salva base com a Seleção final
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


from boruta                    import BorutaPy
from pytz                      import timezone
from datetime                  import datetime
from progressbar               import progressbar 
from geneticalgorithm          import geneticalgorithm as GA
from xgboost                   import XGBClassifier
from category_encoders.woe     import WOEEncoder

from sklearn.ensemble          import RandomForestClassifier
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

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 600)

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 02 . Funções ----------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Correlação

# COMMAND ----------

def point_biserial(df, y, num_columns = None, significancia=0.05):
    '''
    Perform feature selection based on correlation test.

            Parameters:
                    df (pandas.dataframe): A dataframe containing all features and target
                    num_columns (list): A list containing all categorical features. If empty list, the function tries to infer the categorical columns itself
                    y (string): A string indicating the target.

            Returns:
                    columns_remove_pb (list): 

    '''
    correlation = []
    p_values = []
    results = []
    
    
    if num_columns:
        num_columns = num_columns
    else:
        num_columns = df.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()
    
    
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
        correlation_aux, p_value_aux = pointbiserialr(df[col], df[y])
        correlation.append(correlation_aux)
        p_values.append(p_value_aux)
    
    
        if p_value_aux <= significancia:
            results.append('Reject H0')
        else:
            results.append('Accept H0')
    
    
    pb_df = pd.DataFrame({'column':num_columns, 'correlation':correlation, 'p_value':p_values, 'result':results})
    columns_remove_pb =  pb_df.loc[pb_df['result']=='Accept H0']['column'].values.tolist()
  
    
    return pb_df, columns_remove_pb

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Boruta

# COMMAND ----------

class Boruta:
    """
    A class to perform feature selection, based on BorutaPy Class of boruta package
    This version is based only on feature importance of a random forest model and returns results more pretifully
    See https://github.com/scikit-learn-contrib/boruta_py for more details (original implementation)

    ...

    Attributes
    ----------
    n_iter : int
        number of iterations the algorithm will perform
    columns_removed : list 
        list of columns to be removed (Obtained after fit method runs)
 

    Methods
    -------
    fit(X, y):
        Runs Boruta Algorithm. It brings a list of columns We should remove and a boolean vetor.
    """

    def __init__(self, n_iter=100):
        """
        Constructs all the necessary attributes for the boruta object.

        Parameters
        ----------
        n_iter : int
            number of iterations the algorithm will perform
        """
        self.n_iter = n_iter
        self._columns_remove_boruta = None
        self._bool_decision = None
        self._best_features = None

    def fit(self, X, y, cat_columns=True, num_columns=True):
        """
        Runs Boruta Algorithm.

        Parameters
        ----------
        X : pandas.dataframe
            Pandas Data Frame with all features
        y: pandas.dataframe
            Pandas Data Frame with target
    
        Returns
        -------
        None
        """
        X.replace(to_replace=[None], value=np.nan, inplace=True)
        if (num_columns == False) & (cat_columns == True):
            cat_columns = X.select_dtypes(include=['object']).columns.tolist()
            cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
            preprocessor = ColumnTransformer(transformers = [('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X,y)
        elif (cat_columns==False) &  (num_columns==True):
            num_columns = X.select_dtypes(include=['int','float']).columns.tolist() 
            num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
            preprocessor = ColumnTransformer(transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X_processed,y)
        else:     
            cat_columns = X.select_dtypes(include=['object']).columns.tolist()
            num_columns = X.select_dtypes(include=['int','float']).columns.tolist() 
            num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
            cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
            preprocessor = ColumnTransformer(transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X_processed,y)
        bool_decision = [not x for x in selector.support_.tolist()] # apenas invertendo o vetor de true/false
        columns_remove_boruta = X.loc[:,bool_decision].columns.tolist()
        columns_keep_boruta = X.loc[:,selector.support_.tolist()].columns.tolist()
        self._columns_remove_boruta = columns_remove_boruta
        self._bool_decision = bool_decision
        self._best_features = columns_keep_boruta
        

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. Chi2

# COMMAND ----------

def chi_squared(df, y, cat_columns = None, significance=0.05):
    '''
    Performs chi2 hypothesis test to find relationship between predictors and target in a data frame

            Parameters:
                    df (pandas.dataframe): A data frame containing categorical features and target variable
                    y (string): A string that saves the name of target variable
                    cat_columns (list): A list with the name of categorical features. If None, function tries to infer It by itself
                    significance (float): A float number indicating the significance level for the test. Deafult is 0.05

            Retorna:
                    chi2_df (pandas.dataframe): A data frame with the results of the tests
                    columns_remove_chi2 (list): A list of columns that should be removed
                    logs (list): A list of columns that could not be evaluated
    '''
    
    
    p_values = []
    logs = []
    chi2_results = []
    results = []
    
    
    if cat_columns == None:
        cat_columns = df.select_dtypes(['object']).columns.tolist()
    else:
        cat_columns = cat_columns
        
        
    for cat in cat_columns:    
        cross_table = pd.crosstab(df[cat], df[y])
        
        
        if not cross_table[cross_table < 5 ].count().any():    
            cross_table = pd.crosstab(df[cat], df[y])
            chi2, p, dof, expected = chi2_contingency(cross_table.values)
            chi2_results.append(chi2)
            p_values.append(p)
        else:
            logs.append("Column {} could'nt be evaluated".format(cat))
            chi2_results.append(np.nan)
            p_values.append(np.nan)
            
            
    for p in p_values:
        
        
        if p <= significance:
            results.append('Reject H0')
        else:
            results.append('Accept H0')   
            
            
    chi2_df = pd.DataFrame({"column":cat_columns, 'p-value':p_values,'chi2':chi2_results, 'results':results})
    columns_remove_chi2 =  chi2_df.loc[chi2_df['results']=='Accept H0']['column'].values.tolist()
    return  chi2_df, columns_remove_chi2, logs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4. IV

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
# MAGIC # ** 03. Público --------------------------------------------------------------
# MAGIC 
# MAGIC - a. Sem relacionamento ou
# MAGIC - b. Com relacionamento < 3 meses ou 
# MAGIC - c. Com relacionamento >= 3 meses porém 0 transações nos últimos 3 meses

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Bases gerais

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

# COMMAND ----------

join4.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Filtros necessários

# COMMAND ----------

# Público do behavior:
  ## Cliente com pelo menos 3 meses de relacionamento e com pelo menos 1 transação nos últimos 3 meses, ou seja, devemos excluir desse modelo pois já é escopo do Behavior.

join4 = join4.withColumn('escopo_behavior',F.when((F.col('tempo_registro_em_meses') >= 3)&(F.col('ct_tot_pay_03m_trx') >= 1), 'Sim'))

filtro1 = join4.filter(F.col('escopo_behavior').isNull())

# COMMAND ----------

filtro1.count()

# COMMAND ----------

# Remover colunas com mesmo nome
spark.sql("set spark.sql.caseSensitive=true")

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

# COMMAND ----------

# MAGIC %md
# MAGIC # ** 04. Parâmetros Chave --------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1. Exclusões

# COMMAND ----------

## Separação do tipo de modelo/desenvolvimento
join6 = join5.withColumn('exclusao',F.when(F.col('SConcEver60dP0_100') == 1, '01. Mau H0')\
                                       .otherwise(F.when((F.col('H2PA') == 0) | (F.col('HSPA') == 0) | (F.col('SCORE_C2BA') == 0), '02. Score Serasa = 0')\
                                          .otherwise(F.when((F.col('H2PA') == 1) | (F.col('HSPA') == 1) | (F.col('SCORE_C2BA') == 1), '03. No-hit Serasa')\
                                             .otherwise(F.when((F.col('SConcEver60dP6_100').isNull()),'04. Sem perf. 6M Serasa')\
                                                 .otherwise(F.when(F.col('has_deactivated') == 'true', '05. Cliente Inativo')\
                                                     .otherwise(F.when((F.col('idade_em_anos') < 18) | (F.col('idade_em_anos') > 90) , '06. Idade < 18 ou > 90')))))))

# Base final de população escorável = 138.351 clientes divididos em 4 safras
populacao_escoravel = join6.filter(F.col('exclusao').isNull())

# COMMAND ----------

# Durante a construção da base amostral, foi escolhida a performance de 60 dias de atraso em 6 meses, com o corte de valores de atrasos acima de 100 reais. Essa definição foi construída pelo bureau Serasa e será usada durante todo o desenvolvimento.

display(join6.groupBy('flag_consumer','exclusao','ref_portfolio','SConcEver60dP6_100').count().orderBy(F.desc('count')))

# COMMAND ----------

populacao_escoravel.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 05. PRÉ-Seleção de variáveis -----------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1. Exclusão variáveis conceituais
# MAGIC 
# MAGIC Exemplo:
# MAGIC   - Variáveis de data (timestamp);

# COMMAND ----------

selecao1 = populacao_escoravel.drop('clientes_D0_registered_at',\
                                    'upgraded_pro_at',\
                                    'registered_at',\
                                    'pro_activated_at',\
                                    'first_completed_all_transaction_created_at',\
                                    'last_transaction_created_at',\
                                    'dt_registro',\
                                    'data_referencia_aj',\
                                    'clientes_D0_timestamp_processamento',\
                                    'vars_cadastrais_timestamp_processamento',\
                                    'vars_cadastrais_registered_at',\
                                    'first_use_at',\
                                    'created_at',\
                                    'tpv_estats_basicas_timestamp_processamento',\
                                    'tpv_estats_basicas_data_transacoes',\
                                    'tpv_mediana_timestamp_processamento',\
                                    'tpv_mediana_timestamp_processamento',\
                                    'tpv_mediana_data_transacoes',\
                                    'cartoes_cadastrados_visao_cliente_timestamp_processamento',\
                                    'cartoes_cadastrados_visao_cliente_dt_dt_cadastro_cartoes',\
                                    'mixpanel_timestamp_processamento',\
                                    'ConcEver60dP0_100',\
                                    'SConcEver60dP0_100',\
                                    'FLAG_60D0M_1B',\
                                    'FLAG_60D0M_2B',\
                                    'ConcEver60dP6_100',\
                                    'ref_serasa',\
                                    'exclusao',\
                                    'birth_year',\
                                    'ano_nascimento',\
                                    'ano_referencia',\
                                    'idade',\
                                    'dt_cadastro_cliente',\
                                    'mixpanel_dt_cadastro_cliente',\
                                    'safra_cadastro',\
                                    'selecao',\
                                    'has_deactivated',\
                                    'data_referencia',\
                                    'escopo_behavior')      

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2. % de Missing % de Zeros (0)
# MAGIC 
# MAGIC Serão removidas as variáveis cujo percentual de missing seja > 95%

# COMMAND ----------

#Seleção mais simples, porém mais rápida para fazer uma segunda limpeza nas variáveis que possivelmente ainda tenham ficado no step anterior.
def missing(df):
  return df.select([(F.count(F.when(F.col(c).isNull(), c))).alias(c) for c in df.columns])


def zeros(df):
  columns = [item[0] for item in df.dtypes if item[1].startswith('int') or item[1].startswith('long') or item[1].startswith('double') or item[1].startswith('bigint') or item[1].startswith('decimal') or item[1].startswith('bigint') or item[1].startswith('float')]
  df_aux = df.select(columns)          
  return df_aux.select([(F.count(F.when((F.col(c)==0), c))).alias(c) for c in df_aux.columns])

# COMMAND ----------

missing = missing(selecao1)
display(missing)

# COMMAND ----------

zeros = zeros(selecao1)
display(zeros)

# COMMAND ----------

## Removendo variáveis cujos % de missing e zero (missing + zeros) seja > 95% 

selecao2 = selecao1.drop(# Recomevendo %missing e zeros > 95%
'qtde_Privacidade_alterada_',
'qtde_Page_View_-_Credito_-_Documentos_',
'qtde_UA_-_screenview_',
'qtde_3DS_-_Challenge_Response_',
'qtde_DONATIONS_',
'qtde_Transaccaao_DEPOSIT_DEBIT_',
'qtde_Bills_-_Payment_with_Interest_',
'qtde_External_Transfer_Detail_Viewed_',
'qtde_Continued_Screen_-_Option_Selected_',
'qtde_SERVICES_STORES_',
'qtde_Installment_Warning_Viewed_',
'qtde_MGM_-_App_-_PGP_Landing_Page_',
'qtde_Cash-in_-_Touched_Switch_Debt_Card_',
'qtde_External_Transfer_Add_Funds_',
'qtde_External_Transfer_Sent_',
'qtde_External_Transfer_Insufficient_Funds_Back_',
'qtde_2FA_-_Phone_Number_Validated_',
'qtde_2FA_-_Bank_Account_Validated_',
'qtde_Bottom_Sheet_Item_Clicked_',
'qtde_Cash-in_-_Debt_Card_Switched_',
'qtde_Bills_-_Payment_with_Discount_',
'qtde_External_Transfer_Payment_Succeeded_',
'qtde_Bills_-_Discount_Information_Viewed_',
'qtde_Bills_-_Discount_Information_Closed_',
'qtde_Bills_-_Payment_Timeout_Denied_-_Not_Now_',
'qtde_External_Transfer_Intro_Viewed_',
'qtde_Continued_Screen_-_Viewed_',
'qtde_Tabbar_-_Tab_Selected_',
'qtde_Profile_pic_touch_',
'qtde_Bills_-_Payment_Timeout_Denied_',
'qtde_Profile_Accessed_',
'qtde_Bills_-_Discount_Information_Understood_',
'qtde_Bills_-_Interest_Information_Viewed_',
'qtde_Bottom_Sheet_Accessed_',
'qtde_Bills_-_Interest_Information_Closed_',
'qtde_Bills_-_Interest_Information_Understood_',
'qtde_External_Transfer_Privacy_Opened_',
'qtde_Sign_Up_CPF_Pop_Up_-_Viewed_',
'qtde_External_Transfer_Insufficient_Funds_',
'qtde_Social_Status_Changed_',
'qtde_SignUp_CPF_Pop_Up_-_Option_Selected_',
'qtde_Bills_-_Payment_Timeout_Denied_-_Try_Again_',
'qtde_External_Transfer_Card_Detail_Viewed_',
'qtde_Social_Status_Viewed_',
'qtde_External_Transfer_Payment_Method_Opened_',
'qtde_External_Transfer_Started_',
'qtde_3DS_-_Error_',
'qtde_External_Transfer_Card_Clicked_',
'qtde_External_Transfer_Send_Link_',
'qtde_UA_-_Profile_cta_offer_',
'MENSAGEM_SCORE',
'qtde_2FA_-_Back_button_confirmed_',
'qtde_2FA_-_Code_Resended_',
'qtde_2FA_-_Back_button_selected_',
'qtde_2FA_-_Help_Accessed_',
'qtde_Sign_Up_CPF_-_In_Use_',
'qtde_Popup_MGM_',
'qtde_Credito_-_visualizou_oferta_-_feed_',
'qtde_Credito_-_visualizou_oferta_-_carteira_',
'qtde_AppParameters_userToken_nil_',
'qtde_2FA_-_Authorization_Method_Chosen_',
'qtde_2FA_-_Data_Validated_',
'qtde_2FA_-_Authorization_Methods_Viewed_',
'qtde_2FA_-_Started_',
'qtde_FT_payment_info_',
'qtde_FT_recarga_',
'qtde_Finalizaccaao_PAV_',
'qtde_FT_profile_',
'qtde_Rendimento_-_tocou_no_resumo_',
'qtde_2FA_-_Por_que_preciso_fazer_isso_',
'qtde_Limite_P2P_excedido_cartaao_',
'qtde_Credito_-_tap_item_feed_',
'qtde_Page_view_-_Credito_-_listar_faturas_',
'qtde_DynamicLinkFirebase_',
'qtde_Hint_-_Email_sugerido_verificado_',
'qtde_Cashback_debloqueado_',
'qtde_2FA_-_Logou_via_SMS_',
'qtde_Abriu_Aniversariante_do_Dia_',
'qtde_Credito_-_Documentos_-_Fazer_depois_-_comprovante_residencia_',
'qtde_Tela_de_Verificaccaao_de_Telefone_',
'qtde_Tela_de_Confirmaccaao_do_Coodigo_SMS_',
'qtde_Saque24horas_-_Show_Popup_',
'qtde_PCI_-_Migration_Done_',
'qtde_Credito_-_Documentos_-_Fazer_depois_-_Verso_documento_',
'qtde_Alteraccaao_de_Telefone_verificaccaao_SMS_',
'qtde_2FA_-_Reenviar_coodigo_',
'qtde_Payment_Request_Limit_Exceeded_Restart_',
'qtde_Payment_Request_Limit_Exceeded_',
'qtde_2FA_-_Validou_dados_',
'qtde_Creedito_-_oferta_pop-up_CTA_',
'qtde_2FA_-_Logou_via_Email_',
'qtde_2FA_-_Via_dispositivo_antigo_',
'qtde_2FA_-_Via_Email_',
'qtde_Credito_-_Documentos_-_comprovante_residencia_',
'qtde_Solicitou_Recarga_',
'qtde_2FA_-_Via_SMS_',
'qtde_Sucesso_Logout_',
'qtde_2FA_-_Escolher_tipo_de_autorizaccaao_',
'qtde_2FA_-_Iniicio_',
'am_min_paypavppc_01m_trx',
'am_max_paypavppc_01m_trx',
'am_tot_paypavppc_01m_trx',
'am_avg_paypavppc_01m_trx',
'ct_tot_paypavppc_01m_trx',
'am_min_paypavppc_03m_trx',
'am_max_paypavppc_03m_trx',
'am_tot_paypavppc_03m_trx',
'am_avg_paypavppc_03m_trx',
'ct_tot_paypavppc_03m_trx',
'ct_min_paypavppc_03m_trx',
'ct_max_paypavppc_03m_trx',
'ct_avg_paypavppc_03m_trx',
'am_min_paypavppc_06m_trx',
'am_max_paypavppc_06m_trx',
'am_tot_paypavppc_06m_trx',
'am_avg_paypavppc_06m_trx',
'ct_tot_paypavppc_06m_trx',
'ct_min_paypavppc_06m_trx',
'ct_max_paypavppc_06m_trx',
'ct_avg_paypavppc_06m_trx',
'am_min_paypavppc_12m_trx',
'am_max_paypavppc_12m_trx',
'am_tot_paypavppc_12m_trx',
'am_avg_paypavppc_12m_trx',
'ct_tot_paypavppc_12m_trx',
'ct_min_paypavppc_12m_trx',
'ct_max_paypavppc_12m_trx',
'ct_avg_paypavppc_12m_trx',
'am_min_paypavbill_01m_trx',
'am_max_paypavbill_01m_trx',
'am_tot_paypavbill_01m_trx',
'am_avg_paypavbill_01m_trx',
'ct_tot_paypavbill_01m_trx',
'qtde_cartoes_da_bandeira_original',
'qtde_Saque_-_tocou_no_iicone_de_duuvida_',
'qtde_CTA_para_contratar_Creedito_-_Oferta_Creedito_',
'qtde_Payment_Request_Send_',
'qtde_Page_view_-_Credito_-_Cadastro_negado_',
'qtde_ID_Validation_-_Enviou_fotos_da_Identidade_',
'qtde_ID_Validation_-_Enviou_selfie_',
'qtde_FT_tela_transaccaao_PPP_',
'qtde_PicPay_Card_Request_-_Offer_Clicked_',
'qtde_Page_view_-_Oferta_Creedito_',
'qtde_Page_view_-_Credito_-_Infomaccoooes_pendentes_',
'qtde_Creedito_-_oferta_pop-up_dismiss_',
'qtde_No_token_in_request_',
'qtde_Erro_Logout_',
'qtde_Saque24Horas_-_Erro_Autorizacao_',
'qtde_Credito_-_Documentos_-_Fazer_depois_-_frente_documento_',
'qtde_Exibiu_Ajuda_do_CPF_',
'qtde_Parcelamento_sem_juros_desativado_',
'qtde_Saque_-_Cadastrar_outra_conta_',
'scheduled_pro_date',
'qtde_Payment_Request_Notification_Accessed_',
'qtde_Payment_Request_Privacy_Opened_',
'qtde_Credito_-_Enderecco_comercial_',
'qtde_Page_view_-_Credito_-_home_',
'qtde_Credito_-_Documentos_-_Fazer_depois_-_enviou_selfie_',
'qtde_Payment_Request_Error_',
'qtde_First_time_open_',
'qtde_Bills_-_Payment_Due_Date_Denied_',
'qtde_Permissao_autorizada_-_Notificacao_',
'qtde_Cadastro_Verificaccaao_SMS_-_Erro_',
'qtde_Upgrade_-_Tocou_na_oferta_',
'qtde_Landing_Screen_-_Remote_Config_Loaded_',
'qtde_Map_Accessed_',
'qtde_Start_Screen_-_Viewed_',
'qtde_Request_Detail_Viewed_',
'qtde_Saque_-_Corrigir_dados_',
'qtde_Payment_Request_Value_Confirmed_',
'qtde_Cobrar_-_tocou_no_botaao_informe_o_valor_',
'qtde_Insert_Code_-_Option_Selected_',
'qtde_AutomaticLogoutUserBlocked_',
'qtde_AutomaticLogout_',
'qtde_Parcelamento_sem_juros_ativado_',
'qtde_Page_view_-_Credito_-_Documento_identidade_',
'qtde_Payment_Request_Value_Edition_',
'qtde_Page_view_-_Credito_-_Pedido_enviado_',
'qtde_UA_-_popup_agora_nao_',
'qtde_Payment_Request_Pay_',
'qtde_Onboarding_Cash-In_-_Error_',
'qtde_Payment_Request_Transaction_Viewed_',
'qtde_Credito_-_Documentos_-_enviou_selfie_',
'qtde_Credito_-_Documentos_-_verso_documento_',
'qtde_Credito_-_Documentos_-_frente_documento_',
'qtde_Boleto_-_Viu_recibo_',
'qtde_Page_view_-_Credito_-_Resumo_do_pedido_',
'qtde_Recharge_cancelled_',
'qtde_Page_view_-_Credito_-_Dados_adicionais_',
'qtde_DG_-_Reminder_Confirmed_',
'qtde_Favorites_Onboarding_Left_',
'qtde_Register_-_Screen_Viewed_',
'qtde_Saque24Horas_-_Scanner_Informaccoooes_',
'qtde_Cadastro_verificaccaao_SMS_',
'qtde_Cadastro_CPF_',
'qtde_Saque24Horas_-_Concluido_',
'qtde_Verificaccaao_de_cartaao_CANCELADA_',
'qtde_UA_-_popup_fechar_',
'qtde_MGM_-_Ganhou_por_indicar_',
'qtde_Page_view_-_Credito_-_Documentos_',
'qtde_Cadastro_senha_',
'qtde_Payment_Item_Popup_Accessed_',
'qtde_Locais_-_Tocou_no_pin_',
'qtde_Transaccaao_MEMBERSHIP_',
'qtde_Transaccaao_P2P_Cancelada_',
'qtde_Creedito_-_oferta_botaao_Ver_detalhes_',
'qtde_Payment_Item_Popup_Opened_',
'qtde_Onboarding_social_-_Seguiu_',
'qtde_Locais_-_Tocou_em_pagar_no_perfil_',
'qtde_Hub_Viewed_',
'qtde_Receive_Code_-_Whatsapp_Flow_',
'qtde_Cobrar_-_tocou_no_valor_',
'qtde_Saque24Horas_-_Confirmou_',
'qtde_Saque24Horas_-_Scanneou_',
'qtde_Saque24Horas_-_Novo_Status_Original_',
'qtde_Payment_Request_Success_',
'qtde_Payment_Request_Sent_',
'qtde_Page_view_-_Credito_-_Documentos_identidade_',
'qtde_Saque24Horas_-_Saque_Autorizado_',
'qtde_Payment_Request_Value_Not_Informed_',
'qtde_Payment_Request_Detail_Viewed_',
'qtde_PicPay_Card_Request_-_Offer_Webview_',
'qtde_UA_-_Tela_Regras_',
'am_min_paypavinstp2m_01m_trx',
'am_max_paypavinstp2m_01m_trx',
'am_tot_paypavinstp2m_01m_trx',
'am_avg_paypavinstp2m_01m_trx',
'ct_tot_paypavinstp2m_01m_trx',
'qtde_Saque24Horas_-_Limite_Informaccoooes_',
'qtde_Start_Screen_-_Option_Selected_',
'qtde_Page_view_-_Credito_-_Enderecco_',
'qtde_Page_view_-_Credito_-_Local_de_nascimento_',
'am_min_paypavinstbiz_01m_trx',
'am_max_paypavinstbiz_01m_trx',
'am_tot_paypavinstbiz_01m_trx',
'am_avg_paypavinstbiz_01m_trx',
'ct_tot_paypavinstbiz_01m_trx',
'is_geo_setup_matching_region_telcad',
'am_min_paypavinstotr_01m_trx',
'am_max_paypavinstotr_01m_trx',
'am_tot_paypavinstotr_01m_trx',
'am_avg_paypavinstotr_01m_trx',
'ct_tot_paypavinstotr_01m_trx',
'qtde_Favorite_Status_Changed_',
'qtde_Reactivate_Account_',
'qtde_Payment_Request_User_Selected_',
'qtde_Page_view_-_Credito_-_Dados_pessoais_',
'qtde_Friend_Activation_Notification_',
'qtde_Locais_-_Entrou_no_mapa_',
'qtde_UA_-_Tela_Onboarding_',
'is_geo_setup_matching_region_bureaus',
'qtde_Favorites_Onboarding_Accessed_',
'qtde_Favorites_List_Accessed_',
'qtde_Hint_-_Seleciona_sugestaao_de_telefone_',
'qtde_Saque24Horas_-_Entrou_Scanner_',
'qtde_Cobrar_-_naao_informou_valor_',
'qtde_Autoclearance_Notification_',
'qtde_Cadastro_telefone_',
'am_min_paypavinstp2m_03m_trx',
'am_max_paypavinstp2m_03m_trx',
'am_tot_paypavinstp2m_03m_trx',
'am_avg_paypavinstp2m_03m_trx',
'ct_tot_paypavinstp2m_03m_trx',
'ct_min_paypavinstp2m_03m_trx',
'ct_max_paypavinstp2m_03m_trx',
'ct_avg_paypavinstp2m_03m_trx',
'qtde_Recibo_-_Auto_Analise_',
'qtde_Transaccaao_P2M_',
'qtde_Transacao_POS_',
'am_min_paypavinstp2m_06m_trx',
'am_max_paypavinstp2m_06m_trx',
'am_tot_paypavinstp2m_06m_trx',
'am_avg_paypavinstp2m_06m_trx',
'ct_tot_paypavinstp2m_06m_trx',
'ct_min_paypavinstp2m_06m_trx',
'ct_max_paypavinstp2m_06m_trx',
'ct_avg_paypavinstp2m_06m_trx',
'am_min_paypavinstp2m_12m_trx',
'am_max_paypavinstp2m_12m_trx',
'am_tot_paypavinstp2m_12m_trx',
'am_avg_paypavinstp2m_12m_trx',
'ct_tot_paypavinstp2m_12m_trx',
'ct_min_paypavinstp2m_12m_trx',
'ct_max_paypavinstp2m_12m_trx',
'ct_avg_paypavinstp2m_12m_trx',
'qtde_Payment_Request_User_Listing_Viewed_',
'qtde_UA_-_Tela_de_comprovante_',
'am_min_paypavinstotr_03m_trx',
'am_max_paypavinstotr_03m_trx',
'am_tot_paypavinstotr_03m_trx',
'am_avg_paypavinstotr_03m_trx',
'ct_tot_paypavinstotr_03m_trx',
'ct_min_paypavinstotr_03m_trx',
'ct_max_paypavinstotr_03m_trx',
'ct_avg_paypavinstotr_03m_trx',
'qtde_Payment_Request_Value_Started_',
'qtde_User_Activated_-_PRO_',
'qtde_Payment_Request_Share_Type_',
'qtde_UA_-_Tela_cadastro_de_universidade_',
'am_min_paypavinstbiz_03m_trx',
'am_max_paypavinstbiz_03m_trx',
'am_tot_paypavinstbiz_03m_trx',
'am_avg_paypavinstbiz_03m_trx',
'ct_tot_paypavinstbiz_03m_trx',
'ct_min_paypavinstbiz_03m_trx',
'ct_max_paypavinstbiz_03m_trx',
'ct_avg_paypavinstbiz_03m_trx',
'qtde_UA_-_Tela_Benefiicios_',
'qtde_Hint_-_Mostrou_sugestaao_telefone_',
'qtde_UA_-_Tela_Conta_Ativada_',
'qtde_Cadastro_nome_',
'qtde_Hint_-_Seleciona_sugestaao_de_email_',
'am_min_paypavinstgood_01m_trx',
'am_max_paypavinstgood_01m_trx',
'am_tot_paypavinstgood_01m_trx',
'am_avg_paypavinstgood_01m_trx',
'ct_tot_paypavinstgood_01m_trx',
'qtde_Virou_PRO_',
'qtde_Payment_Request_Method_Selected_',
'qtde_Review_',
'qtde_GAMES_',
'qtde_Onboarding_social_-_Cancelou_',
'qtde_Creedito_-_oferta_swype_right_no_Saldo_',
'qtde_Inicio_-_Feed_Scrolled_',
'qtde_Payment_Request_Send_Request_',
'qtde_Saque_Original_-_Vantagens_Swipe_',
'qtde_Hint_-_Telefone_sugerido_verificado_',
'qtde_Hint_-_Mostrou_sugestaao_email_',
'qtde_ID_Validation_-_Respondeu_questionaaario_',
'students_id',
'qtde_Viu_Popup_de_Review_',
'qtde_Upgrade_-_Preencheu_nome_da_mae_',
'qtde_Saque_-_entrou_na_tela_de_previsaao_',
'qtde_Deactivate_Account_',
'qtde_UA_-_Enviou_os_comprovantes_',
'qtde_Converted_-_PRO_',
'am_min_paypavinstbiz_06m_trx',
'am_max_paypavinstbiz_06m_trx',
'am_tot_paypavinstbiz_06m_trx',
'am_avg_paypavinstbiz_06m_trx',
'ct_tot_paypavinstbiz_06m_trx',
'ct_min_paypavinstbiz_06m_trx',
'ct_max_paypavinstbiz_06m_trx',
'ct_avg_paypavinstbiz_06m_trx',
'qtde_UA_-_Anexou_comprovante_',
'qtde_Transaccaao_ECOMMERCE_',
'qtde_Onboarding_de_Verificaccaao_de_Cartaao_RECUSOU_',
'qtde_UA_-_Tela_de_status_',
'qtde_UA_-_Tirou_selfie_',
'am_min_paypavinstotr_06m_trx',
'am_max_paypavinstotr_06m_trx',
'am_tot_paypavinstotr_06m_trx',
'am_avg_paypavinstotr_06m_trx',
'ct_tot_paypavinstotr_06m_trx',
'ct_min_paypavinstotr_06m_trx',
'ct_max_paypavinstotr_06m_trx',
'ct_avg_paypavinstotr_06m_trx',
'qtde_QR_code_-_compartilhou_',
'qtde_Creedito_-_oferta_carteira_Swype_right_saldo_',
'qtde_ID_Validation_-_Finalizou_',
'qtde_Recharge_complete_',
'qtde_FT_saque_',
'qtde_Payment_Request_Value_Informed_',
'am_min_paypavinstbiz_12m_trx',
'am_max_paypavinstbiz_12m_trx',
'am_tot_paypavinstbiz_12m_trx',
'am_avg_paypavinstbiz_12m_trx',
'ct_tot_paypavinstbiz_12m_trx',
'ct_min_paypavinstbiz_12m_trx',
'ct_max_paypavinstbiz_12m_trx',
'ct_avg_paypavinstbiz_12m_trx',
'qtde_Payment_Request_QR_Code_Generated_',
'qtde_Onboarding_cartaao_de_creedito_-_Fechou_',
'qtde_Cadastro_email_',
'qtde_mp_event_',
'am_min_paypavinstgood_03m_trx',
'am_max_paypavinstgood_03m_trx',
'am_tot_paypavinstgood_03m_trx',
'am_avg_paypavinstgood_03m_trx',
'ct_tot_paypavinstgood_03m_trx',
'ct_min_paypavinstgood_03m_trx',
'ct_max_paypavinstgood_03m_trx',
'ct_avg_paypavinstgood_03m_trx',
'qtde_Payment_Request_Hub_Viewed_',
'qtde_UA_-_Universidade_cadastrada_',
'qtde_Logout_',
'qtde_QR_code_-_tocou_em_compartilhar_',
'am_min_paypavinstotr_12m_trx',
'am_max_paypavinstotr_12m_trx',
'am_tot_paypavinstotr_12m_trx',
'am_avg_paypavinstotr_12m_trx',
'ct_tot_paypavinstotr_12m_trx',
'ct_min_paypavinstotr_12m_trx',
'ct_max_paypavinstotr_12m_trx',
'ct_avg_paypavinstotr_12m_trx',
'qtde_Upgrade_-_Validou_identidade_',
'qtde_Rendimento_-_Retirar_',
'qtde_Carteira_-_Income_Accessed_by_value_',
'qtde_Rendimento_-_Acessou_FAQ_',
'qtde_SERVICES_',
'qtde_Debit_Card_-_Not_Accepted_',
'qtde_Upgrade_-_Preencheu_nome_da_maae_',
'qtde_Fingerprint_Auth_Enable_',
'qtde_Saque24Horas_-_Opccoooes_de_valores_',
'qtde_Limite_P2P_excedido_',
'qtde_Promotions_Detail_Accessed_',
'qtde_Verificaccaao_de_cartaao_CONFIRMOU_',
'qtde_Cobrar_-_tipo_de_compartilhamento_',
'qtde_Qr_Code_-_Accessed_',
'cadastrou_e_removeu_algum_cartao_em_D0',
'qtde_cartoes_cadastrados_e_removidos_em_D0',
'qtde_Boleto_-_Viu_status_',
'qtde_Installments_Selected_',
'qtde_Saque_solicitado_',
'am_min_paypavinstgood_06m_trx',
'am_max_paypavinstgood_06m_trx',
'am_tot_paypavinstgood_06m_trx',
'am_avg_paypavinstgood_06m_trx',
'ct_tot_paypavinstgood_06m_trx',
'ct_min_paypavinstgood_06m_trx',
'ct_max_paypavinstgood_06m_trx',
'ct_avg_paypavinstgood_06m_trx',
'qtde_ID_Validation_-_Iniciou_',
'qtde_Upgrade_-_Preencheu_renda_',
'qtde_Upgrade_-_Preencheu_enderecco_',
'qtde_Saque_-_iniciou_cadastro_de_conta_',
'qtde_Verificaccaao_de_Cartaao_INICIOU_',
'am_min_paypavotr_01m_trx',
'am_max_paypavotr_01m_trx',
'am_tot_paypavotr_01m_trx',
'am_avg_paypavotr_01m_trx',
'ct_tot_paypavotr_01m_trx',
'qtde_Scanner_CC_Escaneou_',
'qtde_Cobrar_-_tocou_em_enviar_cobrancca_',
'qtde_Payment_Request_Started_',
'qtde_ID_Validation_-_Acessou_FT_',
'qtde_Finalizou_',
'qtde_Exibiu_popup_cashback_',
'qtde_Installment_Accessed_',
'qtde_Onboarding_de_Verificaccaao_de_Cartaao_ACEITOU_',
'qtde_toque_no_botao_',
'qtde_Payment_Request_Search_Activated_',
'am_min_paypavacq_01m_trx',
'am_max_paypavacq_01m_trx',
'am_tot_paypavacq_01m_trx',
'am_avg_paypavacq_01m_trx',
'ct_tot_paypavacq_01m_trx',
'qtde_Cashback_Primeiro_Recebimento_P2P_',
'qtde_Enviou_fotos_da_identidade_',
'qtde_Respondeu_questionaaario_',
'qtde_MgM_Converted_',
'am_min_paypavinstgood_12m_trx',
'am_max_paypavinstgood_12m_trx',
'am_tot_paypavinstgood_12m_trx',
'am_avg_paypavinstgood_12m_trx',
'ct_tot_paypavinstgood_12m_trx',
'ct_min_paypavinstgood_12m_trx',
'ct_max_paypavinstgood_12m_trx',
'ct_avg_paypavinstgood_12m_trx',
'qtde_Saque_-_opccaao_escolhida_',
'qtde_Cobrar_-_informou_valor_',
'qtde_Cobrar_-_qr_code_gerado_',
'qtde_Notifications_-_Setup_Accessed_',
'qtde_Scanner_CC_Abriu_',
'qtde_Sucesso_ao_escanear_codigo_',
'qtde_Enviou_Selfie_',
'qtde_Transaccaao_DG_',
'qtde_Upgrade_-_Iniciou_',
'qtde_MGM_Recibo_',
'qtde_cartoes_da_bandeira_amex',
'qtde_Privacidade_Alterada_',
'am_min_paypavotr_03m_trx',
'am_max_paypavotr_03m_trx',
'am_tot_paypavotr_03m_trx',
'am_avg_paypavotr_03m_trx',
'ct_tot_paypavotr_03m_trx',
'ct_min_paypavotr_03m_trx',
'ct_max_paypavotr_03m_trx',
'ct_avg_paypavotr_03m_trx',
'qtde_Iniciou_',
'is_mother_name_correct',
'qtde_Promotions_Onboarding_Accessed_',
'qtde_Rendimento_-_Adicionar_',
'qtde_Inseriu_coodigo_promocional_-_sucesso_',
'tpv_intervalo_no_dia',
'tpv_desvio_padrao_no_dia',
'tpv_intervalo_sobre_media',
'tpv_desvio_padrao_sobre_media',
'intervalo_sobre_media_tempo_permanencia_cartao_horas',
'desvio_padrao_sobre_media_tempo_permanencia_cartao_horas',
'intervalo_sobre_media_tempo_permanencia_cartao_minutos',
'desvio_padrao_sobre_media_tempo_permanencia_cartao_minutos',
'intervalo_sobre_media_tempo_permanencia_cartao_segundos',
'desvio_padrao_sobre_media_tempo_permanencia_cartao_segundos',
'intervalo_tempo_permanencia_cartao_horas',
'desvio_padrao_tempo_permanencia_cartao_horas',
'intervalo_tempo_permanencia_cartao_minutos',
'desvio_padrao_tempo_permanencia_cartao_minutos',
'intervalo_tempo_permanencia_cartao_segundos',
'desvio_padrao_tempo_permanencia_cartao_segundos',
'am_min_paypavp2m_01m_trx',
'am_max_paypavp2m_01m_trx',
'am_tot_paypavp2m_01m_trx',
'am_avg_paypavp2m_01m_trx',
'ct_tot_paypavp2m_01m_trx',
'qtde_Cashback_P2P_Recebido_',
'qtde_Carteira_-_tocou_em_retirar_',
'qtde_CONSUMERS_',
'am_min_paypavacq_03m_trx',
'am_max_paypavacq_03m_trx',
'am_tot_paypavacq_03m_trx',
'am_avg_paypavacq_03m_trx',
'ct_tot_paypavacq_03m_trx',
'ct_min_paypavacq_03m_trx',
'ct_max_paypavacq_03m_trx',
'ct_avg_paypavacq_03m_trx',
'qtde_Transaccaao_BILLS_',
'qtde_Recharge_created_',
'qtde_Acessou_FT_',
'qtde_Saiu_da_tela_sem_escanear_',
'qtde_Cobrar_-_iniciou_cobrancca_',
'qtde_Onboarding_Cash-In_-_Payment_Method_Selected_',
'qtde_Permissao_autorizada_-_Armazenamento_',
'am_min_paypavacq_06m_trx',
'am_max_paypavacq_06m_trx',
'am_tot_paypavacq_06m_trx',
'am_avg_paypavacq_06m_trx',
'ct_tot_paypavacq_06m_trx',
'ct_min_paypavacq_06m_trx',
'ct_max_paypavacq_06m_trx',
'ct_avg_paypavacq_06m_trx',
'qtde_Promotions_Onboarding_Viewed_',
'am_min_paypavotr_06m_trx',
'am_max_paypavotr_06m_trx',
'am_tot_paypavotr_06m_trx',
'am_avg_paypavotr_06m_trx',
'ct_tot_paypavotr_06m_trx',
'ct_min_paypavotr_06m_trx',
'ct_max_paypavotr_06m_trx',
'ct_avg_paypavotr_06m_trx',
'qtde_Installment_Button_Viewed_',
'am_min_paypavinstbill_01m_trx',
'am_max_paypavinstbill_01m_trx',
'am_tot_paypavinstbill_01m_trx',
'am_avg_paypavinstbill_01m_trx',
'ct_tot_paypavinstbill_01m_trx',
'qtde_Cupom_aplicado_',
'qtde_QR_code_-_coodigo_lido_',
'qtde_Recarga_-_DDD_',
'am_min_paypavacq_12m_trx',
'am_max_paypavacq_12m_trx',
'am_tot_paypavacq_12m_trx',
'am_avg_paypavacq_12m_trx',
'ct_tot_paypavacq_12m_trx',
'ct_min_paypavacq_12m_trx',
'ct_max_paypavacq_12m_trx',
'ct_avg_paypavacq_12m_trx',
'qtde_Nova_Conta_Bancaaaria_',
'qtde_Onboarding_social_-_convidar_amigos_',
'qtde_Rendimento_-_Scroll_no_resumo_',
'am_min_paypavbiz_01m_trx',
'am_max_paypavbiz_01m_trx',
'am_tot_paypavbiz_01m_trx',
'am_avg_paypavbiz_01m_trx',
'ct_tot_paypavbiz_01m_trx',
'qtde_FT_tela_transaccaao_PAV_',
'qtde_Transaccaao_PAV_',
'qtde_Pagar_-_Search_Accessed_',
'qtde_Credito_-_fatura_pagar_-_trocou_meio_de_pagamento_',
'am_min_recp2pmix_01m_trx',
'am_max_recp2pmix_01m_trx',
'am_tot_recp2pmix_01m_trx',
'am_avg_recp2pmix_01m_trx',
'ct_tot_recp2pmix_01m_trx',
'qtde_Popup_de_sair_do_cadastro_de_cartao_de_credito_',
'qtde_Retirar_Dinheiro_-_Listou_Opccoooes_',
'am_min_paypavp2m_03m_trx',
'am_max_paypavp2m_03m_trx',
'am_tot_paypavp2m_03m_trx',
'am_avg_paypavp2m_03m_trx',
'ct_tot_paypavp2m_03m_trx',
'ct_min_paypavp2m_03m_trx',
'ct_max_paypavp2m_03m_trx',
'ct_avg_paypavp2m_03m_trx',
'am_min_paypavotr_12m_trx',
'am_max_paypavotr_12m_trx',
'am_tot_paypavotr_12m_trx',
'am_avg_paypavotr_12m_trx',
'ct_tot_paypavotr_12m_trx',
'ct_min_paypavotr_12m_trx',
'ct_max_paypavotr_12m_trx',
'ct_avg_paypavotr_12m_trx',
'qtde_cartoes_da_bandeira_hiper',
'qtde_scanned_text_',
'qtde_Adicionou_foto_de_perfil_',
'qtde_API_Error_',
'qtde_Inicio_-_Card_Opportunities_Touched_',
'qtde_Onboarding_Cash-In_-_Payment_Method_Viewed_',
'am_min_payp2pmix_01m_trx',
'am_max_payp2pmix_01m_trx',
'am_tot_payp2pmix_01m_trx',
'am_avg_payp2pmix_01m_trx',
'ct_tot_payp2pmix_01m_trx',
'qtde_Onboarding_Cash-In_-_Carousel_Viewed_',
'qtde_Pagar_-_Search_Result_Viewed_',
'am_min_paypavp2m_06m_trx',
'am_max_paypavp2m_06m_trx',
'am_tot_paypavp2m_06m_trx',
'am_avg_paypavp2m_06m_trx',
'ct_tot_paypavp2m_06m_trx',
'ct_min_paypavp2m_06m_trx',
'ct_max_paypavp2m_06m_trx',
'ct_avg_paypavp2m_06m_trx',
'qtde_Inicio_-_Invite-People_Accessed_',
'qtde_PicPay_Card_Credit_Granted_',
'qtde_First_Screen_Viewed_',
'qtde_Onboarding_cartaao_de_creedito_-_Pulou_',
'qtde_QR_code_-_meu_coodigo_',
'qtde_Promotions_Home_Button_Viewed_',
'qtde_Remote_Config_-_Activate_',
'qtde_Remote_Config_-_Fetch_',
'qtde_Remote_Config_-_Start_',
'am_min_paypavp2m_12m_trx',
'am_max_paypavp2m_12m_trx',
'am_tot_paypavp2m_12m_trx',
'am_avg_paypavp2m_12m_trx',
'ct_tot_paypavp2m_12m_trx',
'ct_min_paypavp2m_12m_trx',
'ct_max_paypavp2m_12m_trx',
'ct_avg_paypavp2m_12m_trx',
'qtde_Respondeu_requisiccaao_de_contatos_do_iOS_',
'am_min_paypavinstbill_03m_trx',
'am_max_paypavinstbill_03m_trx',
'am_tot_paypavinstbill_03m_trx',
'am_avg_paypavinstbill_03m_trx',
'ct_tot_paypavinstbill_03m_trx',
'ct_min_paypavinstbill_03m_trx',
'ct_max_paypavinstbill_03m_trx',
'ct_avg_paypavinstbill_03m_trx',
'qtde_FT_Pagar_-_Principais_',
'qtde_Cashback_Recebido_',
'qtde_Carteira_-_Income_Accessed_by_button_',
'qtde_Landing_Screen_-_Viewed_',
'qtde_Installments_screen_accessed_FT_',
'am_min_paypavmix_01m_trx',
'am_max_paypavmix_01m_trx',
'am_tot_paypavmix_01m_trx',
'am_avg_paypavmix_01m_trx',
'ct_tot_paypavmix_01m_trx',
'qtde_Onboarding_social_-_pulou_convidar_amigos_',
'am_min_recp2pmix_03m_trx',
'am_max_recp2pmix_03m_trx',
'am_tot_recp2pmix_03m_trx',
'am_avg_recp2pmix_03m_trx',
'ct_tot_recp2pmix_03m_trx',
'ct_min_recp2pmix_03m_trx',
'ct_max_recp2pmix_03m_trx',
'ct_avg_recp2pmix_03m_trx',
'qtde_Tela_Recarga_',
'qtde_Qr_Code_Accessed_',
'qtde_Debit_Card_-_Added_',
'qtde_FT_Pagar_-_Store_',
'am_min_recp2pbal_01m_trx',
'am_max_recp2pbal_01m_trx',
'am_tot_recp2pbal_01m_trx',
'am_avg_recp2pbal_01m_trx',
'ct_tot_recp2pbal_01m_trx',
'qtde_FT_Pagar_-_Locais_',
'am_min_payp2pbal_01m_trx',
'am_max_payp2pbal_01m_trx',
'am_tot_payp2pbal_01m_trx',
'am_avg_payp2pbal_01m_trx',
'ct_tot_payp2pbal_01m_trx',
'qtde_Transaccaao_P2P_',
'am_min_paypavgood_01m_trx',
'am_max_paypavgood_01m_trx',
'am_tot_paypavgood_01m_trx',
'am_avg_paypavgood_01m_trx',
'ct_tot_paypavgood_01m_trx',
'am_min_payp2pmix_03m_trx',
'am_max_payp2pmix_03m_trx',
'am_tot_payp2pmix_03m_trx',
'am_avg_payp2pmix_03m_trx',
'ct_tot_payp2pmix_03m_trx',
'ct_min_payp2pmix_03m_trx',
'ct_max_payp2pmix_03m_trx',
'ct_avg_payp2pmix_03m_trx',
'qtde_Installments_hint_bubble_',
'qtde_Rendimento_-_Navegou_entre_meses_',
'am_min_paypavinstbill_06m_trx',
'am_max_paypavinstbill_06m_trx',
'am_tot_paypavinstbill_06m_trx',
'am_avg_paypavinstbill_06m_trx',
'ct_tot_paypavinstbill_06m_trx',
'ct_min_paypavinstbill_06m_trx',
'ct_max_paypavinstbill_06m_trx',
'ct_avg_paypavinstbill_06m_trx',
'am_min_paypavbal_01m_trx',
'am_max_paypavbal_01m_trx',
'am_tot_paypavbal_01m_trx',
'am_avg_paypavbal_01m_trx',
'ct_tot_paypavbal_01m_trx',
'am_min_paypavbiz_03m_trx',
'am_max_paypavbiz_03m_trx',
'am_tot_paypavbiz_03m_trx',
'am_avg_paypavbiz_03m_trx',
'ct_tot_paypavbiz_03m_trx',
'ct_min_paypavbiz_03m_trx',
'ct_max_paypavbiz_03m_trx',
'ct_avg_paypavbiz_03m_trx',
'qtde_Permissao_autorizada_-_Camera_',
'qtde_Recebeu_P2P_',
'am_min_recp2pcred_01m_trx',
'am_max_recp2pcred_01m_trx',
'am_tot_recp2pcred_01m_trx',
'am_avg_recp2pcred_01m_trx',
'ct_tot_recp2pcred_01m_trx',
'am_min_payp2pcred_01m_trx',
'am_max_payp2pcred_01m_trx',
'am_tot_payp2pcred_01m_trx',
'am_avg_payp2pcred_01m_trx',
'ct_tot_payp2pcred_01m_trx',
'qtde_Inicio_-_Card_Opportunities_Panned_',
'qtde_MGM_-_Abriu_tela_',
'qtde_Transaccaao_',
'qtde_Onboarding_social_-_naao_tem_amigos_na_agenda_',
'am_min_recp2pmix_06m_trx',
'am_max_recp2pmix_06m_trx',
'am_tot_recp2pmix_06m_trx',
'am_avg_recp2pmix_06m_trx',
'ct_tot_recp2pmix_06m_trx',
'ct_min_recp2pmix_06m_trx',
'ct_max_recp2pmix_06m_trx',
'ct_avg_recp2pmix_06m_trx',
'qtde_Payment_Item_Accessed_',
'qtde_Permissao_autorizada_-_Localizacao_',
'qtde_Inicio_-_List_Pulled_Down_',
'minimo_tempo_permanencia_cartao_horas',
'minimo_tempo_permanencia_cartao_minutos',
'minimo_tempo_permanencia_cartao_segundos',
'maximo_tempo_permanencia_cartao_horas',
'media_tempo_permanencia_cartao_horas',
'maximo_tempo_permanencia_cartao_minutos',
'media_tempo_permanencia_cartao_minutos',
'maximo_tempo_permanencia_cartao_segundos',
'media_tempo_permanencia_cartao_segundos',
'qtde_Visualizacao_de_tela_',
'cadastrou_e_removeu_algum_cartao',
'qtde_cartoes_cadastrados_e_removidos',
'quantidade_tempo_permanencia_cartao_horas',
'quantidade_tempo_permanencia_cartao_minutos',
'quantidade_tempo_permanencia_cartao_segundos',
'am_min_paypavinstbill_12m_trx',
'am_max_paypavinstbill_12m_trx',
'am_tot_paypavinstbill_12m_trx',
'am_avg_paypavinstbill_12m_trx',
'ct_tot_paypavinstbill_12m_trx',
'ct_min_paypavinstbill_12m_trx',
'ct_max_paypavinstbill_12m_trx',
'ct_avg_paypavinstbill_12m_trx',
'qtde_User_Activated_',
'qtde_QR_code_-_acessou_leitor_',
'am_min_payp2pmix_06m_trx',
'am_max_payp2pmix_06m_trx',
'am_tot_payp2pmix_06m_trx',
'am_avg_payp2pmix_06m_trx',
'ct_tot_payp2pmix_06m_trx',
'ct_min_payp2pmix_06m_trx',
'ct_max_payp2pmix_06m_trx',
'ct_avg_payp2pmix_06m_trx',
'qtde_cartoes_da_bandeira_elo',
'am_min_paypavcred_01m_trx',
'am_max_paypavcred_01m_trx',
'am_tot_paypavcred_01m_trx',
'am_avg_paypavcred_01m_trx',
'ct_tot_paypavcred_01m_trx',
'qtde_Credito_-_Documentos_-_Fazer_depois_-_verso_documento_',
'qtde_Coodigo_Promocional%3A_abriu_dialog_',
'am_min_recp2pmix_12m_trx',
'am_max_recp2pmix_12m_trx',
'am_tot_recp2pmix_12m_trx',
'am_avg_recp2pmix_12m_trx',
'ct_tot_recp2pmix_12m_trx',
'ct_min_recp2pmix_12m_trx',
'ct_max_recp2pmix_12m_trx',
'ct_avg_recp2pmix_12m_trx',
'am_min_paypavmix_03m_trx',
'am_max_paypavmix_03m_trx',
'am_tot_paypavmix_03m_trx',
'am_avg_paypavmix_03m_trx',
'ct_tot_paypavmix_03m_trx',
'ct_min_paypavmix_03m_trx',
'ct_max_paypavmix_03m_trx',
'ct_avg_paypavmix_03m_trx',
'quantidade_transacoes_no_dia',
'tpv_minimo_no_dia',
'tpv_maximo_no_dia',
'tpv_medio_no_dia',
'tpv_mediana_no_dia',
'qtde_cartoes_ativos_cadastrados_no_D0',
'qtde_Inseriu_coodigo_promocional_',
'qtde_Rendimento_-_Entrou_na_tela_de_detalhamento_',
'am_min_paypavbiz_06m_trx',
'am_max_paypavbiz_06m_trx',
'am_tot_paypavbiz_06m_trx',
'am_avg_paypavbiz_06m_trx',
'ct_tot_paypavbiz_06m_trx',
'ct_min_paypavbiz_06m_trx',
'ct_max_paypavbiz_06m_trx',
'ct_avg_paypavbiz_06m_trx',
'am_min_recp2p_01m_trx',
'am_max_recp2p_01m_trx',
'am_tot_recp2p_01m_trx',
'am_avg_recp2p_01m_trx',
'ct_tot_recp2p_01m_trx',
'qtde_Inicio_-_Feed_Viewed_',
'qtde_PicPay_Card_Campaign_',
'is_name_matching_name_on_device',
'qtde_MgM_Inviter_',
'qtde_Inicio_-_Pagination_Loaded_',
'am_min_paypavbiz_12m_trx',
'am_max_paypavbiz_12m_trx',
'am_tot_paypavbiz_12m_trx',
'am_avg_paypavbiz_12m_trx',
'ct_tot_paypavbiz_12m_trx',
'ct_min_paypavbiz_12m_trx',
'ct_max_paypavbiz_12m_trx',
'ct_avg_paypavbiz_12m_trx',
'qtde_MgM_Invited_',
'mgm_inviter_consumer_id',
'am_min_payp2p_01m_trx',
'am_max_payp2p_01m_trx',
'am_tot_payp2p_01m_trx',
'am_avg_payp2p_01m_trx',
'ct_tot_payp2p_01m_trx',
'am_min_paypavgood_03m_trx',
'am_max_paypavgood_03m_trx',
'am_tot_paypavgood_03m_trx',
'am_avg_paypavgood_03m_trx',
'ct_tot_paypavgood_03m_trx',
'ct_min_paypavgood_03m_trx',
'ct_max_paypavgood_03m_trx',
'ct_avg_paypavgood_03m_trx',
'am_min_payp2pmix_12m_trx',
'am_max_payp2pmix_12m_trx',
'am_tot_payp2pmix_12m_trx',
'am_avg_payp2pmix_12m_trx',
'ct_tot_payp2pmix_12m_trx',
'ct_min_payp2pmix_12m_trx',
'ct_max_payp2pmix_12m_trx',
'ct_avg_payp2pmix_12m_trx',
'am_min_payp2pbal_03m_trx',
'am_max_payp2pbal_03m_trx',
'am_tot_payp2pbal_03m_trx',
'am_avg_payp2pbal_03m_trx',
'ct_tot_payp2pbal_03m_trx',
'ct_min_payp2pbal_03m_trx',
'ct_max_payp2pbal_03m_trx',
'ct_avg_payp2pbal_03m_trx',
'am_min_recp2pbal_03m_trx',
'am_max_recp2pbal_03m_trx',
'am_tot_recp2pbal_03m_trx',
'am_avg_recp2pbal_03m_trx',
'ct_tot_recp2pbal_03m_trx',
'ct_min_recp2pbal_03m_trx',
'ct_max_recp2pbal_03m_trx',
'ct_avg_recp2pbal_03m_trx',
'qtde_FT_Pagar_-_Principal_',
'qtde_Onboarding_de_primeira_accaao_',
'qtde_FT_Ajustes_',
'qtde_Alterou_login_',
'am_min_recp2pcred_03m_trx',
'am_max_recp2pcred_03m_trx',
'am_tot_recp2pcred_03m_trx',
'am_avg_recp2pcred_03m_trx',
'ct_tot_recp2pcred_03m_trx',
'ct_min_recp2pcred_03m_trx',
'ct_max_recp2pcred_03m_trx',
'ct_avg_recp2pcred_03m_trx',
'qtde_Creedito_-_oferta_carteira_Swype_left_saldo_',
'am_min_paypavmix_06m_trx',
'am_max_paypavmix_06m_trx',
'am_tot_paypavmix_06m_trx',
'am_avg_paypavmix_06m_trx',
'ct_tot_paypavmix_06m_trx',
'ct_min_paypavmix_06m_trx',
'ct_max_paypavmix_06m_trx',
'ct_avg_paypavmix_06m_trx',
'am_min_paypavbill_03m_trx',
'am_max_paypavbill_03m_trx',
'am_tot_paypavbill_03m_trx',
'am_avg_paypavbill_03m_trx',
'ct_tot_paypavbill_03m_trx',
'ct_min_paypavbill_03m_trx',
'ct_max_paypavbill_03m_trx',
'ct_avg_paypavbill_03m_trx',
'qtde_Novo_Cartaao_de_Creedito_',
'qtde_cartoes_cadastrados_no_D0',
'qtde_Permissao_autorizada_-_Contatos_',
'qtde_Onboarding_social_-_Exibiu_invite_de_seguir_',
'qtde_Transaction_',
'am_min_paypav_01m_trx',
'am_max_paypav_01m_trx',
'am_tot_paypav_01m_trx',
'am_avg_paypav_01m_trx',
'ct_tot_paypav_01m_trx',
'qtde_Coodigo_Promocional%3A_Link_de_compartilhamento_',
'qtde_FT_Notificacoes_',
'am_min_paypavbal_03m_trx',
'am_max_paypavbal_03m_trx',
'am_tot_paypavbal_03m_trx',
'am_avg_paypavbal_03m_trx',
'ct_tot_paypavbal_03m_trx',
'ct_min_paypavbal_03m_trx',
'ct_max_paypavbal_03m_trx',
'ct_avg_paypavbal_03m_trx',
'qtde_Tocou_em_coodigo_promocional_')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ** 06. Seleção final de variáveis ------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1. Bases (Dev + Val + Oot) - Em Pandas

# COMMAND ----------

#Ajuste variavel de domínio do e-mail
selecao = selecao2.withColumn('email_ajuste',F.when(F.lower(F.col('email_domain')) == 'gmail', 'gmail')\
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

selecao = selecao.withColumn('device_model_aj', F.lower(F.col('device_model')))
selecao = selecao.withColumn('device_modelo_D0_aj', F.lower(F.col('device_modelo_D0')))
selecao = selecao.withColumn('device_modelo_ref_aj', F.lower(F.col('device_modelo_ref')))

## Ajustes variáveis do modelo do celular
selecao = selecao.withColumn('modelo_celular',F.when(selecao.device_model_aj.contains('samsung'),'samsung')\
                           .otherwise(F.when(selecao.device_model_aj.contains('motorola'),'motorola')\
                               .otherwise(F.when(selecao.device_model_aj.contains('iphone'),'iphone')\
                                   .otherwise(F.when(selecao.device_model_aj.contains('xiaomi'),'xiaomi')\
                                       .otherwise(F.when(selecao.device_model_aj.contains('lg'),'lg')\
                                           .otherwise(F.when(selecao.device_model_aj.contains('asus'),'asus')\
                                               .otherwise(F.when(selecao.device_model_aj.contains('lenovo'),'lenovo')\
                                                   .otherwise(F.when(selecao.device_model_aj.contains('tcl'),'tcl')\
                                                       .otherwise(F.when(selecao.device_model_aj.contains('positivo'),'positivo')\
                                                           .otherwise(F.when(selecao.device_model_aj.contains('multilaser'),'multilaser')\
                                                               .otherwise(F.when(selecao.device_model_aj.contains('ipad'),'ipad')\
                                                                   .otherwise(F.when(selecao.device_model_aj.contains('quantum'),'quantum')\
                                                                       .otherwise(F.when(selecao.device_model_aj.contains('sony'),'sony')\
                                                                           .otherwise(F.when(selecao.device_model_aj.contains('nokia'),'nokia')\
                                                                               .otherwise(F.when(selecao.device_model_aj.contains('huawei'),'huawei')\
                                                                                   .otherwise(F.when(selecao.device_model_aj.contains('oneplus'),'oneplus')\
                                                                                       .otherwise('outros')))))))))))))))))


selecao = selecao.withColumn('modelo_celular_D0',F.when(selecao.device_modelo_D0_aj.contains('samsung'),'samsung')\
                           .otherwise(F.when(selecao.device_modelo_D0_aj.contains('motorola'),'motorola')\
                               .otherwise(F.when(selecao.device_modelo_D0_aj.contains('iphone'),'iphone')\
                                   .otherwise(F.when(selecao.device_modelo_D0_aj.contains('xiaomi'),'xiaomi')\
                                       .otherwise(F.when(selecao.device_modelo_D0_aj.contains('lg'),'lg')\
                                           .otherwise(F.when(selecao.device_modelo_D0_aj.contains('asus'),'asus')\
                                               .otherwise(F.when(selecao.device_modelo_D0_aj.contains('lenovo'),'lenovo')\
                                                   .otherwise(F.when(selecao.device_modelo_D0_aj.contains('tcl'),'tcl')\
                                                       .otherwise(F.when(selecao.device_modelo_D0_aj.contains('positivo'),'positivo')\
                                                           .otherwise(F.when(selecao.device_modelo_D0_aj.contains('multilaser'),'multilaser')\
                                                               .otherwise(F.when(selecao.device_modelo_D0_aj.contains('ipad'),'ipad')\
                                                                   .otherwise(F.when(selecao.device_modelo_D0_aj.contains('quantum'),'quantum')\
                                                                       .otherwise(F.when(selecao.device_modelo_D0_aj.contains('sony'),'sony')\
                                                                           .otherwise(F.when(selecao.device_modelo_D0_aj.contains('nokia'),'nokia')\
                                                                               .otherwise(F.when(selecao.device_modelo_D0_aj.contains('huawei'),'huawei')\
                                                                                   .otherwise(F.when(selecao.device_modelo_D0_aj.contains('oneplus'),'oneplus')\
                                                                                       .otherwise('outros')))))))))))))))))


selecao = selecao.withColumn('modelo_celular_ref',F.when(selecao.device_modelo_ref_aj.contains('samsung'),'samsung')\
                           .otherwise(F.when(selecao.device_modelo_ref_aj.contains('motorola'),'motorola')\
                               .otherwise(F.when(selecao.device_modelo_ref_aj.contains('iphone'),'iphone')\
                                   .otherwise(F.when(selecao.device_modelo_ref_aj.contains('xiaomi'),'xiaomi')\
                                       .otherwise(F.when(selecao.device_modelo_ref_aj.contains('lg'),'lg')\
                                           .otherwise(F.when(selecao.device_modelo_ref_aj.contains('asus'),'asus')\
                                               .otherwise(F.when(selecao.device_modelo_ref_aj.contains('lenovo'),'lenovo')\
                                                   .otherwise(F.when(selecao.device_modelo_ref_aj.contains('tcl'),'tcl')\
                                                       .otherwise(F.when(selecao.device_modelo_ref_aj.contains('positivo'),'positivo')\
                                                           .otherwise(F.when(selecao.device_modelo_ref_aj.contains('multilaser'),'multilaser')\
                                                               .otherwise(F.when(selecao.device_modelo_ref_aj.contains('ipad'),'ipad')\
                                                                   .otherwise(F.when(selecao.device_modelo_ref_aj.contains('quantum'),'quantum')\
                                                                       .otherwise(F.when(selecao.device_modelo_ref_aj.contains('sony'),'sony')\
                                                                           .otherwise(F.when(selecao.device_modelo_ref_aj.contains('nokia'),'nokia')\
                                                                               .otherwise(F.when(selecao.device_modelo_ref_aj.contains('huawei'),'huawei')\
                                                                                   .otherwise(F.when(selecao.device_modelo_ref_aj.contains('oneplus'),'oneplus')\
                                                                                       .otherwise('outros')))))))))))))))))

# Removendo as variáveis que foram substituídas pelas criadas no item anterior
selecao3 = selecao.drop('email_domain','device_model_aj','device_model','device_modelo_D0_aj','device_modelo_D0','device_modelo_ref_aj', 'device_modelo_ref')

# COMMAND ----------

## Segmentação do grupo que será modelado
tmp = selecao3.toPandas()
 
# Conversões Strings para numéricos 
tmp['am_min_pay_01m_trx']         = tmp['am_min_pay_01m_trx'].astype(float)
tmp['am_max_pay_01m_trx']         = tmp['am_max_pay_01m_trx'].astype(float)
tmp['am_tot_pay_01m_trx']         = tmp['am_tot_pay_01m_trx'].astype(float)
tmp['am_avg_pay_01m_trx']         = tmp['am_avg_pay_01m_trx'].astype(float)
                              
tmp['am_min_pay_03m_trx']         = tmp['am_min_pay_03m_trx'].astype(float)
tmp['am_max_pay_03m_trx']         = tmp['am_max_pay_03m_trx'].astype(float)
tmp['am_tot_pay_03m_trx']         = tmp['am_tot_pay_03m_trx'].astype(float)
tmp['am_avg_pay_03m_trx']         = tmp['am_avg_pay_03m_trx'].astype(float)

tmp['am_min_pay_06m_trx']         = tmp['am_min_pay_06m_trx'].astype(float)
tmp['am_max_pay_06m_trx']         = tmp['am_max_pay_06m_trx'].astype(float)
tmp['am_tot_pay_06m_trx']         = tmp['am_tot_pay_06m_trx'].astype(float)
tmp['am_avg_pay_06m_trx']         = tmp['am_avg_pay_06m_trx'].astype(float)

tmp['am_min_pay_12m_trx']         = tmp['am_min_pay_12m_trx'].astype(float)
tmp['am_max_pay_12m_trx']         = tmp['am_max_pay_12m_trx'].astype(float)
tmp['am_tot_pay_12m_trx']         = tmp['am_tot_pay_12m_trx'].astype(float)
tmp['am_avg_pay_12m_trx']         = tmp['am_avg_pay_12m_trx'].astype(float)

tmp['am_min_payp2p_03m_trx']      = tmp['am_min_payp2p_03m_trx'].astype(float)
tmp['am_max_payp2p_03m_trx']      = tmp['am_max_payp2p_03m_trx'].astype(float)
tmp['am_tot_payp2p_03m_trx']      = tmp['am_tot_payp2p_03m_trx'].astype(float)
tmp['am_avg_payp2p_03m_trx']      = tmp['am_avg_payp2p_03m_trx'].astype(float)

tmp['am_min_payp2p_06m_trx']      = tmp['am_min_payp2p_06m_trx'].astype(float)
tmp['am_max_payp2p_06m_trx']      = tmp['am_max_payp2p_06m_trx'].astype(float)
tmp['am_tot_payp2p_06m_trx']      = tmp['am_tot_payp2p_06m_trx'].astype(float)
tmp['am_avg_payp2p_06m_trx']      = tmp['am_avg_payp2p_06m_trx'].astype(float)

tmp['am_min_payp2p_12m_trx']      = tmp['am_min_payp2p_12m_trx'].astype(float)
tmp['am_max_payp2p_12m_trx']      = tmp['am_max_payp2p_12m_trx'].astype(float)
tmp['am_tot_payp2p_12m_trx']      = tmp['am_tot_payp2p_12m_trx'].astype(float)
tmp['am_avg_payp2p_12m_trx']      = tmp['am_avg_payp2p_12m_trx'].astype(float)

tmp['am_min_paypav_03m_trx']      = tmp['am_min_paypav_03m_trx'].astype(float)
tmp['am_max_paypav_03m_trx']      = tmp['am_max_paypav_03m_trx'].astype(float)
tmp['am_tot_paypav_03m_trx']      = tmp['am_tot_paypav_03m_trx'].astype(float)
tmp['am_avg_paypav_03m_trx']      = tmp['am_avg_paypav_03m_trx'].astype(float)

tmp['am_min_paypav_06m_trx']      = tmp['am_min_paypav_06m_trx'].astype(float)
tmp['am_max_paypav_06m_trx']      = tmp['am_max_paypav_06m_trx'].astype(float)
tmp['am_tot_paypav_06m_trx']      = tmp['am_tot_paypav_06m_trx'].astype(float)
tmp['am_avg_paypav_06m_trx']      = tmp['am_avg_paypav_06m_trx'].astype(float)

tmp['am_min_paypav_12m_trx']      = tmp['am_min_paypav_12m_trx'].astype(float)
tmp['am_max_paypav_12m_trx']      = tmp['am_max_paypav_12m_trx'].astype(float)
tmp['am_tot_paypav_12m_trx']      = tmp['am_tot_paypav_12m_trx'].astype(float)
tmp['am_avg_paypav_12m_trx']      = tmp['am_avg_paypav_12m_trx'].astype(float)

tmp['am_min_payp2pcred_03m_trx']      = tmp['am_min_payp2pcred_03m_trx'].astype(float)
tmp['am_max_payp2pcred_03m_trx']      = tmp['am_max_payp2pcred_03m_trx'].astype(float)
tmp['am_tot_payp2pcred_03m_trx']      = tmp['am_tot_payp2pcred_03m_trx'].astype(float)
tmp['am_avg_payp2pcred_03m_trx']      = tmp['am_avg_payp2pcred_03m_trx'].astype(float)

tmp['am_min_payp2pcred_06m_trx']      = tmp['am_min_payp2pcred_06m_trx'].astype(float)
tmp['am_max_payp2pcred_06m_trx']      = tmp['am_max_payp2pcred_06m_trx'].astype(float)
tmp['am_tot_payp2pcred_06m_trx']      = tmp['am_tot_payp2pcred_06m_trx'].astype(float)
tmp['am_avg_payp2pcred_06m_trx']      = tmp['am_avg_payp2pcred_06m_trx'].astype(float)

tmp['am_min_payp2pcred_12m_trx']      = tmp['am_min_payp2pcred_12m_trx'].astype(float)
tmp['am_max_payp2pcred_12m_trx']      = tmp['am_max_payp2pcred_12m_trx'].astype(float)
tmp['am_tot_payp2pcred_12m_trx']      = tmp['am_tot_payp2pcred_12m_trx'].astype(float)
tmp['am_avg_payp2pcred_12m_trx']      = tmp['am_avg_payp2pcred_12m_trx'].astype(float)

tmp['am_min_payp2pbal_06m_trx']      = tmp['am_min_payp2pbal_06m_trx'].astype(float)
tmp['am_max_payp2pbal_06m_trx']      = tmp['am_max_payp2pbal_06m_trx'].astype(float)
tmp['am_tot_payp2pbal_06m_trx']      = tmp['am_tot_payp2pbal_06m_trx'].astype(float)
tmp['am_avg_payp2pbal_06m_trx']      = tmp['am_avg_payp2pbal_06m_trx'].astype(float)

tmp['am_min_payp2pbal_12m_trx']      = tmp['am_min_payp2pbal_12m_trx'].astype(float)
tmp['am_max_payp2pbal_12m_trx']      = tmp['am_max_payp2pbal_12m_trx'].astype(float)
tmp['am_tot_payp2pbal_12m_trx']      = tmp['am_tot_payp2pbal_12m_trx'].astype(float)
tmp['am_avg_payp2pbal_12m_trx']      = tmp['am_avg_payp2pbal_12m_trx'].astype(float)

tmp['am_min_paypavcred_03m_trx']      = tmp['am_min_paypavcred_03m_trx'].astype(float)
tmp['am_max_paypavcred_03m_trx']      = tmp['am_max_paypavcred_03m_trx'].astype(float)
tmp['am_tot_paypavcred_03m_trx']      = tmp['am_tot_paypavcred_03m_trx'].astype(float)
tmp['am_avg_paypavcred_03m_trx']      = tmp['am_avg_paypavcred_03m_trx'].astype(float)

tmp['am_min_paypavcred_06m_trx']      = tmp['am_min_paypavcred_06m_trx'].astype(float)
tmp['am_max_paypavcred_06m_trx']      = tmp['am_max_paypavcred_06m_trx'].astype(float)
tmp['am_tot_paypavcred_06m_trx']      = tmp['am_tot_paypavcred_06m_trx'].astype(float)
tmp['am_avg_paypavcred_06m_trx']      = tmp['am_avg_paypavcred_06m_trx'].astype(float)

tmp['am_min_paypavcred_12m_trx']      = tmp['am_min_paypavcred_12m_trx'].astype(float)
tmp['am_max_paypavcred_12m_trx']      = tmp['am_max_paypavcred_12m_trx'].astype(float)
tmp['am_tot_paypavcred_12m_trx']      = tmp['am_tot_paypavcred_12m_trx'].astype(float)
tmp['am_avg_paypavcred_12m_trx']      = tmp['am_avg_paypavcred_12m_trx'].astype(float)

tmp['am_min_paypavbal_06m_trx']      = tmp['am_min_paypavbal_06m_trx'].astype(float)
tmp['am_max_paypavbal_06m_trx']      = tmp['am_max_paypavbal_06m_trx'].astype(float)
tmp['am_tot_paypavbal_06m_trx']      = tmp['am_tot_paypavbal_06m_trx'].astype(float)
tmp['am_avg_paypavbal_06m_trx']      = tmp['am_avg_paypavbal_06m_trx'].astype(float)

tmp['am_min_paypavbal_12m_trx']      = tmp['am_min_paypavbal_12m_trx'].astype(float)
tmp['am_max_paypavbal_12m_trx']      = tmp['am_max_paypavbal_12m_trx'].astype(float)
tmp['am_tot_paypavbal_12m_trx']      = tmp['am_tot_paypavbal_12m_trx'].astype(float)
tmp['am_avg_paypavbal_12m_trx']      = tmp['am_avg_paypavbal_12m_trx'].astype(float)

tmp['am_min_paypavmix_12m_trx']      = tmp['am_min_paypavmix_12m_trx'].astype(float)
tmp['am_max_paypavmix_12m_trx']      = tmp['am_max_paypavmix_12m_trx'].astype(float)
tmp['am_tot_paypavmix_12m_trx']      = tmp['am_tot_paypavmix_12m_trx'].astype(float)
tmp['am_avg_paypavmix_12m_trx']      = tmp['am_avg_paypavmix_12m_trx'].astype(float)

tmp['am_min_paypavbill_06m_trx']      = tmp['am_min_paypavbill_06m_trx'].astype(float)
tmp['am_max_paypavbill_06m_trx']      = tmp['am_max_paypavbill_06m_trx'].astype(float)
tmp['am_tot_paypavbill_06m_trx']      = tmp['am_tot_paypavbill_06m_trx'].astype(float)
tmp['am_avg_paypavbill_06m_trx']      = tmp['am_avg_paypavbill_06m_trx'].astype(float)

tmp['am_min_paypavbill_12m_trx']      = tmp['am_min_paypavbill_12m_trx'].astype(float)
tmp['am_max_paypavbill_12m_trx']      = tmp['am_max_paypavbill_12m_trx'].astype(float)
tmp['am_tot_paypavbill_12m_trx']      = tmp['am_tot_paypavbill_12m_trx'].astype(float)
tmp['am_avg_paypavbill_12m_trx']      = tmp['am_avg_paypavbill_12m_trx'].astype(float)

tmp['am_min_paypavgood_06m_trx']      = tmp['am_min_paypavgood_06m_trx'].astype(float)
tmp['am_max_paypavgood_06m_trx']      = tmp['am_max_paypavgood_06m_trx'].astype(float)
tmp['am_tot_paypavgood_06m_trx']      = tmp['am_tot_paypavgood_06m_trx'].astype(float)
tmp['am_avg_paypavgood_06m_trx']      = tmp['am_avg_paypavgood_06m_trx'].astype(float)

tmp['am_min_paypavgood_12m_trx']      = tmp['am_min_paypavgood_12m_trx'].astype(float)
tmp['am_max_paypavgood_12m_trx']      = tmp['am_max_paypavgood_12m_trx'].astype(float)
tmp['am_tot_paypavgood_12m_trx']      = tmp['am_tot_paypavgood_12m_trx'].astype(float)
tmp['am_avg_paypavgood_12m_trx']      = tmp['am_avg_paypavgood_12m_trx'].astype(float)

tmp['am_min_recp2p_03m_trx']      = tmp['am_min_recp2p_03m_trx'].astype(float)
tmp['am_max_recp2p_03m_trx']      = tmp['am_max_recp2p_03m_trx'].astype(float)
tmp['am_tot_recp2p_03m_trx']      = tmp['am_tot_recp2p_03m_trx'].astype(float)
tmp['am_avg_recp2p_03m_trx']      = tmp['am_avg_recp2p_03m_trx'].astype(float)

tmp['am_min_recp2p_06m_trx']      = tmp['am_min_recp2p_06m_trx'].astype(float)
tmp['am_max_recp2p_06m_trx']      = tmp['am_max_recp2p_06m_trx'].astype(float)
tmp['am_tot_recp2p_06m_trx']      = tmp['am_tot_recp2p_06m_trx'].astype(float)
tmp['am_avg_recp2p_06m_trx']      = tmp['am_avg_recp2p_06m_trx'].astype(float)

tmp['am_min_recp2p_12m_trx']      = tmp['am_min_recp2p_12m_trx'].astype(float)
tmp['am_max_recp2p_12m_trx']      = tmp['am_max_recp2p_12m_trx'].astype(float)
tmp['am_tot_recp2p_12m_trx']      = tmp['am_tot_recp2p_12m_trx'].astype(float)
tmp['am_avg_recp2p_12m_trx']      = tmp['am_avg_recp2p_12m_trx'].astype(float)

tmp['am_min_recp2pcred_06m_trx']      = tmp['am_min_recp2pcred_06m_trx'].astype(float)
tmp['am_max_recp2pcred_06m_trx']      = tmp['am_max_recp2pcred_06m_trx'].astype(float)
tmp['am_tot_recp2pcred_06m_trx']      = tmp['am_tot_recp2pcred_06m_trx'].astype(float)
tmp['am_avg_recp2pcred_06m_trx']      = tmp['am_avg_recp2pcred_06m_trx'].astype(float)

tmp['am_min_recp2pcred_12m_trx']      = tmp['am_min_recp2pcred_12m_trx'].astype(float)
tmp['am_max_recp2pcred_12m_trx']      = tmp['am_max_recp2pcred_12m_trx'].astype(float)
tmp['am_tot_recp2pcred_12m_trx']      = tmp['am_tot_recp2pcred_12m_trx'].astype(float)
tmp['am_avg_recp2pcred_12m_trx']      = tmp['am_avg_recp2pcred_12m_trx'].astype(float)

tmp['am_min_recp2pbal_06m_trx']      = tmp['am_min_recp2pbal_06m_trx'].astype(float)
tmp['am_max_recp2pbal_06m_trx']      = tmp['am_max_recp2pbal_06m_trx'].astype(float)
tmp['am_tot_recp2pbal_06m_trx']      = tmp['am_tot_recp2pbal_06m_trx'].astype(float)
tmp['am_avg_recp2pbal_06m_trx']      = tmp['am_avg_recp2pbal_06m_trx'].astype(float)

tmp['am_min_recp2pbal_12m_trx']      = tmp['am_min_recp2pbal_12m_trx'].astype(float)
tmp['am_max_recp2pbal_12m_trx']      = tmp['am_max_recp2pbal_12m_trx'].astype(float)
tmp['am_tot_recp2pbal_12m_trx']      = tmp['am_tot_recp2pbal_12m_trx'].astype(float)
tmp['am_avg_recp2pbal_12m_trx']      = tmp['am_avg_recp2pbal_12m_trx'].astype(float)
 
# Variáveis númericas
  #Substituir null por 0 (Zero)
numericas   = tmp.select_dtypes(include=['int','float']).columns.tolist()
 
for col in numericas:
  tmp[col] = pd.np.where(tmp[col].isin([np.inf, -np.inf]), np.nan, tmp[col])
  tmp[col] = tmp[col].fillna(tmp[col].mean())
  tmp[col] = tmp[col].fillna(0)
  
# Variáveis categóricas 
  #Substituir null por "outros"
categoricas = tmp.select_dtypes(['object']).columns.tolist()
 
for col in categoricas:
  tmp[col] = pd.np.where(tmp[col].isin([np.inf, -np.inf]), np.nan, tmp[col])
  tmp[col] = tmp[col].fillna("others")
  tmp[col] = tmp[col].replace(True, 'True')
  tmp[col] = tmp[col].replace(False, 'False')
  

# Separação em Desenvolvimento (100%) e Out-of-Time
base_tot = tmp.where(tmp['ref_portfolio'] != '2020-04')
base_tot = base_tot.dropna(how='all') 

base_oot = tmp.where(tmp['ref_portfolio'] == '2020-04')
base_oot = base_oot.dropna(how='all') 


# Separação do período de desenvolvimento
  # DEV: 80%
  # VAL: 20%
  
 ## Removendo variáveis identificadoras e caracter (como transformar em DUMMY?)
features = [i for i in base_tot.columns if i not in ['cpf','consumer_id','ref_portfolio','ref_date','SConcEver60dP6_100','performance']]
resposta = 'SConcEver60dP6_100'
  
## Segmentação do grupo que será modelado
tot = base_tot
               
## Criando dataset temporário para testes - Somente com variáveis numéricas")
x, y = tot[features], tot[resposta]


## Separação da base de dados em treino e teste (desenvolvimento e validação)
  ## Necessário verificar qual sera o percentual de separação, gosto de trabalhar com 80/20 ou 70/30
x_base_dev, x_base_val, y_base_dev, y_base_val = train_test_split(x, y, train_size = 0.8, random_state = 123)

base_dev = pd.concat([x_base_dev, y_base_dev], axis = 1)
base_val = pd.concat([x_base_val, y_base_val], axis = 1)
base_dev.reset_index(drop = True, inplace = True)
base_val.reset_index(drop = True, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2. Teste de Correlação

# COMMAND ----------

df_pbcorr, columns_remove_pbcorr = point_biserial(base_dev, resposta, num_columns = None, significancia=0.05)

# COMMAND ----------

columns_remove_pbcorr

# COMMAND ----------

df_pbcorr

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3. Método Boruta

# COMMAND ----------

# definir o objeto
boruta_selector = Boruta(n_iter=100)
X=base_dev.drop(resposta, axis=1)
y=base_dev[resposta]

boruta_selector.fit(X, y, cat_columns=True, num_columns=True)
bool_decision_boruta = [True if col in boruta_selector._columns_remove_boruta else False for col in base_dev.drop(resposta, axis=1)]

# COMMAND ----------

boruta_selector._columns_remove_boruta

# COMMAND ----------

boruta_selector._best_features

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 6.4. Chi2
# MAGIC 
# MAGIC Serão removidas as variáveis cujo percentual de missing seja > 95%

# COMMAND ----------

colunas_categoricas = base_dev.select_dtypes(include=['object']).columns.tolist()
chi2_df, cols_drop_chi2, logs = chi_squared(base_dev, y=resposta, cat_columns = colunas_categoricas, significance=0.05)

# COMMAND ----------

chi2_df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 6.5. IV - Information Value

# COMMAND ----------

# MAGIC %md
# MAGIC #### Categóricas

# COMMAND ----------

vars_iv = colunas_categoricas = base_dev.select_dtypes(include=['object']).columns.tolist()

list_iv = []
for i in vars_iv:
  iv = get_IV(base_dev, i, resposta, cat=True, q=10)
  list_iv.append(iv)
  
data = {'Variáveis': vars_iv,
        'IV': list_iv
       }

df_iv_cat = pd.DataFrame (data, columns = ['Variáveis','IV'])

display(df_iv_cat)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Númericas

# COMMAND ----------

vars_iv = colunas_categoricas = base_dev.select_dtypes(exclude=['object']).columns.tolist()
vars_iv.remove('SConcEver60dP6_100')

list_iv = []
for i in vars_iv:
  iv = get_IV(base_dev, i, resposta, cat=False, q=10)
  list_iv.append(iv)
  
data = {'Variáveis': vars_iv,
        'IV': list_iv
       }

df_iv_num = pd.DataFrame (data, columns = ['Variáveis','IV'])

display(df_iv_num)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 6.6. Seleção Final

# COMMAND ----------

# Removendo as variáveis que foram substituídas pelas criadas no item anterior
selecao4 = selecao3.drop(
            
            #Eliminadas por Correção
            'is_install_before_register',
            'qtde_Boarding_Pass_',
            'qtde_User_Login_',
            'qtde_FT_Carteira_',
            'qtde_Inseriu_coodigo_promocional_-_erro_',
            'qtde_FT_Inicio_',
            'qtde_total_',
            'am_min_pay_01m_trx',
            'am_max_pay_01m_trx',
            'am_avg_pay_01m_trx',
            'am_min_pay_03m_trx',
            'am_max_pay_03m_trx',
            'am_tot_pay_03m_trx',
            'am_avg_pay_03m_trx',
            'am_min_pay_06m_trx',
            'am_min_pay_12m_trx',
            'am_min_payp2p_06m_trx',
            'am_max_payp2p_06m_trx',
            'am_avg_payp2p_06m_trx',
            'am_min_payp2p_12m_trx',
            'am_min_paypav_03m_trx',
            'am_avg_paypav_03m_trx',
            'am_min_paypav_12m_trx',
            'am_min_payp2pcred_06m_trx',
            'am_max_payp2pcred_06m_trx',
            'am_tot_payp2pcred_06m_trx',
            'am_avg_payp2pcred_06m_trx',
            'am_min_payp2pcred_12m_trx',
            'ct_max_paypavcred_12m_trx',
            'am_tot_paypavbill_06m_trx',
            'am_avg_paypavbill_06m_trx',
            'am_min_paypavbill_12m_trx',
            'am_min_recp2p_03m_trx',
            'am_min_recp2p_06m_trx',
            'am_max_recp2p_06m_trx',
            'am_avg_recp2p_06m_trx',
            'am_min_recp2p_12m_trx',
            'am_tot_recp2p_12m_trx',
            'am_avg_recp2p_12m_trx',
            'am_min_recp2pcred_06m_trx',
            'am_max_recp2pcred_06m_trx',
            'am_tot_recp2pcred_06m_trx',
            'am_avg_recp2pcred_06m_trx',
            'am_min_recp2pcred_12m_trx',
            'am_tot_recp2pcred_12m_trx',
            'am_avg_recp2pcred_12m_trx',
            'am_min_recp2pbal_06m_trx',
            'am_min_recp2pbal_12m_trx',
            'am_max_recp2pbal_12m_trx',
            'am_avg_recp2pbal_12m_trx',
  
            #Eliminadas por Boruta
            'tempo_registro_em_anos',
            'ct_avg_pay_12m_trx',
            'am_tot_payp2p_03m_trx',
            'ct_max_payp2p_03m_trx',
            'ct_avg_payp2p_03m_trx',
            'am_tot_payp2p_06m_trx',
            'ct_avg_payp2pcred_06m_trx',
            'am_max_payp2pcred_12m_trx',
            'ct_max_payp2pcred_12m_trx',
            'ct_avg_payp2pcred_12m_trx',
            'am_max_payp2pbal_06m_trx',
            'ct_avg_payp2pbal_06m_trx',
            'am_tot_payp2pbal_12m_trx',
            'ct_avg_payp2pbal_12m_trx',
            'am_min_paypavcred_03m_trx',
            'am_tot_paypavcred_06m_trx',
            'am_avg_paypavcred_06m_trx',
            'ct_tot_paypavcred_06m_trx',
            'ct_max_paypavcred_06m_trx',
            'ct_avg_paypavcred_06m_trx',
            'am_min_paypavcred_12m_trx',
            'am_max_paypavcred_12m_trx',
            'ct_avg_paypavcred_12m_trx',
            'am_min_paypavbal_06m_trx',
            'am_max_paypavbal_06m_trx',
            'ct_max_paypavbal_06m_trx',
            'ct_avg_paypavbal_06m_trx',
            'am_min_paypavbal_12m_trx',
            'am_max_paypavbal_12m_trx',
            'am_tot_paypavbal_12m_trx',
            'am_avg_paypavbal_12m_trx',
            'ct_tot_paypavbal_12m_trx',
            'ct_max_paypavbal_12m_trx',
            'ct_avg_paypavbal_12m_trx',
            'am_min_paypavmix_12m_trx',
            'am_max_paypavmix_12m_trx',
            'ct_tot_paypavmix_12m_trx',
            'ct_max_paypavmix_12m_trx',
            'ct_avg_paypavmix_12m_trx',
            'am_min_paypavbill_06m_trx',
            'am_max_paypavbill_06m_trx',
            'ct_min_paypavbill_06m_trx',
            'ct_tot_paypavgood_12m_trx',
            'am_tot_recp2p_06m_trx',
            'ct_tot_recp2p_06m_trx',
            'ct_min_recp2p_06m_trx',
            'ct_tot_recp2pcred_06m_trx',
            'ct_tot_recp2pcred_12m_trx',
            'ct_min_recp2pcred_12m_trx',
            'ct_max_recp2pcred_12m_trx',
            'ct_avg_recp2pcred_12m_trx',
            'am_max_recp2pbal_06m_trx',
            'am_tot_recp2pbal_06m_trx',
            'am_avg_recp2pbal_06m_trx',
            'ct_tot_recp2pbal_06m_trx',
            'ct_min_recp2pbal_06m_trx',
            'ct_max_recp2pbal_06m_trx',
            'ct_avg_recp2pbal_06m_trx',
            'am_tot_recp2pbal_12m_trx',            
  
  
            #Eliminadas pelo teste de Chi2
            'device_email_domain',
            'ddd',
            'latest_device_os',
            'register_device_os',
            'origin',
            'instituicao_cartao_D0',
            'bandeira_cartao_D0',
            'device_D0',
            'modelo_celular',
            'modelo_celular_D0',
  
            #Eliminadas por IV
            'ct_min_payp2p_06m_trx',
            'ct_min_recp2pcred_06m_trx',
            'am_avg_payp2pbal_06m_trx',
            'ct_max_recp2p_03m_trx',
            'ct_min_recp2p_03m_trx',
            'ct_tot_recp2pbal_12m_trx',
            'ct_avg_recp2pbal_12m_trx',
            'ct_max_recp2p_06m_trx',
            'ct_max_recp2p_12m_trx',
            'ct_max_paypavgood_12m_trx',
            'ct_max_recp2pcred_06m_trx',

            #Removendo variáveis do Mixpanel
            'qtde_TLS_',
            'qtde_Abriu_App_-_BK_',
            'qtde_Cadastrou_username_',
            'qtde_Step_do_cadastro_de_cartao_de_credito_',
            'qtde_Usuaaario_Cadastrado_',
          
            #Variáveis mais aderentes com perfil de fraude e nao para crédito
            'is_email_confirmed',
            'is_name_matching_email',
            'is_name_matching_name_on_cpf',
            'is_device_shared',
            'is_name_fully_matching_name_on_cpf',
            'is_created_on_known_device',
            'is_location_enabled',
            'is_cpf_valid',
            'is_birthdate_valid',
            'is_phone_matching_bureau',
            'is_name_cpf_matching_any_email_device',
            'is_name_matching_any_email_device',
            'is_phone_region_matching_bureau_address',
            'is_area_code_matching_bureau'
           
)

# COMMAND ----------

selecao5 = selecao4.withColumn('instituicao_cartao_ref_aj', F.lower(F.col('instituicao_cartao_ref')))

selecao6 = selecao5.withColumn('agrupamento_instituicao_cartao_ref',\
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

selecao7 = selecao6.drop('instituicao_cartao_ref_aj', 'instituicao_cartao_ref')

# COMMAND ----------

display(selecao7.groupBy('agrupamento_instituicao_cartao_ref').count().orderBy(F.desc('count')))

# COMMAND ----------

display(selecao7.groupBy('agrupamento_instituicao_cartao_ref','SConcEver60dP6_100').count().orderBy('agrupamento_instituicao_cartao_ref','SConcEver60dP6_100'))

# COMMAND ----------

display(selecao7)

# COMMAND ----------

selecao7.write.parquet("s3://picpay-datalake-sandbox/jeobara.zacheski/score_pf_cs/selecao_final_cs_carteira/", mode='overwrite')
