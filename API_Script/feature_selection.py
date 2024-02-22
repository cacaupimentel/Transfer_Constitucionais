#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Import
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('Importing libraries...')
print("--------------------------------------------------------------------------")
import os
import math
import time
import datetime
from datetime import date, timedelta, datetime
#import Orange
import collections
import patsy

# numpy
import numpy as np
print('numpy: %s' % np.__version__)

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
print('matplotlib: %s' % mpl.__version__)
# pandas
import pandas as pd
print('pandas: %s' % pd.__version__)

import IPython
import IPython.display

import seaborn as sns
print('seaborn: %s' % sns.__version__)

# statsmodels
import statsmodels.api as sm
print('statsmodels: %s' % sm.__version__)
#import pandas.util.testing as tm

import pandas.testing as tm

# scipy
import scipy
from scipy import stats
print('scipy: %s' % scipy.__version__)

# Pacotes para testes estatísticos de QQ e Shapito Wilk
from matplotlib import pylab
from pylab import *
#import pylab

# importatando os métodos de Feature Selection - Key Best, Percentile, F Regression(para dataset de regressão)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

# Importando a função Feaure selection
from sklearn.feature_selection import SelectFromModel

# Dividir os bancos
from sklearn.model_selection import train_test_split

# Metricas do SkLearning MAE
from sklearn.metrics import mean_absolute_error

# Metricas do SkLearning MSE
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

# Importando modelos:  Regressão Linear Simples=LinearRegression, Regressão com L2=Ridge e Regressão com Lasso
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from xgboost import XGBRegressor
#from sklearn.inspection import permutation_importance

# RandomForest Regressor
from sklearn.ensemble import RandomForestRegressor

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# bibliotecas importadas
from pca import pca 
import pylab as pl
import random as rd

# bibliotecas para o banco de dados SQLite
import os
import sqlite3

# Importa funções auxiliares e específicas de Machine Learning
import functions_ML  as ml


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# BACO DE DADOS
#-----------------------------------------------------------------------------------------------------------------------------------------------------

# Connecting to DB
# Abre o banco
connection, cursor = ml.checaArquivo('database.db')


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# DATASET
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('Open and treat dataset')
print("--------------------------------------------------------------------------")


### Abrindo arquivo csv
#datarec = pd.read_csv('dataset/02_CSVdatasetMes_original.csv', sep =';')
### Dataset de Receitas Orcamentarias no Google Drive
#datarec = pd.read_csv('dataset/CSVdataset.csv', sep =';')
datarec = pd.read_csv('dataset/03_CSVdatasetMes_original.csv', sep =';')

# Tranforma o index  de RangeIndex para DatetimeIndex


datarec.Data_Receita = pd.to_datetime(datarec.Data_Receita)
datarec = datarec.set_index('Data_Receita')
datarec.sort_index(inplace=True)

datarec = datarec.replace({',': '.'}, regex=True)

# Substirui a vérgula por ponto
datarec = datarec.replace({',': '.'}, regex=True)

# Ajuste do tipo dos dados para leitura em csv
for col in datarec.columns:
    if (col != 'RECEITA' and col != 'UF'):
        datarec[col] = pd.to_numeric(datarec[col],errors = 'coerce')


# Identifica as receitas que estão no arquivo para seleção das features
itensreceitas = pd.unique(datarec['RECEITA'])
itensreceitas


# Receitas que serão trabalhadas por quantidade de amostras superiores a 30
itensreceitas = np.array(['CFEM', 'FEP', 'FPM', 'ICMS', 'IPI', 'IPVA'], dtype=object)

# Apaga dados antriores
sql = "delete from Receitas"
res = cursor.execute(sql).fetchall()

connection.commit()
# ------> GRavar no banco estes dados
for r in itensreceitas:
    _nome = str(r)
    print("Pesquisando receita: ", _nome)
    sql = "select * from Receitas where id ='" + _nome  + "' "
    reg_count = cursor.execute(sql).fetchall()
    if len(reg_count) == 0:
        #Atualiza
        #print("Atualiza Registro")
        sql = "insert into Receitas (id, nome) values ("
        sql = sql + "'" + _nome + "', "
        sql = sql + "'" + _nome + "' )"
        print(sql)
        #print(sql)
        res = cursor.execute(sql).fetchall()
        # Atualiza banco
        connection.commit()


# Apaga dados antriores
sql = "delete from Capitals"
res = cursor.execute(sql).fetchall()
# Atualiza banco
connection.commit()

# Identifica as Capitais do estudo
itenscapitais = pd.unique(datarec['UF'])
# ------> GRavar no banco estes dados
for r in itenscapitais:
    _nome = str(r)
    print("Pesquisando Capitals: ", _nome)
    sql = "select * from Capitals where uf = '" + _nome  + "' "
    reg_count = cursor.execute(sql).fetchall()
    if len(reg_count) == 0:
        #Atualiza
        #print("Atualiza Registro")
        sql = "insert into Capitals (uf, nome) values ("
        sql = sql + "'" + _nome + "', "
        sql = sql + "'" + ml.Ufs(_nome) + "' )"
        print(sql)
        #print(sql)
        res = cursor.execute(sql).fetchall()
        # Atualiza banco
        connection.commit()



#-----------------------------------------------------------------------------------------------------------------------------------------------------
# NORMALIZING DATASET
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('Normalizando dataset')
print("--------------------------------------------------------------------------")
# Veriicações
# verificando se a coluna possui mais de 70% de valores zerados para apagar
for col in datarec.columns:
  repetidos = collections.Counter(datarec[col])
  qnt = repetidos[0]/datarec[col].count()*100
  if qnt>70:
    print(f'A coluna {col} possui {qnt}% valores ZEROS')
    # apaga a coluna
    datarec = datarec.drop(columns=[col])

# Verificando valores nulos
for col in datarec.columns:
    if datarec[col].isna().values.any():
        qnt = datarec[col].isna().sum()
        print(f'A coluna {col} possui {qnt} valores nulos')
        # Repetindo os valores anteriores para os valores nulos
        datarec[col].fillna( method ='ffill', inplace = True) 



#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Teste de Shapiro-Wilk
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('Teste de Shapiro-Wilk -  testas as features sem o realizado no FPM')
print("--------------------------------------------------------------------------")
# testas as features sem o realizado no FPM

#for uf in itenscapitais:

#  for item in itensreceitas:
    
#    datatmp = datarec[datarec['RECEITA']== item].copy()
#    datatmp = datatmp.drop(['RECEITA', 'UF'], axis='columns')
#    ml.teste_shapiro_rec(datatmp, 0.05, uf, item)



#-----------------------------------------------------------------------------------------------------------------------------------------------------
# SELEÇÃO DE FEATURES - MODELOS INDiViDUAIS
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('Preparando o datasetpara modelos isolados de ML')
print("--------------------------------------------------------------------------")

# Cria o período anterior para modelos regressivos
p = 3
datarec_ml = datarec.copy()
# faz o loop para criar as variáveis targets anteriores
for v in range(p):
  # coloca o f para unção concatenar o nome com o numero
  datarec_ml[f'x_{v + 1}'] = datarec_ml.Realizado.shift(v + 1)
  # Reordena as colunas
  datarec_ml = datarec_ml[[f'x_{v + 1}'] + datarec_ml.columns[:-1].tolist()]

print("")
print('Verifica o percentual dos valores Nan')
print("--------------------------------------------------------------------------")
# Verifica o percentual dos valores Nan
datarec_ml[datarec_ml.columns[datarec_ml.isnull().any()]].isnull().sum() * 100 / datarec_ml.shape[0]
datarec_ml.dropna(axis=0, inplace=True)

#print(datarec_ml.isna().sum())

print("")
print('Divisão e escalonamento')
print("--------------------------------------------------------------------------")
#Testando
datst = datarec_ml[datarec_ml.RECEITA == str('CFEM')].copy()
Xtrain, Xval, ytrain, yval = ml.DivisaoDtSet_CriaTarget(datst, 0.7, True)
 
#print(type(yval))

print("")
print('Testando escalonamento')
print("--------------------------------------------------------------------------")
# Testando escalonamento
datst = datarec_ml[datarec_ml.RECEITA == str('FPM')].copy()
Xtraintes, ytraines = ml.Normaliza(Xtrain, ytrain, Xtrain.columns)
#print(type(Xtraintes), type(ytraines))


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# SELEÇÃO DE FEATURES - PCA
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('INICIANDO PCA')
print("--------------------------------------------------------------------------")


pca()
# Lista para armazenar os resultados por Receita
score_PCA = []

# Contador para o Id
id_cont = 1
# Id feature PCA
id_feature_cont = 1

# Apaga dados antriores
sql = "delete from Rank"
res = cursor.execute(sql).fetchall()
# Atualiza banco
connection.commit()

# Apaga dados antriores
sql = "delete from FeaturesPCA"
res = cursor.execute(sql).fetchall()
# Atualiza banco
connection.commit()

sql = "delete from Features"
res = cursor.execute(sql).fetchall()
# Atualiza banco
connection.commit()


# As receitas foram avaliadas separadas. Segunda iteração verificar qual o problema de incluir no laço.
# Decompor a matriz de variâncias e covariâncias em componentes principais (aplicar o PCA)
for uf in itenscapitais:

  # Varrendos os tipos de receitas
  for iterc in itensreceitas:

     # Faz uma cópia do Datasete, pois será modificado
    dataPCA = datarec_ml[(datarec_ml['RECEITA'] == iterc) & (datarec_ml['UF'] == uf)].copy()

    # Retira o índice
    dataPCA = dataPCA.reset_index()

    # Testa a quantidad de linhas
    if np.size(dataPCA, 0)  == 0:
        continue

    print("")
    print('PCA - Receita: ', iterc)
    print("----------------------------------------------------------------------------------------------------------")

    nvdtrecDatas, nvdtrecReal, mPca, resultado, autovalorpca = ml.Resultado_PCA(iterc, 0.95, dataPCA)
    # Pesos das componentes principais AutoVetores
    #print(resultado)

    k = ml.Identifica_Qnt_Componentes(iterc, resultado, mPca)

    # Variância das componentes principais do FPM
    #print('Variância das componentes principais Receita ' + iterc)
    #print(autovalorpca)

    # Relacionar as features que fazem parte do autovetor das componentes principais.
    # Aqui mostra 
    #print('Mostrando colunas para ' + iterc)
    #print(resultado['loadings'].T)

    # Faço o rankeamento das Features que foram reduzidas em cada componente
    # Esta é a lista que deve ser gravada para saber o ranking das features (importância das features)
    ReankFeatures = ml.Cria_Rankfeatures(resultado)
    #print('Features rakeadas para ' + iterc)
    #print(ReankFeatures)

    # Pivotando a tabela apenas para demonstração
    fdtPCACFEM = pd.pivot_table(ReankFeatures, values='loading', index=['feature', 'type'], columns=['PC'], aggfunc=np.sum, margins=True)
    #print('Pivotando a tabela para demonstração de ' + iterc)
    #print(fdtPCACFEM)

    # Treinando o novo dataset criado pelo PCA para avaliar o desempenho dele
    MAE, MSE, RMSE, R2, pred, yval = ml.Avaliar_PCA(autovalorpca, nvdtrecDatas, nvdtrecReal, 100, 2, 0.7)
    # Salvando os resultados na lista
    score_PCA.append([uf,iterc, k, 'PCA', 'RandomForestRegressor', MAE, MSE, RMSE, R2, 
                      ReankFeatures.PC, ReankFeatures.feature, ReankFeatures.loading, 
                      ReankFeatures.index.array, ReankFeatures.type, 
                      resultado['explained_var']])

    # Gravando no banco
    #Atualiza
    print("Atualiza Registro")
    sql = "insert into Rank (id, uf, receita, k, metodo , model , MAE , MSE, RMSE, R2) values ("
    sql = sql +  str(id_cont) + ", "
    sql = sql + "'" + uf + "', "
    sql = sql + "'" + iterc + "', "
    sql = sql +  str(k) + ", "
    sql = sql + "'PCA', "
    sql = sql + "'RANDOMFOREST', "
    sql = sql + str(MAE) + ", "
    sql = sql +  str(MSE) + ", "
    sql = sql +  str(RMSE) + ", "
    sql = sql +  str(R2) + ") "
    #print(sql)
    res = cursor.execute(sql).fetchall()
    # Atualiza banco
    connection.commit()
    id_cont = id_cont + 1


    print("Atualiza Registro Features PCA")
    #Gravando Features PCA
    for i in range(len(ReankFeatures.PC)):
        print(i, ReankFeatures.PC[i], ReankFeatures.feature[i], ReankFeatures.loading[i], ReankFeatures.index.array[i], ReankFeatures.type[i]  )
        # Gravando no banco
        #Atualiza
        sql = "insert into FeaturesPCA (id,id_rank,indice,Rankeameto,feature,importance,score) values ("
        sql = sql + str(id_feature_cont) + ", "
        sql = sql + str(id_cont) + ", "
        sql = sql + str(i+1) + ", "
        sql = sql + "'" + str(ReankFeatures.PC[i]) + "', "
        sql = sql + "'" + str(ReankFeatures.feature[i]) + "', "
        sql = sql + str(ReankFeatures.loading[i]) + ", "
        sql = sql + "'" + str(ReankFeatures.type[i]) + "') "
        #print(sql)
        res = cursor.execute(sql).fetchall()
        # Atualiza banco
        connection.commit()
        #Atualiza o contador de id

        # Incluindo as features
        print("Atualiza Features")

        #Gravando Features 
        # Gravando no banco
        #Atualiza
        sql = "insert into Features (id,id_rank,indice,feature,importance) values ("
        sql = sql + str(id_feature_cont) + ", "
        sql = sql + str(id_cont) + ", "
        sql = sql + str(i+1) + ", "
        sql = sql + "'" + str(ReankFeatures.feature[i]) + "', "
        sql = sql + str(ReankFeatures.loading[i]) + ") "
        #print(sql)
        res = cursor.execute(sql).fetchall()
        # Atualiza banco
        connection.commit()

        #Atualiza o contador de id
        id_feature_cont = id_feature_cont + 1

    #print("ReankFeatures.Loading PCA")
    #print("     ", ReankFeatures.loading)
    #print("ReankFeatures.index.array) PCA")
    #print("     ", ReankFeatures.index.array)
    #print("ReankFeatures.type PCA")
    #print("     ", ReankFeatures.type)
    #print("ReankFeatures explained_var")
    #print("     ", resultado['explained_var'])

    # Incrementa o contador de id dos registros

    #Cria o Dataset de predição
    pred = pd.DataFrame(data={"target":yval, "prediction":pred})

    pred.shape[1]

    metrica = ' com métrica RMSE=' + str(np.round(RMSE,4))
    print(metrica)




#-----------------------------------------------------------------------------------------------------------------------------------------------------
# XGBoost
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('INICIANDO STANDALANOE XGBOOST')
print("--------------------------------------------------------------------------")

scores_xgb = []
lstFeatures = []
lstImportancia = []
lstPrediction = []
lstTarget = []

# Mensurando o tempo de execução
Totinicio_exe = time.time()
# Inclui um limete para as features
limete_xgb = 0.005

for uf in itenscapitais:

  print('************ Treinando UF: ', uf, ' ******************** ')

  # Verificando quais os melhor valores por receita
  for item in itensreceitas:
      
      # Mensurando o tempo de execução
      RecIni_exe = time.time()
      
      dtreceita = datarec_ml[(datarec_ml.RECEITA == item) & (datarec_ml.UF == uf)].copy()
      # Dividindos dados X,y
      y = dtreceita.Realizado
      X = dtreceita.drop(['RECEITA', 'Realizado','UF'], axis=1)

      # Dividi-se os dados X,y DataSet em Treino e Teste
      Xtr, Xval, ytr, yval = ml.DivisaoDtSet_CriaTarget(dtreceita, 0.6, True)
      #print(Xtr.info())
      #print(type(Xtr))
      #print(type(Xval))

      # normalizando o dataset
      Xtr, ytr = ml.Normaliza(Xtr, ytr, Xtr.columns)
      Xval, yval = ml.Normaliza(Xval, yval, Xtr.columns)

      MAE, MSE, RMSE, R2, features_exgb, pred = ml.Monta_XGBoost( scores_xgb, Xtr, Xval, ytr, yval, 0.08, 1300, 0.5)
      
      dtxg = pd.DataFrame(features_exgb, columns=['features', 'score'])
      #print(dtxg)
      
      dtxg = dtxg[dtxg['score']>=limete_xgb]
      dtxg['score'] = np.round(dtxg['score']*100, 2)

      titulo = "Xgboost Importância da Features Receita=" + item
      
      # Plotando um gráfico de importância das features
      #PlotaImportanciaFeatures(titulo, dtxg, 'score', 'features', 10, 9)

      #Cria o Dataset de predição
      predic = pd.DataFrame(data={"target":yval, "prediction":pred})
      
      metrica = ' com métrica RMSE=' + str(np.round(RMSE,4))
      # Plotando um gráfico comparativo da predição
      #Plota_Avalia_Predicao(predic, 'XGBoos', item, metrica)

      x_array = np.array(features_exgb)
      for i in x_array:
          #print(i[0], i[1])
          scores_xgb.append([uf, item, MAE, MSE, RMSE, R2, i[0], i[1], pred, yval ])
          lstPrediction.append(pred)
          lstTarget.append(yval)
      
      # Mensurando o tempo de execução
      Recfim_exe = time.time()
      print(item, ' Duração=', (Recfim_exe - RecIni_exe), 'seg com RMSE=', RMSE)
          
# Mensurando o tempo de execução
Totfim_exe = time.time()
print('Tempo de execução Total ', (Totfim_exe - Totinicio_exe), 'seg')

# Criando dataframe a partir do vetor de scores
featuresdt_xgb = pd.DataFrame(scores_xgb, columns=['UF', 'RECEITA', 'MAE', 'MSE', 'RMSE', 'R2', 'Features', 'score', 'prediction', 'target']) 
featuresdt_xgb['score'] = pd.to_numeric(featuresdt_xgb['score'],errors = 'coerce')


featuresdt_xgb['MAE'] = featuresdt_xgb['MAE'].round(10)
featuresdt_xgb['MSE'] = featuresdt_xgb['MSE'].round(10)
featuresdt_xgb['R2'] = featuresdt_xgb['R2'].round(10)
featuresdt_xgb['RMSE'] = featuresdt_xgb['RMSE'].round(10)
featuresdt_xgb['score'] = featuresdt_xgb['score'].round(10)

melhor_xgb = pd.pivot_table(featuresdt_xgb[featuresdt_xgb['score']>=limete_xgb], values='score', index=['Features'], 
                            columns=['UF', 'RECEITA', 'RMSE', 'MAE', 'MSE', 'R2'], aggfunc=np.max, fill_value=0)

dt1 = featuresdt_xgb[featuresdt_xgb['score']>=0.005]

# Contador do indice da tabela de predições
id_pred_cont = 1

reg = ""
#  Gravado o dataset
cont_ind = 1
for i in dt1.index: 

    # Testa se houve alteracao em um destes campo, caso sim, grava novo registro
    tmp = dt1['UF'][i] + dt1['RECEITA'][i] +str(dt1['MAE'][i])+ str(dt1['MSE'][i])+str(dt1['RMSE'][i])+str(dt1['R2'][i])

    if reg != tmp:
        reg = tmp
        #Zera contador do indice
        cont_ind = 1
        id_cont = id_cont + 1
        # Gravando no banco
        print('Imprimindo i: ', i)
        #Atualiza
        print("Atualiza Registro")
#        print('Imprimindo dt[i]: ', dt1['UF'][i], 
#            dt1['RECEITA'][i], dt1['MAE'][i], 
#            dt1['MSE'][i], dt1['RMSE'][i], 
#            dt1['R2'][i], 
#            'Features: ', dt1['Features'][i], 
#            'score: ', dt1['score'][i], 
#            'prediction: ', dt1['prediction'][i], 
#            'target: ', dt1['target'][i])

        sql = "insert into Rank (id,uf,receita,k,metodo,model,MAE,MSE,RMSE,R2) values ("
        sql = sql + str(id_cont) + ", "
        sql = sql + "'" + str(dt1['UF'][i]) + "', "
        sql = sql + "'" + str(dt1['RECEITA'][i]) + "', "
        sql = sql +  str(len(lstFeatures)) + ", "
        sql = sql + "'STANDALONE', "
        sql = sql + "'XGBOOST', "
        sql = sql + str(dt1['MAE'][i]) + ", "
        sql = sql + str(dt1['MSE'][i]) + ", "
        sql = sql + str(dt1['RMSE'][i]) + ", "
        sql = sql +  str(dt1['R2'][i]) + ") "
        print(sql)
        res = cursor.execute(sql).fetchall()
        # Atualiza banco
        connection.commit()
        #Atualiza o contado do indice

        print('type do prediction: ', type(dt1['prediction'][i]), ' type do target: ', type(dt1['target'][i]))
        #print('prediction: ', dt1['prediction'][i], 'target: ', dt1['target'][i])
        

        print( dt1['prediction'][i] )
        print( dt1['target'][i] )


    sFeature =  dt1['Features'][i]
    sScore =  dt1['score'][i]

    sql = "insert into Features (id,id_rank,indice,feature,importance) values ("
    sql = sql + str(id_feature_cont) + ", "
    sql = sql + str(id_cont) + ", "
    sql = sql + str(cont_ind) + ", "
    sql = sql + "'" + sFeature + "', "
    sql = sql + str(sScore) +  ") "
    print(sql)
    res = cursor.execute(sql).fetchall()
    # Atualiza banco
    connection.commit()
    #Atualiza o contador de id
    cont_ind = cont_ind + 1
    id_feature_cont = id_feature_cont + 1

     
    # Guarda target
    print(type(dt1['target']))
    #for j in dt1['target']:
    #    lstPrediction.append(float(j))




    ## Incluindo as features
#    print("Atualiza Predições")
#    print(len(lstPrediction))
#    for n in range(len(lstPrediction)):
#        #Gravando Features 
#        # Gravando no banco
#        #Atualiza
#        sql = "insert into Predicao (id,id_rank,indice,prediction,target) values ("
#        sql = sql + str(id_feature_cont) + ", "
#        sql = sql + str(id_cont) + ", "
#        sql = sql + str(n+1) + ", "
#        sql = sql + str(lstPrediction[n]) + ", "
#        sql = sql + str(lstTarget[n]) + ") "
#        print(sql)
#        res = cursor.execute(sql).fetchall()
#        # Atualiza banco
#        connection.commit()
#        #Atualiza o contador de id
#        id_pred_cont = id_pred_cont + 1


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# GRADIETBOOSTINGREGRESSOR
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('INICIANDO GRADIET BOOSTING REGRESSOR')
print("--------------------------------------------------------------------------")

score_gdr = []

# Mensurando o tempo de execução
Totinicio_exe = time.time()
# Inclui um limete para as features
limite_gb = 0.005

Recinicio_exe = 0

for uf in itenscapitais:

    print('************ Treinando UF: ', uf, ' ******************** ')


    # Verificando quais os melhor valores por receita
    for item in itensreceitas:

        # Mensurando o tempo de execução
        Recinicio_exe = time.time()

        dtreceita = datarec_ml[(datarec_ml.RECEITA == item) & (datarec_ml.UF == uf)].copy()
        # Dividindos dados X,y
        y = dtreceita.Realizado
        X = dtreceita.drop(['RECEITA', 'Realizado','UF'], axis=1)

        # Dividi-se os dados X,y DataSet em Treino e Teste
        Xtr, Xval, ytr, yval = ml.DivisaoDtSet_CriaTarget(dtreceita, 0.6, True)
        #print(Xtr.info())
        #print(type(Xtr))
        #print(type(Xval))

        # normalizando o dataset
        Xtr, ytr = ml.Normaliza(Xtr, ytr, Xtr.columns)
        Xval, yval = ml.Normaliza(Xval, yval, Xtr.columns)

        pred, MAE, MSE, RMSE, R2, features_gbr = ml.Monta_GradientBoosting(Xtr, ytr, Xval, yval)

        dtgb = pd.DataFrame(features_gbr, columns=['features', 'score'])
        dtgb = dtgb[dtgb['score']>=limite_gb]
        dtgb['score'] = np.round(dtgb['score']*100, 2)
        #print(dtxg)

        titulo = "GradientBoosting Importância da Features Receita=" + item

        # Plotando um gráfico de importância das features
        #PlotaImportanciaFeatures(titulo, dtgb, 'score', 'features', 10, 9)

        #Cria o Dataset de predição
        predic = pd.DataFrame(data={"target":yval, "prediction":pred})

        metrica = ' com métrica RMSE=' + str(np.round(RMSE,4))
        # Plotando um gráfico comparativo da predição
        #Plota_Avalia_Predicao(predic, 'GradientBoosting', item, metrica)

        plt.show()
        x_array = np.array(features_gbr)
        for i in x_array:
            #print(i[0], i[1])
            score_gdr.append([uf, item, MAE, MSE, RMSE, R2, i[0], i[1], pred, yval ])

        # Mensurando o tempo de execução
        Recfim_exe = time.time()
        print(item,' Duração=', (Recfim_exe - Recinicio_exe), 'seg')

# Mensurando o tempo de execução
Totfim_exe = time.time()
print('Tempo de execução Total ', (Totfim_exe - Totinicio_exe)/60, 'mim')

# Criando dataframe a partir do vetor de scores
featuresdt_gdr = pd.DataFrame(score_gdr, 
                              columns=['UF', 'RECEITA', 'MAE', 'MSE', 'RMSE', 'R2', 'Features', 'score', 'prediction', 'target']) 
featuresdt_gdr['score'] = pd.to_numeric(featuresdt_gdr['score'],errors = 'coerce')

print(featuresdt_gdr)

featuresdt_gdr['MAE'] = featuresdt_gdr['MAE'].round(10)
featuresdt_gdr['MSE'] = featuresdt_gdr['MSE'].round(10)
featuresdt_gdr['R2'] = featuresdt_gdr['R2'].round(10)
featuresdt_gdr['RMSE'] = featuresdt_gdr['RMSE'].round(10)
featuresdt_gdr['score'] = featuresdt_gdr['score'].round(10)

melhor_gdr = pd.pivot_table(featuresdt_gdr[featuresdt_gdr['score']>=limite_gb], values='score', index=['Features'], 
                            columns=['UF', 'RECEITA', 'RMSE', 'MAE', 'MSE', 'R2'], aggfunc=np.max)
print(melhor_gdr)



dt1 = featuresdt_gdr

#  Gravado o dataset
for i in dt1.index: 


    #Pega as features e joga na lista
    #features = str(dt1['Features'][i])
    #features = features.replace("[","")
    #features = features.replace("]","")
    # Cria a lista
    lstFeatures = dt1['Features'][i].toList()
    print(lstFeatures)

    #Pega as features e joga na lista
    #Importancia = str(dt1['score'][i])
    #Importancia = Importancia.replace("[","")
    #Importancia = Importancia.replace("]","")
    # Cria a lista
    lstImportancia = dt1['score'][i].toList()
    print(lstImportancia)

    # Cria a lista
    lstPrediction = dt1['prediction'][i].toList()
    print(lstPrediction)

    # Cria a lista
    lstTarget = dt1['target'][i].toList()
    print(lstTarget)


    # Gravando no banco
    #Atualiza
    print("Atualiza Registro")
    sql = "insert into Rank (id,uf,receita,k,metodo,model,MAE,MSE,RMSE,R2) values ("
    sql = sql + str(id_cont) + ", "
    sql = sql + "'" + str(dt1['UF'][i]) + "', "
    sql = sql + "'" + str(dt1['RECEITA'][i]) + "', "
    sql = sql +  str(len(lstFeatures)) + ", "
    sql = sql + "'STANDALONE', "
    sql = sql + "'GRADIETBOOSTINGREGRESSOR', "
    sql = sql + str(dt1['MAE'][i]) + ", "
    sql = sql + str(dt1['MSE'][i]) + ", "
    sql = sql + str(dt1['RMSE'][i]) + ", "
    sql = sql +  str(dt1['R2'][i]) + ") "
    print(sql)
    res = cursor.execute(sql).fetchall()
    # Atualiza banco
    connection.commit()


    # Incluindo as features
    print("Atualiza Features")
    for n in range(len(lstFeatures)):
        #Gravando Features 
        # Gravando no banco
        #Atualiza
        if lstImportancia[n] <= 50:
            sql = "insert into Features (id,id_rank,indice,feature,importance, prediction, target) values ("
            sql = sql + str(id_feature_cont) + ", "
            sql = sql + str(id_cont) + ", "
            sql = sql + str(n+1) + ", "
            sql = sql + "'" + str(lstFeatures[n]) + "', "
            sql = sql + str(lstImportancia[n]) + ", "
            sql = sql + str(lstPrediction[n]) + ", "
            sql = sql + str(lstTarget[n]) + ") "
            print(sql)
            res = cursor.execute(sql).fetchall()
            # Atualiza banco
            connection.commit()
            #Atualiza o contador de id
            id_feature_cont = id_feature_cont + 1


    #Atualiza o contado do indice
    id_cont = id_cont + 1


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# RANDON FOREST
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('RaNDON FOREST')
print("--------------------------------------------------------------------------")


# scores_rf = []
# Métodos ensembles como o algoritmo Random Forest, podem ser usados para estimar a importância de cada atributo. 
# Ele retorna um score para cada atributo, quanto maior o score, maior é a importância desse atributo.
scores_rf = []


# Apaga dados antriores
sql = "delete from RandomForest"
res = cursor.execute(sql).fetchall()
# Atualiza banco
connection.commit()


#tirando a raiz quadrada das qnt de features
nfeatures = int( math.sqrt( len(datarec_ml.columns))/2)
print(nfeatures)

# Mensurando o tempo de execução
Totinicio_exe = time.time()

# Id manaual
id_ramdomforest = 1

for uf in itenscapitais:

    print('************ Treinando UF: ', uf, ' ******************** ')
    
    # Verificando quais os melhor valores por receita
    for item in itensreceitas:

        # Mensurando o tempo de execução
        Recinicio_exe = time.time()
        
        dtreceita = datarec_ml[(datarec_ml.RECEITA == item) & (datarec_ml.UF == uf)].copy()
        
        # Dividindos dados X,y DataSet em Treino e Teste
        Xtrain, Xval, ytrain, yval = ml.DivisaoDtSet_CriaTarget(dtreceita, 0.6, True)
        #print(Xtrain.info())
        #print(type(Xtrain))
        #print(type(Xval))

        # normalizar datasets treinos, validação e test
        X_nor, y_nor = ml.Normaliza(Xtrain, ytrain, Xtrain.columns)
        XvalNorm, yval = ml.Normaliza(Xval, yval, Xval.columns)


        # incrementa os parâmetros na qnt features
        for t in range(nfeatures+2):
            if t == 0: 
                ntree = 50
            else:
                ntree = 150*t

            # usando parte da sequencia de fibonacci
            for i in [1,2,3,5]:
                nexenos = i+1
                #print(item, ntree, nexenos)
                mdl_rf, MAE, MSE, RMSE, R2, pred = ml.Monta_RandomForest(X_nor, y_nor, XvalNorm, yval, ntree, nexenos)
                
                sorted_idx = np.argsort(mdl_rf.feature_importances_)
                features_rf_name = []
                features_rf = []

                # Mensurando o tempo de execução
                inicio_exe = time.time()
        
                # guarda as features
                for index in sorted_idx:
                    features_rf_name.append([X_nor.columns[index]])
                    features_rf.append(mdl_rf.feature_importances_[index]) 
        
                scores_rf.append([uf, item, ntree, nexenos, MAE, MSE, RMSE, R2, 
                                features_rf_name, features_rf, pred, yval])
                
                # Mensurando o tempo de execução
                fim_exe = time.time()
                print('Nr Arvores=', ntree, ' Nr amostras=', nexenos, ' Duração=', (fim_exe - inicio_exe), 'seg e com RMSE=', RMSE)
                
                #Gravar RANDOMFOREST
                print("Grava Random Forest selection")
                # Gravando no banco
                #Atualiza
                sql = "insert into RandomForest (id, receita,ntree,nexenos,duracao,RMSE) values ("
                sql = sql + str(id_ramdomforest) + ","
                sql = sql + "'" + item + "', "
                sql = sql + str(ntree) + ", "
                sql = sql + str(nexenos) + ", "
                sql = sql + str(fim_exe - inicio_exe) + ", "
                sql = sql + str(RMSE) + ") "
                print(sql)
                res = cursor.execute(sql).fetchall()
                # Atualiza banco
                connection.commit()
                #Atualiza o contador de id
                id_ramdomforest = id_ramdomforest + 1


        # Mensurando o tempo de execução
        Recfim_exe = time.time()
        print('Tempo de execução Receita ', item, '=', (Recfim_exe - Recinicio_exe), 'seg')


# Mensurando o tempo de execução
Totfim_exe = time.time()
print('Tempo de execução Total ', (Totfim_exe - Totinicio_exe)/60, 'mim')

featuresdt_rf = pd.DataFrame(scores_rf, columns=['UF', 'RECEITA','NrArvores', 'QntAmosNo','MAE', 'MSE', 'RMSE', 'R2', 'Features','Importancia','prediction', 'target'])

featuresdt_rf['MAE'] = featuresdt_rf['MAE'].round(15)
featuresdt_rf['MSE'] = featuresdt_rf['MSE'].round(15)
featuresdt_rf['R2'] = featuresdt_rf['R2'].round(15)
featuresdt_rf['RMSE'] = featuresdt_rf['RMSE'].round(15)

melhores_rf = featuresdt_rf.drop(['MSE', 'R2', 'Features','Importancia'], axis='columns')
melhores_rf

# Pivotando a tabela valores maiores que 7338,637
melhor_rf = pd.pivot_table(melhores_rf, values=['RMSE','MAE'], index=['NrArvores', 'QntAmosNo'], columns=['RECEITA'], aggfunc=np.min)
print("Pivotando a tabela valores maiores que 7338,637")
print(melhor_rf)

print("Melhores features")
print(featuresdt_rf)

# Gravar no banco
print("DT1")
dt1 = featuresdt_rf[featuresdt_rf['RMSE'].isin(featuresdt_rf.groupby('RECEITA')['RMSE'].min())]
print(dt1)

#  Gravado o dataset
for i in dt1.index: 


    #Pega as features e joga na lista
    #features = str(dt1['Features'][i])
    #features = features.replace("[","")
    #features = features.replace("]","")
    # Cria a lista
    lstFeatures = dt1['Features'][i].toList()
    print(lstFeatures)

    #Pega as features e joga na lista
    #Importancia = str(dt1['score'][i])
    #Importancia = Importancia.replace("[","")
    #Importancia = Importancia.replace("]","")
    # Cria a lista
    lstImportancia = dt1['score'][i].toList()
    print(lstImportancia)

    # Cria a lista
    lstPrediction = dt1['prediction'][i].toList()
    print(lstPrediction)

    # Cria a lista
    lstTarget = dt1['target'][i].toList()
    print(lstTarget)


    # Gravando no banco
    #Atualiza
    print("Atualiza Registro")

    sql = "insert into Rank (id,uf,receita,k,metodo,model,nrArvores,qntAmosNo,MAE,MSE,RMSE,R2) values ("
    sql = sql + "'" + str(id_cont) + "', "
    sql = sql + "'" + str(dt1['UF'][i]) + "', "
    sql = sql + "'" + str(dt1['RECEITA'][i]) + "', "
    sql = sql +  str(len(lstFeatures)) + ", "
    sql = sql + "'STANDALONE', "
    sql = sql + "'RANDOMFOREST', "
    sql = sql + str(dt1['NrArvores'][i]) + ", "
    sql = sql + str(dt1['QntAmosNo'][i]) + ", "
    sql = sql + str(dt1['MAE'][i]) + ", "
    sql = sql + str(dt1['MSE'][i]) + ", "
    sql = sql + str(dt1['RMSE'][i]) + ", "
    sql = sql +  str(dt1['R2'][i]) + ") "
    print(sql)
    res = cursor.execute(sql).fetchall()
    # Atualiza banco
    connection.commit()


    # Incluindo as features
    print("Atualiza Features")
    for n in range(len(lstFeatures)):
        #Gravando Features 
        # Gravando no banco
        #Atualiza
        if lstImportancia[n] <= 50:
            sql = "insert into Features (id,id_rank,indice,feature,importance) values ("
            sql = sql + str(id_feature_cont) + ", "
            sql = sql + str(id_cont) + ", "
            sql = sql + str(n+1) + ", "
            sql = sql + "'" + str(lstFeatures[n]) + "', "
            sql = sql + str(lstImportancia[n]) + ") "
            print(sql)
            res = cursor.execute(sql).fetchall()
            # Atualiza banco
            connection.commit()
            #Atualiza o contador de id
            id_feature_cont = id_feature_cont + 1


    #Atualiza o contado do indice
    id_cont = id_cont + 1




for item in itensreceitas:
    
    limite_import = 0.005
    titulo = "Random Forest Feature Importance Receita " + item
    # Plota a importancia das features
    dtimp = pd.DataFrame(data={"Importancia":dt1[dt1['RECEITA'] == item].Importancia.tolist()[0],
                               "Features":np.array( dt1[dt1['RECEITA'] == item].Features.transpose().tolist()[0]).transpose()[0].tolist()})
                               
    dtimp = dtimp[dtimp["Importancia"]>=limite_import]
    dtimp["Importancia"] = np.round(dtimp["Importancia"]*100,2)
    
    #PlotaImportanciaFeatures(titulo, dtimp, itscore, itfeatures, ftaml, ftamh):
    
    # Plotando um gráfico de importância das features
    #PlotaImportanciaFeatures(titulo, dtimp, 'Importancia', 'Features', 10, 7)

    #Cria o Dataset de predição
    pred = pd.DataFrame(data={"target":np.array(dt1[dt1['RECEITA'] == item].target.tolist()[0]), 
                              "prediction":np.array(dt1[dt1['RECEITA'] == item].prediction.tolist()[0])})
    
    metrica = ' com métrica RMSE=' + str(np.round(dt1.RMSE.unique()[0],4))
    # Plotando um gráfico comparativo da predição
    #ml.Plota_Avalia_Predicao(pred, 'RandomForestRegressor', item, metrica)
    
# Guarda os melhores valores para utilizar no valor combinado   RECEITA	NrArvores	QntAmosNo	
melhor_rf = pd.DataFrame(data={"RECEITA":np.array(dt1['RECEITA']), 
                               "NrArvores":np.array(dt1['NrArvores']),
                               "QntAmosNo":np.array(dt1['QntAmosNo'])})

print("Melhor_rf")
print(melhor_rf)



#-----------------------------------------------------------------------------------------------------------------------------------------------------
# FILTRO COMBINADO
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print("")
print('INICIANDO O FILTRO COMBINADO')
print("--------------------------------------------------------------------------")

#Testando os valores para evitar o error da série
uf= "MA"
for it in itensreceitas:
  print('Receita: ', it)

  dfrf = melhor_rf[(melhor_rf['RECEITA']==it) & (melhor_rf['UF']==uf)].copy()

  print('Type n_estimators: ', type(dfrf['NrArvores']))
  ntree = int("".join([str(i) for i in dfrf['NrArvores']]))
  print('n_estimators: ', ntree)
  
  print('Type min_samples_leaf: ', type(dfrf['QntAmosNo']))
  nexenos = int("".join([str(i) for i in dfrf['QntAmosNo']]))
  print('min_samples_leaf: ', nexenos)
  print(type(ntree), type(nexenos))


# Vetor com os scores gerais
scores = []

receita = "FPM"
uf = "MA"
datarec_ml[(datarec_ml.RECEITA == str(receita)) &  (datarec_ml['UF'] == str(uf))]

# Executar multiplos treinamentos com a função de combinação de modelos para seleção de features
# Mensurando o tempo de execução
Totinicio_exe = time.time()

for uf in itenscapitais:

    print(" ")
    print("=========================================================================================")
    print("Capital: ", uf)
    print("=========================================================================================")

    for receita in itensreceitas:
        #receita = "FPM"
        # Mensurando o tempo de execução
        Recinicio_exe = time.time()

        # Mensurando o tempo de execução
        inicio_exe = time.time()  
        # Filtra o banco por RECEITAS
        datst = datarec_ml[(datarec_ml.RECEITA == str(receita)) &  (datarec_ml['UF'] == str(uf))].copy()
        #datst.head()
        print(" ")
        print("Capital: ", uf, " Receita: ", receita, " Qnt Registros ", datst.x_1.count() )
        print("=========================================================================================")

        # verifica se a quantidade de registros é maior que 0
        if datst.x_1.count() <= 0:
            continue

        ralpha = 4
        lalpha = 4.0
        # Executa multiplos treinamentos de Features
        #print('Type datst: ', type(datst))
        #print('Type scores: ', type(scores))
        #print('Type ralpha: ', type(ralpha))
        #print('Type lalpha: ', type(lalpha))
        ml.trainingAll(datst, scores, receita, uf, ralpha, lalpha, True)

        # Mensurando o tempo de execução
        Recfim_exe = time.time()
        print("Capital: ", uf, " Receita: ", receita, ' Duração=', (Recfim_exe - Recinicio_exe)/60, 'min')

        
# Mensurando o tempo de execução
Totfim_exe = time.time()
print('Tempo de execução Total ', (Totfim_exe - Totinicio_exe)/60, 'mim')

# Criando dataframe a partir do vetor de scores
featuresdt = pd.DataFrame(scores,columns=["UF", "RECEITA","k","Features","score","Seletor","Ensemble",
                                          "R2", "MAE", "MSE", "RMSE", "Target", "Previsão"])

srt1 = featuresdt
#  Gravado o dataset
for i in dt1.index: 


    #Pega as features e joga na lista
    #features = str(dt1['Features'][i])
    #features = features.replace("[","")
    #features = features.replace("]","")
    # Cria a lista
    lstFeatures = dt1['Features'][i].toList()
    print(lstFeatures)

    #Pega as features e joga na lista
    #Importancia = str(dt1['score'][i])
    #Importancia = Importancia.replace("[","")
    #Importancia = Importancia.replace("]","")
    # Cria a lista
    lstImportancia = dt1['score'][i].toList()
    print(lstImportancia)

    # Cria a lista
    lstPrediction = dt1['prediction'][i].toList()
    print(lstPrediction)

    # Cria a lista
    lstTarget = dt1['target'][i].toList()
    print(lstTarget)


    # Gravando no banco
    #Atualiza
    print("Atualiza Registro")
    sql = "insert into Rank (id,uf,receita,k,metodo,model,MAE,MSE,RMSE,R2) values ("
    sql = sql + str(id_cont) + ", "
    sql = sql + "'" + str(dt1['UF'][i]) + "', "
    sql = sql + "'" + str(dt1['RECEITA'][i]) + "', "
    sql = sql +  str(len(lstFeatures)) + ", "
    sql = sql + "'COMBINADO', "
    sql = sql + "'" + str(dt1['Seletor'][i]) + "x" + str(dt1['Ensemble'][i])  + "', "
    sql = sql + str(dt1['MAE'][i]) + ", "
    sql = sql + str(dt1['MSE'][i]) + ", "
    sql = sql + str(dt1['RMSE'][i]) + ", "
    sql = sql +  str(dt1['R2'][i]) + ") "
    print(sql)
    res = cursor.execute(sql).fetchall()
    # Atualiza banco
    connection.commit()


    # Incluindo as features
    print("Atualiza Features")
    for n in range(len(lstFeatures)):
        #Gravando Features 
        # Gravando no banco
        #Atualiza
        sql = "insert into Features (id,id_rank,indice,feature,importance) values ("
        sql = sql + str(id_feature_cont) + ", "
        sql = sql + str(id_cont) + ", "
        sql = sql + str(n+1) + ", "
        sql = sql + "'" + str(lstFeatures[n]) + "', "
        sql = sql + str(lstImportancia[n]) +  ") "
        print(sql)
        res = cursor.execute(sql).fetchall()
        # Atualiza banco
        connection.commit()
        #Atualiza o contador de id
        id_feature_cont = id_feature_cont + 1


    #Atualiza o contado do indice
    id_cont = id_cont + 1

# Atualiza banco
connection.commit()
# Fecha o banco
connection.close()

