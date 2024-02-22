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
# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
# pandas
import pandas as pd

import patsy
import statsmodels.api as sm
import pandas.testing as tm
import scipy
from scipy import stats
from matplotlib import pylab
from pylab import *

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
import sqlite3

def teste_shapiro_rec(tsdf, alpha, uf, item):
# função para testar a normalidade dos dados por receita  
    print('Capital de Estudo: ', uf)
    print('Aplicando teste de Shapiro-Wilk para a RECEITA: ', item)
    cols = tsdf.columns
    qnt = 0
    #print(columns)
    for i in cols:
        texto = '\n'
        hipotese = ''
        a,b= stats.shapiro(tsdf[[i]])
        texto = texto + "\n Estatística=" + str(a) + '\n' + "p-value=" + str(b)
        #print ("Estatística=", a, "p-value=", b)
        if b < alpha:  
            #hipotese = f' \n A hipótese nula pode ser rejeitada! \n A variável {i} ​​não está normalmente distribuídas'
            qnt = qnt + 1
        else:
            hipotese = f' \n A hipótese nula NÃO pode ser rejeitada!\n a variável {i} esta distribuida normalmente'
            # Imprime apenas que esta distribuido normalmente
            #print(hipotese)
            print(texto + hipotese)

    # Ao final informa quantas colunas não estão distribuidas normalmente
    print(qnt,' colunas não possuem distribuição normal')

#Divisão e escalonamento
#O 'Dataset' foi dividi e testou-se duas funções a de min e max e a Padrão. 
# A última teve melhores resultados. Como é uma série temporal, foi o utilizado o 'shuffle' 
# como 'False' para não embaralhar/misturar os dados.
def DivisaoDtSet_CriaTarget(datset, perdivtreino, treinaNorm):
  
    # Dividindos dados X,y
    y = datset.Realizado
    if treinaNorm:
        X = datset.drop(['UF','RECEITA','Realizado'], axis=1)
    else:
        X = datset.drop(['Realizado'], axis=1)

    # Dividir o DataSet em Treino e Teste
    XtrainNorm, XvalNorm, ytrain, yval = train_test_split(X, y, train_size=perdivtreino,
                                                        random_state=0, shuffle=False)
    #Testando divisão sequencial
    #train_tam = int(len(X)*2/3)
    #XtrainNorm = X[:train_tam]
    #XvalNorm = X[train_tam:] 
    #ytrain = y[:train_tam]
    #yval = y[train_tam:]

    #print(type(XtrainNorm))
    #print(type(ytrain))

    return XtrainNorm, XvalNorm, ytrain, yval


# De acordo com a documentação do 'scikit learn', o processamento do 'StandardScaler()' 
# é uma padronização das features que remove a média escalonando para a uma variãncia 
# de unitária, ou seja, Gaussiano com média 0 e variância 1. As amostras são calculadas por:
# z = (x - u) / s
# Mas este processamento não é adequado para outliers. Segundo a documentação do 'scikit learn', 
# o que mais se ajusta seria o 'robust_scale'
# Dimensione os recursos usando estatísticas robustas para outliers.
# Este Scaler remove a mediana e dimensiona os dados de acordo com o intervalo de quantis (o padrão 
# é IQR: intervalo interquartil). O IQR é o intervalo entre o 1º quartil (25º quantil) e o 3º quartil (75º quantil).
# A centralização e o dimensionamento acontecem independentemente em cada recurso, computando as estatísticas relevantes
# nas amostras do conjunto de treinamento. A mediana e o intervalo interquartil são então armazenados para serem usados 
# ​​em dados posteriores usando o transformmétodo.
# A padronização de um conjunto de dados é um requisito comum para muitos estimadores de aprendizado de máquina. 
# Normalmente, isso é feito removendo a média e escalando para a variância unitária. No entanto, os valores discrepantes 
# podem frequentemente influenciar a média / variância da amostra de forma negativa. Nesses casos, a mediana 
# e o intervalo interquartil fornecem resultados melhores.(disponível em:
#  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)
# A documentação ainda indica
# Não use a robust_scalemenos que você saiba o que está fazendo. Um erro comum é aplicá-lo a todos os dados antes de dividir 
# em conjuntos de treinamento e teste. Isso irá influenciar a avaliação do modelo porque as informações vazaram do conjunto de
#  teste para o conjunto de treinamento. Em geral, recomendamos o uso RobustScalerdentro de um Pipeline , a fim de evitar
#  a maioria dos riscos de dados vazamento: .
# pipe = make_pipeline(RobustScaler(), LogisticRegression()) 
# (disponível em: 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale)
def Normaliza(datanormal, dtestnormal, colunas):

    #dfmdlNorm = Normalizer().fit(datanormal)
    #dfNorm = dfmdlNorm.transform(datanormal)
    #print(type(datanormal))
    #print(type(dtestnormal))

    dtestnormal = pd.DataFrame(dtestnormal)

    # Standard para algoritimo de Gradiente e Redes Neurais
    #stmodel = StandardScaler()
    #stmodel = stmodel.fit(datanormal)
    #dfNorm = stmodel.transform(datanormal)
    #dfTest = stmodel.transform(np.array(dtestnormal))
    dfNorm = RobustScaler(quantile_range=(25, 75)).fit_transform(datanormal)
    dfTest = RobustScaler(quantile_range=(25, 75)).fit_transform(dtestnormal)

    #print(dfNorm)
    #print(type(dfNorm))
    #print(type(dfTest))

    # A normalização teve péssimo desempenho
    #min_max = MinMaxScaler()
    #dfNorm = min_max.fit_transform(datanormal) 

    #Transforma o array em datafreme
    dfNorm = pd.DataFrame(dfNorm, columns=colunas)
    dfTest = np.ravel(dfTest)

    return dfNorm, dfTest


# Ajustando os valores para o PCA
def transfDadosPCA(dtpca):
    colpca = dtpca.columns
    for cl in colpca:
        dtpca[cl] = (dtpca[cl] - dtpca[cl].mean())/dtpca[cl].std()

    return dtpca

# Função para aplicação do PCA, baseado em Ravi [2019]
def Monta_pca(dt, n):
    
    # Guarda as features
    ccolunas = dt.columns
    print("Monta PCA dt : ", dt)
    print("Monta PCA n  : ", n)

    # instancia o modelo
    model = pca(n_components=n, normalize=True)
    results = model.fit_transform(dt)
    #print('PCA ', results.shape)
    
    # Autovetor do componente
    autovetorepca = results['loadings']
    
    ## Ordem das Componentes
    pca_df = results['PC']
    #print (results.shape)
    
    return model, results, pca_df




#  Com esta função ajusta-se o dataset com Receita específica, separam-se as datas e o valor Realizado para ser submetido ao PCA.
def Resultado_PCA(iterc, percreduz, dataPCA):
    
    # Mensurando o tempo de execução
    inicio_exe = time.time()
    
    # retira as colunas alvo e da receitas
    if iterc != 'Todas':
        nvdtreceita = dataPCA[dataPCA['RECEITA']== iterc].copy()
    else:
        nvdtreceita = dataPCA.copy()

    #print(nvdtreceita.info())
    nvdtrecDatas = nvdtreceita['Data_Receita']
    nvdtreceita = nvdtreceita.select_dtypes(include=['float', 'int'])
    # Normaliza o Realizado, pois a função também vai normalizar os dados
    nvdtrecRealizado = transfDadosPCA(pd.DataFrame(nvdtreceita['Realizado']))

    Colunasdt = list(nvdtreceita.drop(['Realizado'], axis='columns'))
    #print('Colunas', Colunasdt)

    nvdtreceita = nvdtreceita[Colunasdt]
    #print(nvdtreceita.shape)

    # Quantidade de componentes reduzida a 95%
    a = 0.9

    # 2. Decomponha a matriz de variâncias e covariâncias em componentes principais.
    # Chama a função
    mPca, resultado, autovalorpca = Monta_pca(nvdtreceita, percreduz)
    
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print('Tempo de execução ', fim_exe - inicio_exe)

    return nvdtrecDatas, nvdtrecRealizado, mPca, resultado, autovalorpca




# Identificar a quantidade de componentes que explicam mais de 90% da variabilidade dos dados
def Identifica_Qnt_Componentes(iterc, resultado, mPca):
    k = resultado['loadings'].shape[0]
    print(iterc + ' explicado com k=', k)

    # Na API não terá apresentação de gráficos
    # mPca.plot(figsize=(16,6))
    # plt.show()

    return k



# Relacionar as features que fazem parte do autovetor das componentes principais.
# Cria um Dataset das features com melhores importância
def Cria_Rankfeatures(resultado):

    ReankFeatures = resultado['topfeat']
    
    # Substituindo valores PC
    ReankFeatures['PC']= ReankFeatures['PC'].replace(['PC1','PC2','PC3','PC4','PC5','PC6','PC7',
                                                      'PC8','PC9'],
                                                     ['PC01','PC02','PC03','PC04','PC05','PC06',
                                                      'PC07','PC08','PC09'])
    
    ReankFeatures['loading']=ReankFeatures['loading']*100

    return ReankFeatures





# Código disponível em https://seaborn.pydata.org/tutorial/categorical.html  
def PCA_plot_Graficos_features(iterc, k, mPca, ReankFeatures):
    
    print('Gráfico de Explicação da Receita ' + iterc)
    mPca.biplot(n_feat=k, legend=False, figsize=(16,10), label=True)
    plt.show()

    sns.set_theme(style="whitegrid", color_codes=True)
    #sns.barplot(x='loading', y='feature', hue='type', col='PC', data=ReankFeatures)
    for i in ReankFeatures['PC'].unique():
        sns.catplot(x="loading", y="feature", hue="type", 
                      data=ReankFeatures[ReankFeatures['PC'].eq(i)].sort_values(["PC", "loading"]), 
                      height=8, kind="bar", palette="muted")
        plt.title('Autovetores de ' + i + ' - Receita ' + iterc)
        plt.show();



# Avaliando as componentes criados pelo PCA
# Apensando ao dataset foi submetido ao modelo de RandomForest
def PCA_Ajuste_Dataset(autovalorpca, nvdtrecDatas, nvdtrecRealizado, perdiv):

    # Ajusta Dataset
    X_nor = autovalorpca
    X_nor['Data_Receita'] = nvdtrecDatas
    X_nor['Realizado'] = nvdtrecRealizado
    #print(X_nor)

    # formatando a Data
    X_nor.Data_Receita = pd.to_datetime(X_nor.Data_Receita, format='%Y-%m-%d')

    # Tranforma o index  de RangeIndex para DatetimeIndex
    X_nor = X_nor.set_index('Data_Receita')
    X_nor.sort_index(inplace=True) 
    
    # Dividindos dados X,y DataSet em Treino e Teste
    XTrain, Xval, y_nor, yval = DivisaoDtSet_CriaTarget(X_nor, perdiv, False)

    return XTrain, Xval, y_nor, yval



def Avaliar_PCA(autovalorpca, nvdtrecDatas, nvdtrecRealizado, ntree, nexenos, perdiv):

    # Ajusta o Dataset
    X_nor, Xval, y_nor, yval = PCA_Ajuste_Dataset(autovalorpca, nvdtrecDatas, nvdtrecRealizado, perdiv)

    # Aplicando ao RandomForest
    mdl_rf = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=nexenos, 
                                    bootstrap=False, random_state=0, n_jobs=-1)
    mdl_rf.fit(X_nor, y_nor)

    pred = mdl_rf.predict(Xval)

    MAE = mean_absolute_error(yval, pred)
    MSE = mean_squared_error(yval, pred)
    RMSE = mean_squared_error(yval, pred, squared=False)
    R2 = r2_score(yval, pred)

    return MAE, MSE, RMSE, R2, pred, yval




# Plotando um gráfico comparativo da predição
def Plota_Avalia_Predicao(mdata, modelo, iterec, metrica):

    plt.figure(figsize=(15, 5))
    x_ax = mdata.index
    y_pred=mdata["prediction"]
    y_orig=mdata["target"]
  
    plt.plot(x_ax, y_orig, label="Original")
    plt.plot(x_ax, y_pred, label="Predição")
    plt.title("Comparativo PrediçaõxOriginal do " + modelo + ' na Receita ' + iterec)
    plt.xlabel('X-Datas' + metrica)
    plt.ylabel('Y-Valores ' + iterec)
    plt.legend(fancybox=True, shadow=True)
    plt.grid(True)
    plt.show();


# Criando o modelo
def Monta_RandomForest(X_nor, y_nor, XvalNorm, yval, ntree, nexenos):

    mdl_rf = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=nexenos, 
                                   bootstrap=False, random_state=0, n_jobs=-1)
    mdl_rf.fit(X_nor, y_nor)

    pred = mdl_rf.predict(XvalNorm)
    
    MAE = mean_absolute_error(yval, pred)
    MSE = mean_squared_error(yval, pred)
    RMSE = mean_squared_error(yval, pred, squared=False)
    R2 = r2_score(yval, pred)

    return mdl_rf, MAE, MSE, RMSE, R2, pred



# Criando o modelo
# Criando o modelo
def Monta_XGBoost(scores_xgb, Xtr, Xval, ytr, yval, tx_lr, ntree, nexenos):

    mdl_exgb = XGBRegressor(learning_rate=tx_lr,
                            n_estimators=ntree,
                            max_depth=6,
                            min_child_weight=1,
                            subsample=0.75,
                            colsample_bynode=nexenos,
                            random_state=0,
                            booster="gbtree",
                            objective='reg:squarederror',
                            importance_type='gain')
    
    mdl_exgb.fit(Xtr, ytr)

    sorted_idx = np.argsort(mdl_exgb.feature_importances_)[::-1]
    features_exgb = []
    
    # guarda as features
    for index in sorted_idx:
        features_exgb.append([Xtr.columns[index], float(mdl_exgb.feature_importances_[index])])
        #Adicionando as featueres a listas
        #print(Xtr.columns[index], mdl_exgb.feature_importances_[index])
        #lstFeatures.append(str(Xtr.columns[index]))
        #lstImportancia.append(float(mdl_exgb.feature_importances_[index]))

    p = mdl_exgb.predict(Xval)

    MAE = mean_absolute_error(yval, p)
    MSE = mean_squared_error(yval, p)
    RMSE = mean_squared_error(yval, p, squared=False)
    R2 = r2_score(yval, p)

    return MAE, MSE, RMSE, R2, features_exgb, p


def Monta_GradientBoosting(X_train, y_train, Xval, yval):
    
    # parametros em comum
    params = {"n_estimators": 500,
              "max_depth": 4,
              "min_samples_split": 5,
              "learning_rate": 0.01,
              "loss": "huber",
             }
    
    gbr_ls = GradientBoostingRegressor(**params)
    mdl_gbr = gbr_ls.fit(X_train, y_train)

    mdl_importancia = mdl_gbr.feature_importances_
    sorted_idx = np.argsort(mdl_importancia)
    features_gbr = []
    
    # guarda as features
    for index in sorted_idx:
        features_gbr.append([X_train.columns[index], mdl_importancia[index]]) 

    p = mdl_gbr.predict(Xval)

    MAE = mean_absolute_error(yval, p)
    MSE = mean_squared_error(yval, p)
    RMSE = mean_squared_error(yval, p, squared=False)
    R2 = r2_score(yval, p)

    return p, MAE, MSE, RMSE, R2, features_gbr



# Cria função para buscar os melhores parâmetros da Floresta Aleatória
def parametros_rf_arvores_amostras(rf_receita, rf_uf):

  dfrf = melhor_rf[(melhor_rf['RECEITA']==rf_receita) & (melhor_rf['UF']==rf_uf)].copy()

  ntree = int("".join([str(i) for i in dfrf['NrArvores']]))
  nexenos = int("".join([str(i) for i in dfrf['QntAmosNo']]))
  
  return ntree, nexenos

def Fit_Modelo(receita, uf, fEnsemble, k, Xtrain2, ytrain):

    #if receita == "FEP":
    #  print('Entrou na função de Fit do Modelo')

    if fEnsemble in "RandomForestRegressor":
        # Busca a quantidade de árvores e de amostras 
        ntree, nexenos = parametros_rf_arvores_amostras(receita, uf)
        
        mdl = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=nexenos,
                                    bootstrap=False, max_features=k, random_state=0, n_jobs=-1)

    if fEnsemble in "XGBRegressor":   
        mdl = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1,
                           max_depth=10, min_child_weight = 5, subsample = 0.6, seed = 42)
    
    if fEnsemble in "GradientBoostingRegressor":   
        mdl = GradientBoostingRegressor(loss='huber', max_depth=3, n_estimators=100, 
                                        learning_rate=0.1, random_state=0, 
                                        max_features=k)

    #if receita == "FEP":
    #  print('Início Fit modelo: ', fEnsemble)

    mdl.fit(Xtrain2, ytrain)

    #if receita == "FEP":
    #  print('Fim Fit modelo: ', fEnsemble)

    return mdl

def Montagem_Modelo(receita, uf, fSelect, nk, ralpha, lalpha, Xtrain, ytrain, XtrainNorm, XvalNorm, Xval):

    #if receita == "FEP":
    #  print('Entrou na função de Montagem do Modelo')

    if fSelect in "KBest":
        selector = SelectKBest(score_func=f_regression, k=nk)
    else:
        if fSelect in "LinearRegression":
            selector_model = LinearRegression()
        if fSelect in "RegressionRidge":
            selector_model = Ridge(alpha = ralpha)
        if fSelect in "RegressionLasso":
            selector_model = Lasso(alpha = lalpha)
        if fSelect in "RandomForestRegressor":
            # Busca a quantidade de árvores e de amostras 
            ntree, nexenos = parametros_rf_arvores_amostras(receita, uf) 

            selector_model = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=nexenos, 
                                                   bootstrap=False, random_state=1, n_jobs=-1 )
        # Cria uma seleção das qnt de k features melhores
        selector = SelectFromModel(selector_model, max_features=nk, threshold=-np.inf)   
        
    # utiliza os valores comuns
    if fSelect in "RandomForestRegressor":
        Xtrain2 = selector.fit_transform(Xtrain, ytrain)
        #print('RandomForestRegressor Xtrain2.shape=', Xtrain2.shape)
        Xval2 = selector.transform(Xval)
        #print('RandomForestRegressor Xval2.shape=', Xval2.shape)
    else:
        Xtrain2 = selector.fit_transform(XtrainNorm, ytrain) 
        Xval2 = selector.transform(XvalNorm)
    
    
    # Guarda os valores das importâncias
    if fSelect in "KBest":
        importancai = selector.score_func(Xtrain2, ytrain)[1]
    else:
        if fSelect in "RandomForestRegressor":
            importancai = selector.estimator_.feature_importances_[selector.get_support()]
        else:
            importancai = selector.estimator_.coef_[selector.get_support()]
        #print(importancai)
    
    
    return selector, Xtrain2, Xval2, importancai

def feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, fSelect, fEnsemble, receita, uf, ralpha, lalpha):

    #if receita == "FEP":
    #  print('Entrou na função de treinamento')
    # seleciona as features em conjuntos, escolhendo no dataset de traino
    # escolhendo as features para testar
    for k in range(2, Xtrain.shape[1]):

        selector, Xtrain2, Xval2, importancai = Montagem_Modelo(receita, uf, fSelect, k, ralpha, lalpha, 
                                                                Xtrain, ytrain, XtrainNorm, XvalNorm, Xval)

        # Verifica na mascara do get_support para selecionar as k melhores
        Xtrain.columns[selector.get_support()]

        mdl = Fit_Modelo(receita, uf, fEnsemble, k, Xtrain2, ytrain)

        MAE, MSE, RMSE, R2 = 0, 0, 0, 0

        p = mdl.predict(Xval2)
        
        #if fSelect in "RandomForestRegressor":
        #    print('RandomForest Xval2.shape=', Xval2.shape)
        #    print('RandomForest yval.shape=', yval.shape)
        
        # Agora podemos calcular o MES
        MAE = mean_absolute_error(yval, p)
        MSE = mean_squared_error(yval, p)
        RMSE = mean_squared_error(yval, p, squared=False)
        R2 = r2_score(yval, p)
 
        #print("Checando o erro com numero de  = features (k)")
        print("k Features {} - R2 {}".format(k,R2), "  MAE  = {}".format(MAE), " RMSE = {}".format(RMSE), "      MSE  = {}".format(MSE))
        
        scores.append([uf, receita,k, np.array(Xtrain.columns[selector.get_support()]), importancai,
                       fSelect, fEnsemble, R2, MAE, MSE, RMSE, np.array(yval), np.array(p)])
    
    return

def trainingAll(dataset, scores, receita, uf, ralpha, lalpha, treinaNorm):

    print('Fase 01: Divide os dados Treino e Teste')
    # Dividindos dados X,y DataSet em Treino e Teste
    # inclui 3 para calculo do trimestre e 70% para treino
    Xtrain, Xval, ytrain, yval = DivisaoDtSet_CriaTarget(dataset, 0.7, treinaNorm)
    print("Dataset: Treino X=", Xtrain.shape, ' y=', ytrain.shape, ' Teste X=' , Xval.shape, ' y=', yval.shape)
    print("--------------------------------------------------") 

    #print(Xtrain.info())
    #print('Type Xtrain', type(Xtrain))
    #print('Type Xval', type(Xval))

    print('Fase 02: Divide os dados normalizar datasets')
    # normalizar datasets treinos, validação e test
    XtrainNorm, ytrain = Normaliza(Xtrain, ytrain, Xtrain.columns)
    XvalNorm, yval = Normaliza(Xval, yval, Xval.columns)
    print("Dataset Normailzado: Treino X=",XtrainNorm.shape,' y=', ytrain.shape,' Teste X=',XvalNorm.shape,' y=', yval.shape)
    print("--------------------------------------------------") 
    #print('Type Xtrain', type(Xtrain))
    #print('Type Xval', type(Xval))
    #print(Xtrain.info())
    
    print('Fase 03: Combinações de Seleção de Features e treinamentos')
    # Mensurando o tempo de execução
    ini_exe = time.time()
    # Combinações de Seleção de Features e treinamentos
    print(" ")
    print(receita, "=> 01_30 KBest vs RandomForestRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "KBest", 
                        "RandomForestRegressor", receita, uf, ralpha, lalpha)
    
    ## usados LinearRegression, RandomForestRegressor, XGBRegressor, GradientBoostingRegressor
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 02_30 KBest vs RandomForestRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 03_30 LinearRegression vs RandomForestRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "LinearRegression", 
                      "RandomForestRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 04_30 LinearRegression vs RandomForestRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 05_30 RegressionRidge vs RandomForestRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RegressionRidge", 
                      "RandomForestRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 06_30 RegressionRidge vs RandomForestRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 07_30 RegressionLasso vs RandomForestRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RegressionLasso", 
                      "RandomForestRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 08_30 RegressionLasso vs RandomForestRegressor Duração=', (fim_exe - ini_exe), 'seg')
     
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 09_30 RandomForestRegressor vs RandomForestRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RandomForestRegressor", 
                      "RandomForestRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 10_30 RandomForestRegressor vs RandomForestRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 11_30 KBest vs XGBRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "KBest", 
                      "XGBRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 12_30 KBest vs XGBRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 13_30 LinearRegression vs XGBRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "LinearRegression", 
                      "XGBRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 14_30 LinearRegression vs XGBRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 15_30 RegressionRidge vs XGBRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RegressionRidge", 
                      "XGBRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 16_30 RegressionRidge vs XGBRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 17_30 RegressionLasso vs XGBRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RegressionLasso", 
                      "XGBRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 18_30 RegressionLasso vs XGBRegressor Duração=', (fim_exe - ini_exe), 'seg')
     
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 19_30 RandomForestRegressor vs XGBRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RandomForestRegressor", 
                      "XGBRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 20_30 RandomForestRegressor vs RandomForestRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 21_30 KBest vs GradientBoostingRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "KBest", 
                      "GradientBoostingRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 22_30 KBest vs GradientBoostingRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 23_30 LinearRegression vs GradientBoostingRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "LinearRegression", 
                      "GradientBoostingRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 24_30 LinearRegression vs GradientBoostingRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 25_30 RegressionRidge vs GradientBoostingRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RegressionRidge", 
                      "GradientBoostingRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 26_30 RegressionRidge vs GradientBoostingRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    print(" ")
    print(receita, "=> 27_30 RegressionLasso vs GradientBoostingRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores, "RegressionLasso", 
                      "GradientBoostingRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 28_30 RegressionLasso vs GradientBoostingRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    # Mensurando o tempo de execução
    ini_exe = time.time()
    # Retirei porque esta com error Verificar separadamente para ver o que é
    print(" ")
    print(receita, "=> 29_30 RandomForestRegressor vs GradientBoostingRegressor")
    print("--------------------------------------------------") 
    feature_select_func(XtrainNorm, XvalNorm, Xtrain, Xval, ytrain, yval, scores,"RandomForestRegressor", 
                        "GradientBoostingRegressor", receita, uf, ralpha, lalpha)
    # Mensurando o tempo de execução
    fim_exe = time.time()
    print(receita, '=> 30_30 RandomForestRegressor vs GradientBoostingRegressor Duração=', (fim_exe - ini_exe), 'seg')
    
    return



#-----------------------------------------------------------------------------------------------------------------------------------------------------
# FUNÇÔES LOCAIS
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Checa se o arquivo ja existe, se nao cria toda estrutura
def checaArquivo(arq):

  # Testa se o arquivo existe
  if os.path.isfile(arq):
    #print("Arquivo: " + arq + " já existe!" )
    # Conecta com o banco
    conn = sqlite3.connect(arq,check_same_thread=False)
    # Cria um cursor de manipulação de dados
    curs = conn.cursor()
  else:
    #print("Criando as tabelas do arquivo")

    # Conecta com o banco
    conn = sqlite3.connect(arq)
    # Cria um cursor de manipulação de dados
    curs = conn.cursor()
    # criando a tabela Pagra
                          
  return conn, curs

def Ufs(uf):

    cidade = ""
    if uf == "AC":
        cidade = "Rio Branco"
    if uf == "AL":	
        cidade = "Maceió"	
    if uf == "AP":
        cidade = "Macapá"
    if uf == "AM":
        cidade = "Manaus"
    if uf == "BA":	
        cidade = "Salvador"
    if uf == "CE":	
        cidade = "Fortaleza"
    if uf == "DF":
        cidade = "Brasília"		
    if uf == "ES":
        cidade = "Vitória"	
    if uf == "GO":
        cidade = "Goiânia"	
    if uf == "MA":	
        cidade = "São Luís"
    if uf == "MT":	
        cidade = "Cuiabá"	
    if uf == "MS":	
        cidade = "Campo Grande"	
    if uf == "MG":	
        cidade = "Belo Horizonte"	
    if uf == "PA":
        cidade = "Belém	Pará"	
    if uf == "PB":	
        cidade = "João Pessoa"	
    if uf == "PR":
        cidade = "Curitiba"	
    if uf == "PE":
        cidade = "Recife"	
    if uf == "PI":
        cidade = "Teresina"		
    if uf == "RJ":
        cidade = "Rio de Janeiro"
    if uf == "RN":
        cidade = "Natal	Rio Grande do Norte"		
    if uf == "RS":
        cidade = "Porto Alegre"
    if uf == "RO":
        cidade = "Porto Velho"	
    if uf == "RR":
        cidade = "Boa Vista"	
    if uf == "SC":
        cidade = "Florianópolis"
    if uf == "SP":
        cidade = "São Paulo"
    if uf == "SE":
        cidade = "Aracaju"	
    if uf == "TO":
        cidade = "Palmas"

    return cidade


# Rtorno de html para retorno no flask
def html(content):  # Also allows you to set your own <head></head> etc
   return '<html><head>custom head stuff here</head><body>' + content + '</body></html>'
