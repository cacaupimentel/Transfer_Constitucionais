# -*- encoding: utf-8 -*-
import os
from datetime import date

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta



## "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?&$format=json&$select=Indicador,Data,DataReferencia,Mediana,baseCalculo"

## "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/
# ExpectativaMercadoMensais
# ?&$format=json
# &$select=Indicador,Data,DataReferencia,Mediana"

def Cria_URL_BCB_olinda(udtini, udtfim, stabela, scampos):
    # String de pesquisa da url
    s = ""
    s = stabela
    s = s + "?&$format=json"
    s = s + "&$select=" + scampos
    s = s + "&dataInicial=" + udtini
    s = s + "&dataFinal=" + udtfim

    # Monta base da consulta
    url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/" + s

    print("Tabela BCB: ", stabela)
    print('------------------------------------------------------------------------------------')
    print("webdriver gets url....")
    print(url)

    return url


### Variáveis globais para trazer apenas as expectativas mais recentes
# Da Data Atual, subtrai 2 meses colocando, pois o banco publica sempre depois de 2 meses
mardata = (date.today() + relativedelta(years=4)).replace(day=1)
maiordata = date.strftime(mardata, format="%m/%d/%Y")
print('Maior Data: ', maiordata)

# Esta data deve ficar fixa pois é a menor que a consulta disponibiliza "MM/dd/yyyy"
meodata = (date.today() + relativedelta(months=-1)).replace(day=1)
menordata = date.strftime(meodata, format="%m/%d/%Y")
print('Menor Data: ', menordata)

caminho = "C:/Users/cacau/Documents/MESTRADO/DISSERTACAO/Dados_Abertos"

# Pega o Caminho já definido
# caminho = dataset['Caminho'][0]

# Seleciona o caminho do Arquivo
basedir = caminho + "/ArquivosParaUnificar/"

# Nome do arquivo
nomearq = "f_FOCUS.csv"

# cria um diretório para armazenar o arquivo comparâmetro para não criar se já existir
os.makedirs(os.path.abspath(os.path.dirname(basedir)), exist_ok=True)

basedir = os.path.abspath(os.path.dirname(basedir))

# Apaga o arquivo se foi criado
if os.path.isfile(os.path.join(basedir, nomearq)):
    # Se existe apaga o arquivo
    os.remove(os.path.join(basedir, nomearq))

# Abre o arquivo csv

# | Nome | Tipo | Título | Descrição |
# |------|------|--------|-----------|
# | Indicador | texto | Indicador | Câmbio/ IGP-DI / IGP-M / INPC / IPA-DI / IPA-M / IPCA / IPCA Administrados /
#                                   IPCA Alimentação no domicílio / IPCA Bens industrializados / IPCA Livres /
#                                   IPCA Serviços / IPCA-15 / IPC-Fipe / Produção industrial / Taxa de desocupação |
# | Data | texto | Data | Data do cálculo da estatística |
# | DataReferencia | texto | Data de Referência | Data de referência para qual a estatística é esperada |
# | Media | decimal | Média | Média das expectativas fornecidas pelas instituições credenciadas |
# | Mediana | decimal | Mediana | Mediana das expectativas fornecidas pelas instituições credenciadas |
# | DesvioPadrao | decimal | Desvio Padrão | Desvio padrão das expectativas fornecidas pelas instituições credenciadas |
# | Minimo | decimal | Mínimo | Mínimo das expectativas fornecidas pelas instituições credenciadas |
# | Maximo | decimal | Máximo | Máximo das expectativas fornecidas pelas instituições credenciadas |
# | numeroRespondentes | inteiro | Número de Respondentes | Número de instituições credenciadas que forneceram suas expectativas |
# | baseCalculo | inteiro | Base de Cálculo | Base de cálculo para as estatísticas baseada no prazo de validade das expectativas
#                                             informadas pelas instituições informantes:
#                                              - 0: uso das expectativas mais recentes informadas pelas instituições participantes
#                                              a partir do 30º dia anterior à data de cálculo das estatísticas
#                                              - 1: uso das expectativas mais recentes informadas pelas instituições participantes
#                                              a partir do 4º dia útil anterior à data de cálculo das estatísticas |

## Documentação disponível em: https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/documentacao#ExpectativasMercadoSelic


i = 0

Registros = []

# Garda na variável a resposta para o método GET do site
# resposta = requests.get("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?&$format=json&$select=Indicador,Data,DataReferencia,Mediana")
urls = Cria_URL_BCB_olinda(menordata, maiordata, "ExpectativaMercadoMensais",
                           "Indicador,Data,DataReferencia,Mediana,baseCalculo")

resposta = requests.get(urls)
arq_json = resposta.json()

obj = dict(arq_json)
values = obj.get("value")

for l in values:

    object = dict(l)
    Data = l.get("Data")
    indicador = l.get("Indicador")
    DataReferencia = l.get("DataReferencia")
    Mediana = l.get("Mediana")
    if l.get("baseCalculo") != 0:
        baseCalculo = "4 dias"
    else:
        baseCalculo = "30 dias"

    # Adiciona a lista
    Registros.append([Data, indicador, DataReferencia, Mediana, baseCalculo])
    print(Data, indicador, DataReferencia, Mediana, baseCalculo)

# Criando o dataset
dataset = pd.DataFrame(Registros, columns=['Data', 'Idicador', 'DataReferencia', 'Mediana', 'BaseCalculo'])

# Salva o arquivo na Pasta de unificação
dataset[['Data', 'Idicador', 'DataReferencia', 'Mediana', 'BaseCalculo']].to_csv(os.path.join(basedir, nomearq), index=False, sep=';')
