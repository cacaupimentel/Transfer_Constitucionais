# -*- encoding: utf-8 -*-
# import time
import requests
import pandas
from datetime import datetime
import os

dadospowerbi = {'Caminho': ['C://Users/47127279349/ownCloud/SEMFAZ/MESTRADO/DISSERTACAO/Dados_Abertos'],
                'DataIni': ['2010-01-01'],
                'DataFim': ['2022-12-31']}

# Simula o dataset do Power BI
dataset = pandas.DataFrame(dadospowerbi)

# print(datetime.now())
# Lista para guardas os dados 
dataset_list = []
# Aqui são as variáveis que temos que pegar
list_cabe = ["idMuni", "ano", "Mes", "fonte", "nomefonte", "categoria", "nomecategoria", "origem",
             "nomeorigem", "especie", "nomeespecie", "rubrica", "nomerubrica", "alinea",
             "nomealinea", "subalinea", "nomesubalinea", "prevista", "arrecadada"]

# Cria variável para caminho de armazenamento dos arquivos
caminho = dataset['Caminho'][0]
basedir = caminho + "/ReceitasPublicas/Capitais/CampoGrande/DADOS/"

# cria um diretório para armazenar o arquivo comparâmetro para não criar se já existir
os.makedirs(os.path.abspath(os.path.dirname(basedir)), exist_ok=True)

# Nome do arquivo
nomearq = "campo_grande.csv"
# print(nomearq)

# Separa o Ano Atual
AnoAtual = datetime.now().year - 1
# print(AnoAtual)

# Identifica menorData
d_dataini = dataset['DataIni'][0]
# print(d_dataini)

# Identifica a maiorData
d_datafim = dataset['DataFim'][0]
# print(d_datafim)

# Verifica se o arquivo já existe
if os.path.isfile(os.path.join(basedir, nomearq)):
    # Se existe abre e verifica qual a maior data para baixar os últimos 2 anos
    # abre no dataset
    dataset = pandas.read_csv(os.path.join(basedir, nomearq), sep=';')

    # Transforma o type para numerico no ano
    dataset['ano'] = pandas.to_numeric(dataset['ano'], errors='coerce')

    # Limita o arquivo
    dataset = dataset[dataset.ano < AnoAtual]

    # Salva o arquivo com menos 2 anos anos atuais
    dataset.to_csv(os.path.join(basedir, nomearq), index=False, sep=';')

    # Guarda a data para iniciar a busca dos novos dados
    strAnoAtual = str(AnoAtual) + "-01-01"
    print('Data Inicial: ', strAnoAtual)

    # Cria o intervalo de datas
    datesStart = pandas.date_range(start=strAnoAtual, end=d_datafim, freq='MS')
    datesEnd = pandas.date_range(start=strAnoAtual, end=d_datafim, freq='M')

    # Abro o arquivo para incluir os novos valores
    arq_csv = open(os.path.join(basedir, nomearq), "a", encoding="utf-8")

else:
    # Caso contrário cria um novo arquivo csv
    arq_csv = open(os.path.join(basedir, nomearq), "w", encoding="utf-8")
    tmp = "idMuni;ano;Mes;fonte;nomefonte;categoria;nomecategoria;origem;nomeorigem;especie;nomeespecie;rubrica;nomerubrica" \
          ";alinea;nomealinea;subalinea;nomesubalinea;prevista;arrecadada "
    arq_csv.write(tmp + '\n')

    # Cria o intervalo de datas
    datesStart = pandas.date_range(start=d_dataini, end=d_datafim, freq='MS')
    datesEnd = pandas.date_range(start=d_dataini, end=d_datafim, freq='M')


# https://data-export.campogrande.ms.gov.br/api/transparencia/receita/lista/
# ?consulta_post=receitas&draw=1
# &ano=2013
# &receita=1
# &data-inicial=01%2F01%2F2013
# &data-final=31%2F01%2F2013
# &orgao=33
# &categoria=1&file_type=csv

# https://data-export.campogrande.ms.gov.br/api/transparencia/receita/lista/?consulta_post=receitas&draw=1&ano=2013&receita=1&data-inicial=01%2F01%2F2013&data-final=31%2F01%2F2013&orgao=33&categoria=1&file_type=csv


def cria_url(uano, udtini, udtfim):
    # String de pesquisa da url
    s = "?consulta_post=receitas&draw=1"
    s = s + "&ano=" + uano
    s = s + "&receita=1"
    s = s + "&data-inicial=" + udtini
    s = s + "&data-final=" + udtfim
    s = s + "&orgao=33&file_type=csv"

    # Monta base da consulta
    curl = "https://data-export.campogrande.ms.gov.br/api/transparencia/receita/lista/" + s

    print("Inicio: ", dtInicio)
    print("Fim   : ", dtFim)
    print('------------------------------------------------------------------------------------')
    print("webdriver gets url....")
    print(curl)

    return curl


def retira_espacos(stexto):
    sajuste = stexto.strip()
    # remove os textos duplicados
    " ".join(sajuste.split())

    return sajuste


# Verifica a quantidade de caracteres para padronizar em 12 dígitos
def qnt_digitos(stexto, ntam):
    # remove os espaços em branco
    stexto = stexto.strip()
    if len(stexto) > ntam:
        resultado = stexto[:ntam]
        print("Reduzindo tamanho Código: ", stexto, str(len(stexto)), resultado)
    else:
        resultado = stexto + ("0" * (ntam - len(stexto)))
        print("Aumentando tamanho Código: ", stexto, str(len(stexto)), resultado)

    return resultado


# Função para ler os arquivos das urls
def ler_arq_url(lcodmuni, lu, lmes):
    # Garda na variável a resposta para o método GET do site
    resposta = requests.get(lu)
    arq_json = resposta.json()
    # Transfora em dicionário
    lobjeto = dict(arq_json)

    # Varrendo a collection
    for key, value in lobjeto.items():
        if key == "data":
            dadosur = value

            for k in dadosur:
                tmp = ""
                list_campos = []
                list_campos.append(lcodmuni)
                list_campos.append(k.get("ano"))
                list_campos.append(lmes)
                list_campos.append(k.get("fonte"))
                list_campos.append(k.get("nomefonte"))
                list_campos.append(qnt_digitos(str(k.get("categoria")), 12))
                list_campos.append(retira_espacos(k.get("nomecategoria")))
                list_campos.append(qnt_digitos(str(k.get("origem")), 12))
                list_campos.append(retira_espacos(k.get("nomeorigem")))
                list_campos.append(qnt_digitos(str(k.get("especie")), 12))
                list_campos.append(retira_espacos(k.get("nomeespecie")))
                list_campos.append(qnt_digitos(str(k.get("rubrica")), 12))
                list_campos.append(retira_espacos(k.get("nomerubrica")))
                list_campos.append(qnt_digitos(str(k.get("alinea")), 12))
                list_campos.append(retira_espacos(k.get("nomealinea")))
                list_campos.append(qnt_digitos(str(k.get("subalinea")), 12))
                list_campos.append(retira_espacos(k.get("nomesubalinea")))
                list_campos.append(k.get("prevista"))
                list_campos.append(k.get("arrecadada"))

                tmp = '; '.join(map(str, list_campos))
                # print(tmp)

                arq_csv.write(tmp + '\n')

                dataset_list.append(list_campos)

    # print('Lista completa: ')
    # print(dataset_list)
    return dataset_list


# faz o laço para buscar os arquivos
i = 0
while i < len(datesStart):
    #  Converte as datas
    dtInicio = datesStart[i].strftime("%d/%m/%Y")
    dtFim = datesEnd[i].strftime("%d/%m/%Y")

    # Datas para Url
    data_inicio = str(datesStart[i].strftime("%d")) + '%2F' + str(datesStart[i].strftime("%m")) + '%2F' + str(
        datesStart[i].strftime("%Y"))
    data_final = str(datesEnd[i].strftime("%d")) + '%2F' + str(datesEnd[i].strftime("%m")) + '%2F' + str(
        datesEnd[i].strftime("%Y"))

    # cria string da url da API
    url = cria_url(datesStart[i].strftime("%Y"), data_inicio, data_final)

    # Consome a API lendo o arquivo json criado
    dataset_list = ler_arq_url("5002704", url, str(datesStart[i].strftime("%m")))

    # Incrementa a contagem das datas
    i = i + 1

arq_csv.write('\n')
# Fecha o WebBrowser
arq_csv.close()

dataset = pandas.DataFrame(dataset_list, columns=list_cabe)
