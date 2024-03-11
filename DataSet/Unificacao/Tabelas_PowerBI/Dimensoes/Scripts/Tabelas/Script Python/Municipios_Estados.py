# -*- encoding: utf-8 -*-
# import time
from bs4 import BeautifulSoup
import html.parser
from collections import OrderedDict
import pandas as pd
import numpy as np
# from itertools import tee, repeat, zip_longest
from datetime import datetime
import time
import os
import requests


# Funçao de comparação de strings
def isEqual(a, b):
    # Premissa de que sejam iguais
    igual = True

    # Primeiro obtem tamanhos
    a = str(a)
    b = str(b)

    lena = len(a)
    lenb = len(b)
    if lena > lenb or lenb > lena:
        igual = False

    # print('len a', lenA)
    # print('len b', lenB)
    # se passo dos comandos testar
    if igual:
        # ja que os tamanhos são iguais vamos verrer caracter
        # por caracter pelo tamnho de A
        for i in range(lena):
            # print(a[i:i+1], " = ", b[i:i+1])
            if a[i:i + 1] != b[i:i + 1]:
                # print("Diferente")
                igual = False
                break

    return igual


##-------------------------------------------------------------------------------------------------------
## Lista de Municipios IBGE
##-------------------------------------------------------------------------------------------------------

# Garda na variável a resposta para o método GET do site
resposta = requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/municipios?orderBy=nome")
arq_json = resposta.json()

# Transfora em dicionário
objeto = dict()
# objeto = dict(arq_json)

lst_MunicipiosIBGE = []
lst_Estados = []

# Varrendo a collection
for k in arq_json:
    objeto = dict(k)
    # Pega Id e nome
    id = objeto.get("id")
    nome = objeto.get("nome")
    # Extrai regiao imediata
    regiaoA = dict(objeto.get("regiao-imediata"))
    # Extrai Região intermediaria
    regiaoB = dict(regiaoA.get("regiao-intermediaria"))
    #  Pega dados do Estado
    regiaoC = dict(regiaoB.get("UF"))

    # pega os campos do estado
    idEstado = regiaoC.get("id")
    nomeEstado = regiaoC.get("nome")
    siglaEstado = regiaoC.get("sigla")
    # print(id, nome, idEstado, siglaEstado, nomeEstado )

    # Prepara os campos do estado
    lst_tmp = []
    lst_tmp.append(idEstado)
    lst_tmp.append(nomeEstado)
    lst_tmp.append(siglaEstado)
    # checa se ja existe na lista de estados
    if lst_tmp not in lst_Estados:
        lst_Estados.append(lst_tmp)

    # Prepara os campos de
    lst_tmp = []
    # Pega campos Id, nome, sigla, estado
    lst_tmp.append(id)
    lst_tmp.append(nome)
    lst_tmp.append(siglaEstado)
    lst_tmp.append(idEstado)
    # Adiciona na lista de Municipioos
    lst_MunicipiosIBGE.append(lst_tmp)

# cria data frames para pesquisas
dfEstado = pd.DataFrame(lst_Estados, columns=["id", "nome", "sigla"])
dfMunicipio = pd.DataFrame(lst_MunicipiosIBGE, columns=["id", "nome", "uf", "id_estado"])

dfEstado.to_csv("estados.csv", index=False)
dfMunicipio.to_csv("municipios.csv", index=False)
