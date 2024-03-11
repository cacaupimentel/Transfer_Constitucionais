# -*- encoding: utf-8 -*-
#import time
from bs4 import BeautifulSoup
import html.parser 
from collections import OrderedDict
import pandas as pd
import numpy as np
#from itertools import tee, repeat, zip_longest
from datetime import datetime
import time
import os
import requests


# Verifica se a pagina est carregada
def pag_loading(sdriver):
  while True:
    dx = sdriver.execute_script("return document.readyState")
    if dx == "complete":
      break
    else:
      yield False
  return 

def page_is_loading(driver):
    while True:
        x = driver.execute_script("return document.readyState")
        if x == "complete":
            return True
        else:
            yield False

def searchList(lista, valor):
    municipio = ""
    uf = ""
    for l in lista:
      if l[0] == valor:
        uf = l[1]
        municipio = l[2]
        

    return uf,municipio

basedir = os.path.abspath(os.path.dirname("C://tmp/claudia/webscrap_receitas/datasets/")) 

#df = pd.read_csv('datasets/Cessoes_de_Direitos.csv' )

# Mudando o nome das colunas
#df.rename(columns={'CPF CNPJ do titular': 'CPFCNPJ', 'Municipio(s)': 'Municipio', 'Substância(s)': 'Substancia'}, inplace=True)
#df.rename(columns={'Tipo de requerimento': 'TipoRequerimento', 'Fase Atual': 'FaseAtual', 'Tipo(s) de Uso': 'TipoUso'}, inplace=True)
#df.rename(columns={'Situação': 'Situaçao', 'Data da Cessão': 'DataCessao'}, inplace=True)
#df = df[['CPFCNPJ','Titular','Municipio','Substancia']]

# Converte as colunas em string
#df['CPFCNPJ'] = df['CPFCNPJ'].astype(str)
#df['Titular'] = df['Titular'].astype(str)
#df['Municipio'] = df['Municipio'].astype(str)
#df['Substancia'] = df['Substancia'].astype(str)
#df = df[['CPFCNPJ','Titular','Municipio','Substancia']]

# Retira mascara do CPFCNJ
#df['CPFCNPJ'] = df['CPFCNPJ'].replace('.','')
#df['CPFCNPJ'] = df['CPFCNPJ'].replace('-','')
#df['CPFCNPJ'] = df['CPFCNPJ'].replace('/','')
#df['CPFCNPJ'] = df['CPFCNPJ'].replace('*','')
#Elimina os repetidos
#df = df.drop_duplicates(['CPFCNPJ','Titular','Municipio','Substancia'], keep=False)

list_CNPJ = []



i = 0

# Abre o arquivo csv
CFEM_csv = open(os.path.join(basedir, 'CFEM.csv'),"w" ,encoding="utf-8")
tmp = "Ano,Mes,CpfCnpj,Substancia,Uf,Municipio,ValorRecolhido"
CFEM_csv.write(tmp + '\n')

# Abre o arquivo csv
arq_csv = open(os.path.join(basedir, 'CFEM_Arrecadacao.csv'),"r")  # ,encoding="utf-8"
tmp = "Ano, Mês, Processo, AnoDoProcesso, Tipo_PF_PJ, CPF_CNPJ, Substância, UF, Município, QuantidadeComercializada, UnidadeDeMedida, ValorRecolhido"


Ano = ""
Mês  = ""
Processo  = ""
AnoDoProcesso  = ""
Tipo_PF_PJ  = ""
CPF_CNPJ  = "" 
Substância  = "" 
UF  = ""
Município  = ""
QuantidadeComercializada  = ""
UnidadeDeMedida  = ""
ValorRecolhido  = ""


#  Testa se tem logs
if os.path.exists(os.path.join(basedir, 'log_CNPJ')):
  # Abre o arquivo csv
  arq_log = open(os.path.join(basedir,  'log_CNPJ'),"r")  # ,encoding="utf-8"
  # lê o log e joga na lista de CNPJ
  for l in arq_log:
    vl = l.split(",")
    # pega os campos
    cnpj = vl[0]     
    uf = vl[1].replace("\n","")
    municipio  = vl[2].replace("\n","")   
    #adiciona na list de cnpj
    list_CNPJ.append([cnpj,uf,municipio])
  arq_log.close()
  print


cont = 0
valor = 0.0
for l in arq_csv:
  vl = l.split(";")

  Ano = vl[0].replace('"','')     
  Mes  = vl[1].replace('"','')     
  Processo  = vl[2].replace('"','')     
  AnoDoProcesso  = vl[3].replace('"','')     
  Tipo_PF_PJ  = vl[4].replace('"','')     
  CPF_CNPJ = CPF_CNPJ.rstrip().lstrip()
  CPF_CNPJ  = vl[5].replace('"','').replace('*','')      
  Substancia  = vl[6].replace('"','')      
  UF  = vl[7].replace('"','')     
  Municipio  = vl[8].replace('"','')     
  QuantidadeComercializada  = vl[9].replace('"','')     
  UnidadeDeMedida  = vl[10].replace('"','')     
  ValorRecolhido  = vl[11].replace('"','').replace("\n","").replace(".","").replace(",",".")      

  if i == 0:
    i = i + 1
    continue

  if int(Ano) < 2010:
    i = i + 1
    continue

  if len(CPF_CNPJ) == 0:
    i = i + 1
    continue

  Municipio = Municipio.lstrip().rstrip()
  UF = UF.lstrip().rstrip()

  if (len(Municipio) == 0 ):
    
    objUF, objMUNI  = searchList(list_CNPJ, CPF_CNPJ) 
    
    # Testa se o CNPJ já existe na lista
    if len(objMUNI) > 0:

        print(objMUNI)
        print(objUF)
        Municipio = objMUNI
        UF = objUF
        print("CNPJ já pesquisado: ", CPF_CNPJ)
        print(cont)
        print('------------------------------------------------------------------------------------------------------------------------------------')
        cont = cont + 1

    else:

      print(Ano + ", " +  Mes + ", " +  Tipo_PF_PJ + ", " +  CPF_CNPJ + ", " +  Substancia + ", " +  UF + ", " +  Município + ", " +  QuantidadeComercializada + ", " +  UnidadeDeMedida + ", " +  ValorRecolhido)
      # Não achouu nenhum na tabela de consceções
      #MUNICIPIO =  df[  df['CPFCNPJ'] == str(CPF_CNPJ)   ].values
      
      #procurar na API CNPJ
      response = requests.get("https://www.receitaws.com.br/v1/cnpj/"+CPF_CNPJ+"/")
      print(response.status_code)
      obj = dict(response.json())
      objMUNI = str(obj.get("municipio"))
      objUF = str(obj.get("uf"))
      if len(objMUNI) >0:
        print("Nova pesquisa do CNPJ: ", CPF_CNPJ)
        print(objMUNI)
        print(objUF)
      
        Municipio = objMUNI.replace("\n","")
        UF = objUF.replace("\n","")
        list_CNPJ.append([CPF_CNPJ,UF,Municipio])

        cont = cont + 1
        print(cont)
        print("pause...")
        time.sleep(20)
        print("continuing...")
        print('------------------------------------------------------------------------------------------------------------------------------------')

        #grava log
        #Log CNPJ
        log_CNPJ_csv = open(os.path.join(basedir, 'log_CNPJ'),"w")
        #tmp = "Ano,Mes,CpfCnpj,Substancia,Uf,Municipio,ValorRecolhido"
        for ll in list_CNPJ:
            tmp = ll[0] + "," + ll[1] + "," + ll[2].replace("\n","")
            log_CNPJ_csv.write(tmp + '\n')
        log_CNPJ_csv.close()

      else:
        i = i + 1
        continue

  # Gravando os dados com CPF e municipios recuperados
  tmp = ""
  tmp = tmp + Ano + ","
  tmp = tmp + Mes + ","
  tmp = tmp + CPF_CNPJ + ","
  tmp = tmp + Substancia + ","
  tmp = tmp + UF + ","
  tmp = tmp + Municipio + ","
  tmp = tmp + ValorRecolhido 
  CFEM_csv.write(tmp + '\n')



  i = i + 1

print(cont)
print(valor)

# conta datsas


# fecha csv
arq_csv.close()


