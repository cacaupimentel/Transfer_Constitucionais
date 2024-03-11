# -*- encoding: utf-8 -*-
#import time

from bs4 import BeautifulSoup
import html.parser 
from collections import OrderedDict
import pandas as pd
import numpy as np
#from itertools import tee, repeat, zip_longest
from datetime import datetime
import os
import shutil
import sys

# import openpyxl module
# pip install PyPDF2
from PyPDF2 import PdfReader
import re
import pandas as pd
from os import walk
from os import listdir
from os.path import isfile, join
from unidecode import unidecode

# pip install unidecode
# pip3 install openpyxl 
# pip install pytest-shutil
# pip install xlrd

# Abre a tabela de municipios
dfMunicipios = pd.read_csv("municipios.csv")
# transforma os nomes do municipios em lower case
dfMunicipios["nome"] = dfMunicipios["nome"].map(lambda x: x.lower())
# tira a ascentuação
dfMunicipios["nome"] = dfMunicipios["nome"].apply(unidecode)


# Abre o dataset de erros de municipios
dfErros = pd.read_csv("erros.csv")


# lista dos IPIs
Registros = []

basedir = os.path.abspath(os.path.dirname("C:/tmp/claudia/webscrap_receitas/datasets/IPM/")) 




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
      if a[i:i+1] != b[i:i+1]:
          # print("Diferente")
          igual = False
          break

  return igual


def Estado_sigla(uf):

    estado = ""
    
    if isEqual("IPM_Alagoas",uf):
        estado = "AL"
    if isEqual("IPM_Bahia",uf):
        estado = "BA"
    if isEqual("IPM_Ceará",uf):
        estado = "CE"
    if isEqual("IPM_MatoGrosso",uf):
        estado = "MT"
    if isEqual("IPM_Minas Gerais",uf):
        estado = "MG"
    if isEqual("IPM_Pará",uf):
        estado = "PA"
    if isEqual("IPM_Tocantins",uf):
        estado = "TO"

    return estado



# Abre o arquivo csv
#arq_csv = open("Belem.csv","w",encoding="utf-8")
#tmp = ""
#tmp = tmp + "coodigoIBGE," + "mes," + "ano," + "descricao," + "fonte,"
#tmp = tmp + "aplicacao," + "valorOrcadoAtualizado," + "valorOrcado,"
#tmp = tmp + "valorArrecadado"

##arq_csv.write(tmp + '\n')

# lista Pastas
#onlyfiles = [f for f in listdir(basedir) if isfile(join(basedir, f))]

# Pega a lista de pastas
list_Folders = [f for f in listdir(basedir) if not isfile(join(basedir, f))]

# Varre as pastas
for f in list_Folders:
    print(" ")
    print(" ")
    print("Pasta: ",f)
    print("------------------------------------------------------------------------------------------------------------")

    #Acha o estado pelo nome da pasta
    UF = Estado_sigla(f)


    # Monta o path dos locais dos arquivos
    files_path = join(basedir, f) 
    files_path = join(files_path , "Arquivos_Leitura_pdf")
    

    #Checa se existe diretorio  "Arquivos_Leitura_pdf" dentro desta pasta
    if not os.path.exists( files_path):
        print("        Não existe pasta \Arquivos_Leitura_pdf em ",join(basedir, f) )
        print(" ")
        continue

    
    # Peha a lista de arquivos
    lst_files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    # Varre os arquivos desta parsta
    for a in lst_files:
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #Teste temporario
        if  not isEqual( "Portaria GSEF nº 558-2018 - Republicação IPM - Judicial Boca da Mata - DOE-AL 11-07-18.pdf",a):
            continue
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print("        Arquivo: ",a)
        print(" ")

        arq_path = join(files_path, a)

        flgAdd = False
        #Converter para texto e União de toda as paginas
        reader = PdfReader(arq_path)

        arq_text = ""
        pg_tmp = ""
        text = ""

        for i in range(len(reader.pages)):
            page = reader.pages[i]
            pg_tmp = text 
            text = page.extract_text()
            text = bytes(text, "utf-8")
            text = text.decode(encoding='utf-8', errors='strict')
        
            print(text)

            # Testa se começa a incluir texto   
            if ( ("ANO BASE" in text) and ("APLICAÇÃO" in text)  and ("DA PORTARI" in text)   and   ("ANEXO" in text)  ) :  
                flgAdd = True 

            # Testa se grava
            if flgAdd:
                # se as páginas são difeentes
                if pg_tmp != text: 
                    arq_text = arq_text + text
                    #print(text)

            # Testa se para de incluir texto
            if "TOTAL" in text:
                flgAdd = False 
                break
        # Texto
        #print(arq_text)


        # extraindo os dados
        flgRead = False
        linhas = arq_text.split("\n")
        lastWord = ""
        for l in linhas:

            #print(l)

            # Testa se Achou a coluna
            if ("FINAL" in l) or ("IND.FINAL" in l): # or (colunas3 in l) or (colunas4 in l):
                #print("Achou texto inicio")
                flgRead = True
                continue

            # Testa se Achou TOTAL
            if "TOTAL" in l:
                #print("Achou texto final")
                break

            # Lê colunas da linha
            if flgRead:
                

                #Garante que não tenha duplo espaço antes do percentual

                tmp = l.lstrip().rstrip()
                tmp = tmp.replace("  "," ")

                #print(tmp)

                # Lista de sentenças que precisam ser eliminada
                # por estarem mal formaatadas ou grudadas em numeroos
                tmp = tmp.replace("Edição Eletrônica Certificada Digitalmente","")
                tmp = re.sub('Edição Eletrônica Certificada Digitalmente', '', tmp)
                tmp = re.sub('Diário OficialEstado de Alagoasconforme LEI N° /', '', tmp)
                tmp = re.sub('Diário OficialEstado de Alagoasconforme LEI N°/', '', tmp)
                tmp = re.sub('Diário OficialEstado de Alagoasconforme LEI N° / ', '', tmp)
                tmp = tmp.replace("Diário OficialEstado de Alagoasconforme LEI N° /SAO MIGUEL DOS MILAGRES","SAO MIGUEL DOS MILAGRES")
                tmp = re.sub('ANEXO ÚNICO', '', tmp)
                tmp = tmp.replace(" -","")
                tmp = tmp.replace("Maceió - segunda-feira","")
                tmp = tmp.replace("Maceió segunda-feira","")
                tmp = tmp.replace("Maceió - segunda-feira","")
                tmp = tmp.replace("Maceió - terça-feira","")
                tmp = re.sub('Maceió - quarta-feira', '', tmp)
                tmp = tmp.replace("Maceió - quarta-feira","")
                tmp = tmp.replace("Maceió - quarta-feira","")
                tmp = tmp.replace("Maceió quarta-feira","")
                tmp = re.sub('Maceió quinta-feira', '', tmp)
                tmp = tmp.replace("Maceió - sexta-feira","")
                tmp = re.sub('/g[0-9][0-9][0-9]', '', tmp)
                tmp = re.sub('/g[0-9][0-9]', '', tmp)
                tmp = re.sub('/g[0-9]', '', tmp)
                tmp = re.sub('/g', '', tmp)
                tmp = tmp.lstrip().rstrip()

                # separa e conta as colunas
                colunas = tmp.split(" ")
                numColunas = len(colunas) 

                #  O nome dos municipios veem em linhas separadas
                # Se na linhas tem menos de 4 palaveas, provavelmente é parte do nome
                if  numColunas < 5:
                    # Guarda a parte do nome do municipio
                    #lastWord = tmp
                    lastWord = lastWord + tmp
                    #print("privious line: ", l)

                else:
                    # Testa se tem palavra anterior
                    if len(lastWord)>0:
                        municipio = lastWord + " " + tmp
                        lastWord = "" 
                    else:
                        municipio = tmp
                    
                    # filtra os sinais
                    municipio = municipio.replace(".","").replace(",","")
                    
                    # removendo os numeros e deixando somento o nome do municipio
                    municipio = re.sub('[0-9]', '', municipio)
                    municipio = municipio.lstrip().rstrip()
                    percentual = str(colunas[numColunas-1])

                    # tira o formato do numero
                    percentual = colunas[numColunas-1].replace(".","").replace(",",".")
                    # removendo o nome do municipo e deixando os numeros apenas
                    tmp = tmp.replace(tmp,"") 


                    # procura o  municipio na tabela de municiios
                    ErroMunicipio = dfErros[ dfErros['errado'] == municipio ].values
                    if len(ErroMunicipio) > 0:
                        #Pega o campo certo
                        municipio = ErroMunicipio[0][1]
                        #print(municipio)

                    # procura o  municipio na tabela de municiios
                    Mun = dfMunicipios[ ( (dfMunicipios['nome'] == municipio.lower()) &  (dfMunicipios['uf'] ==  UF) )].values
                    #Mun = dfMunicipios['id'][ dfMunicipios['nome'] == municipio.lower() ]
                    #print(Mun)
                    
                    # Se achou o municipio
                    if len(Mun) > 0:
                        
                        CodigoIBG = Mun[0][0]
                        Municipio = Mun[0][1]
                        #print( "Codigo IBGE: ",CodigoIBG )
                        #print( "Nome IBGE  : ",Municipio )
                        #print(" ")
                        #print(municipio + " > " + percentual)

                        #print(" ")
                        #print(l)

                        #print("------------------------------------------------------------------------------------------------------------")
                        #print(" ")
                    else:
                        print("Não achou:    ")
                        print(municipio)
                        print(l)
                        print("------------------------------------------------------------------------------------------------------------")
                        print(" ")
        break

    sys.exit()