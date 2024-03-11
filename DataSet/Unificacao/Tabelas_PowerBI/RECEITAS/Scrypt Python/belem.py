# -*- encoding: utf-8 -*-
#import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from webdriver_manager.chrome import ChromeDriverManager
import selenium.webdriver.support.ui as ui
from selenium.common.exceptions import *
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
# Import Select class
from selenium.webdriver.support.ui import Select

from bs4 import BeautifulSoup
import html.parser 
from collections import OrderedDict
import pandas as pd
import numpy as np
#from itertools import tee, repeat, zip_longest
from datetime import datetime
import time


# Verifica se a pagina est carregada
def pag_loading(sdriver):
    while True:
    sdx = sdriver.execute_script("return document.readyState")
    if sdx != "complete":
        yield False
    else:
        break
    return


def page_is_loading(driver):
    while True:
        x = driver.execute_script("return document.readyState")
        if x == "complete":
            return True
        else:
            yield False


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



#-------------------------------------------------------------------------------------
# INICIANDO O PROCESSO 
#-------------------------------------------------------------------------------------

options = Options()
options.add_argument("start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('--disable-blink-features=AutomationControlled')
#options.add_argument('--headless')
options.add_argument("--disable-xss-auditor")
options.add_argument("--disable-web-security")
options.add_argument("--allow-running-insecure-content")
options.add_argument("--no-sandbox")
options.add_argument("--disable-setuid-sandbox")
options.add_argument("--disable-webgl")
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")
options.add_argument('--disable-logging')
# Passa a Url para WebDriver Selenium e carrega o google Chromme com a pagina
# Iniciando o driver
driver = webdriver.Chrome(ChromeDriverManager().install(),chrome_options=options)
  
# Wait for the page to load
driver.implicitly_wait(15)

url1 = "http://portaltransparencia.belem.pa.gov.br/receita/receita-detalhada/"


# Cria o intervalo de datas
datesStart = pd.date_range(start='2010-01-01', end='2022-09-01', freq = 'MS')
datesEnd = pd.date_range(start='2010-01-30', end='2022-09-01', freq = 'M')

# Abre o arquivo csv
#arq_csv = open("Belem.csv","w",encoding="utf-8")
#tmp = ""
#tmp = tmp + "coodigoIBGE," + "mes," + "ano," + "descricao," + "fonte,"
#tmp = tmp + "aplicacao," + "valorOrcadoAtualizado," + "valorOrcado,"
#tmp = tmp + "valorArrecadado"

##arq_csv.write(tmp + '\n')

i = 0

#while i<len(datesStart):

driver.get(url1)

#      Datas para Url
mes  = int( datesStart[i].strftime("%m") ) 
ano  = int(datesStart[i].strftime("%Y"))

ano_index = 0
ano_ini = 2022
# Indice do ano
ano_index = ano_ini - ano - 1
print(ano_index)

time.sleep(1)

# Seleciona valor do ano
xpath1 = "//select[contains(@id,'ctl00_Content_ddl_nrAno')]"
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH , xpath1)))
driver.execute_script("""arguments[0].click();""", element)
driver.execute_script("""arguments[0].index = arguments[1];""", element, ano_index) 



# conta datsas
i = i + 1


# fecha csv
#arq_csv.close()
# fecha driver
#driver.close