# -*- encoding: utf-8 -*-
# import time
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
import pandas


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


# -------------------------------------------------------------------------------------
# INICIANDO O PROCESSO 
# -------------------------------------------------------------------------------------

options = Options()
options.add_argument("start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('--disable-blink-features=AutomationControlled')
# options.add_argument('--headless')
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
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

# Wait for the page to load
driver.implicitly_wait(15)

url1 = "https://saoluis.giap.com.br/apex/saoluis/f?p=839:3"

# Cria o intervalo de datas
# datesStart = pd.date_range(start='2010-01-01', end='2022-09-01', freq = 'MS')
# datesEnd = pd.date_range(start='2010-01-30', end='2022-09-01', freq = 'M')

datesStart = pandas.date_range(start='2015-01-01', end='2022-09-01', freq='MS')
datesEnd = pandas.date_range(start='2015-01-30', end='2022-09-01', freq='M')

# Abre o arquivo csv
arq_csv = open("sao_luis.csv", "w", encoding="utf-8")
tmp = ""
tmp = tmp + "coodigoIBGE," + "mes," + "ano," + "descricao," + "fonte,"
tmp = tmp + "aplicacao," + "valorOrcadoAtualizado," + "valorOrcado,"
tmp = tmp + "valorArrecadado"

arq_csv.write(tmp + '\n')

i = 0

while i < len(datesStart):

    driver.get(url1)

    #      Datas para Url
    mes = int(datesStart[i].strftime("%m"))
    ano = int(datesStart[i].strftime("%Y"))

    ano_index = 0
    ano_ini = 2009
    # Indice do ano
    ano_index = ano - ano_ini

    # Seleciona valor do ano
    categoria = driver.find_element(By.XPATH, "//select[@id='GLOBAL_EXERCICIO']")
    drop = Select(categoria)
    drop.select_by_index(ano_index)
    # time.sleep(1)

    # Seleciona indice do mes iicio
    # Find id of option                    #  Ocorrendo error aqui. Achando 2 elementos com este nome
    #categoria = driver.find_element_by_id('P3_MES_INICIAL')
    xpath1 = "//*[@id='P3_MES_INICIAL']"
    categoria = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH , xpath1)))
    drop = Select(categoria)
    drop.select_by_index(mes)

    # Seleciona indice do mes fim
    xpath1 = "//*[@id='P3_MES_FINAL']"
    categoria = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH , xpath1)))
    #categoria = driver.find_element_by_id('P3_MES_FINAL')cls

    drop = Select(categoria)
    drop.select_by_index(mes)

    # Seleciona indice do camp data
    xpath1 = "//*[@id='P3_QUEBRA']"
    categoria = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH , xpath1)))
    #categoria = driver.find_element_by_id('P3_QUEBRA')
    drop = Select(categoria)
    drop.select_by_index(3)

    # Seleciona indice do camp orçamento
    xpath1 = "//*[@id='P3_TIPO_RECEITA']"
    categoria = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH , xpath1)))
    #categoria = driver.find_element_by_id('P3_TIPO_RECEITA')
    drop = Select(categoria)
    drop.select_by_index(1)


    try:
        elem = driver.find_element_by_xpath("//button[@id='B133250412938450781']")
        elem.click()
    except:
        pass

    print('ok..')

    # Extraindo o conteudo da pagina decodificada
    content = driver.page_source.encode('utf-8').strip()

    # Passando para o Beautifulsoup como html
    soup = BeautifulSoup(content, 'html.parser')

    # //*[@id="report_relatorio_pagina_3"]
    html = soup.find("table", {"id": "report_relatorio_pagina_3"}).findAll("td")

    cont = 1

    tmp = ""
    coodigoIBGE = ""
    descricao = ""
    fonte = ""
    aplicacao = ""
    valorOrcadoAtualizado = ""
    valorOrcado = ""
    valorArrecadado = ""
    for td in html:
        # print(td)

        # Se achar o total para o laço
        if "Total:" in td.text:
            break

        if td == tmp:
            continue

        if cont == 1:
            descricao = td.text
            tmp = td

        if cont == 2:
            aplicacao = td.text
            tmp = td

        if cont == 3:
            fonte = td.text
            tmp = td

        if cont == 4:
            valorOrcadoAtualizado = td.text
            valorOrcadoAtualizado = valorOrcadoAtualizado.lstrip()
            valorOrcadoAtualizado = valorOrcadoAtualizado.rstrip()
            valorOrcadoAtualizado = valorOrcadoAtualizado.replace(".", "")
            valorOrcadoAtualizado = valorOrcadoAtualizado.replace(",", ".")
            print(td.text)
            tmp = td

        if cont == 5:
            valorOrcado = td.text
            valorOrcado = valorOrcado.lstrip()
            valorOrcado = valorOrcado.rstrip()
            valorOrcado = valorOrcado.replace(".", "")
            valorOrcado = valorOrcado.replace(",", ".")
            tmp = td

        if cont == 6:
            valorArrecadado = td.text
            valorArrecadado = valorArrecadado.lstrip()
            valorArrecadado = valorArrecadado.rstrip()
            valorArrecadado = valorArrecadado.replace(".", "")
            valorArrecadado = valorArrecadado.replace(",", ".")
            tmp = td

            print('----------------------------------------------------------')
            print(descricao)
            print(fonte)
            print(aplicacao)
            print(valorOrcadoAtualizado)
            print(valorOrcado)
            print(valorArrecadado)
            # Grava arquivo
            tmp = ""
            tmp = tmp + "2111300,"
            tmp = tmp + str(mes) + ","
            tmp = tmp + str(ano) + ","
            tmp = tmp + descricao + ","
            tmp = tmp + fonte + ","
            tmp = tmp + aplicacao + ","
            tmp = tmp + valorOrcadoAtualizado + ","
            tmp = tmp + valorOrcado + ","
            tmp = tmp + valorArrecadado + ','

            coodigoIBGE = ""
            descricao = ""
            fonte = ""
            aplicacao = ""
            valorOrcadoAtualizado = ""
            valorOrcado = ""
            valorArrecadado = ""
            arq_csv.write(tmp + '\n')
            cont = 0

        cont = cont + 1

    # conta datsas
    i = i + 1

# fecha csv
arq_csv.close()
# fecha driver
driver.close
