# -*- encoding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import os
import pandas
import time

# Simula o dataset do Power BI
ddpwerbi = {'Caminho': ['C://Users/47127279349/ownCloud/SEMFAZ/MESTRADO/DISSERTACAO/Dados_Abertos'],
            'DataIni': ['2005-01-01'],
            'DataFim': ['2023-01-18']}

# Simula o dataset do Power BI
dataset = pandas.DataFrame(ddpwerbi)


# Verifica se a pagina est carregada
def pag_loading(sdriver):
    while True:
        dx = sdriver.execute_script("return document.readyState")
        if dx == "complete":
            break
        else:
            yield False
    return


def page_is_loading(pdriver):
    while True:
        x = pdriver.execute_script("return document.readyState")
        if x != "complete":
            yield False
        else:
            return True


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
driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

# Wait for the page to load
driver.implicitly_wait(15)

url1 = "https://www42.bb.com.br/portalbb/daf/beneficiario,802,4647,4652,0,1.bbx"

# Cria variável para caminho de armazenamento dos arquivos
caminho = dataset['Caminho'][0]
basedir = caminho + "/ReceitasPublicas/DARF_BB/"
print('Caminho do Arquivo: ', basedir)

# cria um diretório para armazenar o arquivo comparâmetro para não criar se já existir
os.makedirs(os.path.abspath(os.path.dirname(basedir)), exist_ok=True)

# Nome do arquivo
nomearq = "BB.csv"
print('Nome do arquivo: ', nomearq)

# Separa o Ano Atual
AnoAtual = datetime.now().year - 1
print('Ano Atual: ', AnoAtual)

# Identifica menorData
d_dataini = dataset['DataIni'][0]
print('Data de Inicio: ', d_dataini)

# Identifica a maiorData
d_datafim = dataset['DataFim'][0]
print('Data Fim: ', d_datafim)

# Calcula dois meses a mais para a data final
d_dtfims = (date.today() + relativedelta(months=1))
d_dtfims = date.strftime(d_dtfims, "%Y-%m-%d")
print('Data Final atual: ', d_dtfims)

# Cria o intervalo de datas
# Mas antes testa se já existe no arquivo

if os.path.isfile(os.path.join(basedir, nomearq)):
    # Se existe abre e verifica qual a maior data para baixar os últimos 2 anos
    # abre no dataset
    dataset = pandas.read_csv(os.path.join(basedir, nomearq), sep=',', encoding='UTF-8')
    print(dataset['datadeposito'])

    # Transforma o type para data ajustando o formato para o do arquivo
    dataset['datadeposito'] = pandas.to_datetime(dataset['datadeposito'], errors='ignore', format='%d/%m/%Y')

    # Limita o arquivo
    dataset = dataset[dataset['datadeposito'].year < AnoAtual]

    # Salva o arquivo com menos 2 anos anos atuais
    dataset.to_csv(os.path.join(basedir, nomearq), sep=',', index=False)

    # Guarda a data para iniciar a busca dos novos dados
    strAnoAtual = str(AnoAtual) + "-01-01"
    print('Data Inicial Arquivo existente: ', strAnoAtual)

    # Cria o intervalo de datas
    datesStart = pandas.date_range(start=strAnoAtual, end=d_datafim, freq='MS')

    # A data final tem que ter 2 meses a mais
    datesEnd = pandas.date_range(start=strAnoAtual, end=d_dtfims, freq='M')

    # Abro o arquivo para incluir os novos valores
    arq_csv = open(os.path.join(basedir, nomearq), "a", encoding="utf-8")

else:
    # Caso contrário cria um novo arquivo csv
    arq_csv = open(os.path.join(basedir, nomearq), "w", encoding="utf-8")

    # Abre o arquivo csv
    # arq_csv = open("BB.csv", "w", encoding="utf-8")
    tmp = "cidade, receita, datadeposito, parcela, valordistribuido"
    arq_csv.write(tmp + '\n')

    # Cria o intervalo de datas
    datesStart = pandas.date_range(start=d_dataini, end=d_datafim, freq='MS')
    datesEnd = pandas.date_range(start=d_dataini, end=d_dtfims, freq='M')

    # datesStart = pandas.date_range(start='2010-01-01', end='2022-11-15', freq='MS')
    # datesEnd = pandas.date_range(start='2010-01-30', end='2022-12-15', freq='M')

cidades_lista = ["BELEM", "BELO HORIZONTE", "CAMPO GRANDE", "FORTALEZA", "MACEIO", "PALMAS", "SALVADOR", "SAO LUIS",
                 'SANTAREM']

for lcidade in cidades_lista:

    i = 0
    print("CIDADE: ", lcidade)
    while i < len(datesStart):

        print("Lendo url principal")
        driver.get(url1)

        #      Datas para Url
        mes = int(datesStart[i].strftime("%m"))
        ano = int(datesStart[i].strftime("%Y"))

        ano_index = 0
        ano_ini = 2022
        # Indice do ano

        # time.sleep(1)

        #  Converte as datas
        dtInicio = datesStart[i].strftime("%d/%m/%Y")
        dtFim = datesEnd[i].strftime("%d/%m/%Y")
        print("Data Inicio: ", dtInicio)
        print("Data Fim   : ", dtFim)

        # Seleciona a Capital
        print("Settig Cidade..")
        xpath1 = "//*[contains(@id,'formulario:txtBenef')]"
        element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
        driver.execute_script("""arguments[0].value = arguments[1];""", element, lcidade)

        # Clica em continuar
        print("Click button Pesquisa..")
        xpath1 = "//*[contains(@name,'formulario:j_id16')]"
        element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
        driver.execute_script("""arguments[0].click();""", element)

        if "FORTALEZA" in lcidade:
            print("Settig Cidade Index..")
            xpath1 = "//*[contains(@id,'comboBeneficiario')]"
            element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
            # driver.execute_script("""arguments[0].index = arguments[1];""", element, 1)
            select_fr = Select(element)
            select_fr.select_by_index(1)

        if "CAMPO GRANDE" in lcidade:
            print("Settig Cidade Index..")
            xpath1 = "//*[contains(@id,'comboBeneficiario')]"
            element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
            # driver.execute_script("""arguments[0].index = arguments[1];""", element, 1)
            select_fr = Select(element)
            select_fr.select_by_index(2)

        if "PALMAS" in lcidade:
            print("Settig Cidade Index..")
            xpath1 = "//*[contains(@id,'comboBeneficiario')]"
            element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
            # driver.execute_script("""arguments[0].index = arguments[1];""", element, 1)
            select_fr = Select(element)
            select_fr.select_by_index(2)

        # Coloca a data inicial
        print("Settig Data Inicio..")
        xpath1 = "//*[contains(@id,'formulario:dataInicial')]"
        element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
        driver.execute_script("""arguments[0].value = arguments[1];""", element, dtInicio)

        # Coloca a data final
        print("Settig Data Fim..")
        xpath1 = "//*[contains(@id,'formulario:dataFinal')]"
        element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
        driver.execute_script("""arguments[0].value = arguments[1];""", element, dtFim)

        # Coloca a data final
        print("Click Button Pesquisar..")
        xpath1 = "//*[contains(@name,'formulario:j_id20')]"
        element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, xpath1)))
        driver.execute_script("""arguments[0].click();""", element)

        time.sleep(1)

        print("Lendo Html da cidade", lcidade)
        print(
            "---------------------------------------------------------------------------------------------------------")

        # Extraindo o conteudo da pagina decodificada
        content = driver.page_source.encode('utf-8').strip()

        # Passando para o Beautifulsoup como html
        soup = BeautifulSoup(content, 'html.parser')

        # Grava o html do Formulario
        html = soup.find("table", {"id": "formulario:demonstrativoList"}).find_all("td")
        # print(html)

        cont = 1

        # Variáveis
        cidade = ""
        receita = ""
        datadeposito = ""
        parcela = ""
        valordistribuido = ""
        lastData = ""

        # Flag do total
        flgTotais = False
        flgProximaReceita = False
        flagGrava = True

        # Ler a tabela do html
        for td in html:

            tdtexto = td.text.rstrip().lstrip()

            # flag sinaliza se grava ou não o registro
            if "TOTAL DOS REPASSES NO PERIODO" in tdtexto:
                print("                   ---------> FIM DA PAGINA ")
                break

            if len(receita) == 0:
                # Pula linhas em branco
                if len(tdtexto) == 0:
                    continue
            # else:
            #     receita = td.text
            #     #print('receita = ', receita)
            #     continue

            # Guarda a cidade
            if len(cidade) == 0:
                cidade = tdtexto
                # print('cidade = ', cidade)
                continue

            # Guarda a receita
            if len(receita) == 0:
                receita = tdtexto
                # print('receita = ', receita)
                continue
            # CREDITO FUNDO

            # print(tdtexto)
            # print("---------------------------------------------------------------")

            # Pula o cabeçalho
            if tdtexto in ['DATA', 'PARCELA', 'VALOR DISTRIBUIDO']:
                continue

            # Le sequencia dos campos
            if len(receita) > 0:

                # Checa o total
                if "TOTAL:" in tdtexto:
                    print("                   ---------> Achou TOTAL:   tdtexto ", tdtexto)
                    flagGrava = False

                # Mão grava nada até achar 'CREDITO FUNDO'
                if "TOTAIS" in tdtexto:
                    print("                   ---------> Achou TOTAIS:  tdtexto", tdtexto)
                    flgTotais = True

                if "CREDITO FUNDO" in tdtexto:
                    print("                   ---------> Achou CREDITO FUNDO:   tdtexto", tdtexto)
                    flgProximaReceita = True

                # Campo daya
                if cont == 1:
                    if len(tdtexto) > 0:
                        datadeposito = tdtexto.replace(".", "/")
                    else:
                        datadeposito = lastData

                if cont == 2:
                    parcela = tdtexto

                if cont == 3:
                    tipo = tdtexto
                    tipo = tdtexto[len(tdtexto) - 1: len(tdtexto)]
                    tdtexto = tdtexto.replace("R$ ", "")
                    tdtexto = tdtexto.rstrip().lstrip()
                    if "D" == tipo.rstrip():
                        valordistribuido = "-" + tdtexto
                    else:
                        valordistribuido = tdtexto

                    valordistribuido = valordistribuido[:len(valordistribuido) - 2]
                    valordistribuido = valordistribuido.replace(".", "").replace(",", ".")

                    cont = 0

                    if flgTotais:
                        flagGrava = False

                    # testa as codições para gravar ou não            
                    if len(parcela + valordistribuido) == 0:
                        # não grava
                        flagGrava = False

                    # Se achou total não grava nada 
                    if flagGrava:
                        # print("Tipo: ", tipo)
                        print("cidade: ", cidade)
                        print("receita: ", receita)
                        print("datadeposito: ", datadeposito)
                        print("parcela: ", parcela)
                        print("valordistribuido: ", valordistribuido)
                        print("---------------------------------------------------------------")
                        # Grava Arquivo
                        tmp = ""
                        tmp = tmp + lcidade + ","
                        tmp = tmp + receita + ","
                        tmp = tmp + datadeposito + ","
                        tmp = tmp + parcela + ","
                        tmp = tmp + valordistribuido
                        arq_csv.write(tmp + '\n')

                    if len(tdtexto) > 0:
                        lastData = datadeposito
                    datadeposito = ""
                    parcela = ""
                    valordistribuido = ""

                    # Atualiza o flag para false
                    flagGrava = True

                    if flgProximaReceita:
                        # Zera a receita
                        receita = ""
                        # Libera o flag e Indica que saiu da zona dos TOTAIS
                        flgTotais = False
                        # Libera o flag e indica que saiu dos totais
                        flgProximaReceita = False

                cont = cont + 1

        # conta datsas
        i = i + 1

# fecha csv
arq_csv.close()

# fecha driver
driver.close()

# Inclui o id_Muni
bsdirmun = caminho + "\ArquivosParaUnificar"
muniearq = "\d_Capitais_ESTUDO.csv"

if os.path.isfile(os.path.join(bsdirmun, muniearq)):
    # Se existe abre para mesclar os ids
    # abre o dataset com os dados migrados
    dataset = pandas.read_csv(os.path.join(basedir, nomearq), sep=';')

    # abre o aruivo dos municipios estudados
    dtmuni = pandas.read_csv(os.path.join(bsdirmun, muniearq), sep=';')

    # Renomeia a coluna para ficarem iguais
    dtmuni.rename(columns={'MunicipioMaiusculo': 'cidade'}, inplace=True)

    # Mescla os dois arquivos no nome do município
    dataset = pandas.merge(dataset.copy(), dtmuni[['cidade', 'id_Muni']].copy(), how="left", on="cidade")

    # Salva o arquivo com menos 2 anos anos atuais
    dataset.to_csv(os.path.join(basedir, nomearq), index=False, sep=';')
