# -*- encoding: utf-8 -*-
#import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import *
from selenium.webdriver.chrome.options import Options
# Import Select class

#from itertools import tee, repeat, zip_longest
import time

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

url1 = "https://app.powerbi.com/view?r=eyJrIjoiYjE1ZDQzNTAtNTUxMC00MTc2LWEyMTEtZjdkZjRlZjk4YzUyIiwidCI6IjNlYzkyOTY5LTVhNTEtNGYxOC04YWM5LWVmOThmYmFmYTk3OCJ9"




# Abre o arquivo csv
arq_csv = open("CONFAZ_ICMS.csv","w",encoding="utf-8")
tmp = ""
tmp = tmp + "Ano,"
tmp = tmp + "Data,"
tmp = tmp + "Estado,"
tmp = tmp + "Subitem ICMS," 
tmp = tmp + "Arrecadacao"

arq_csv.write(tmp + '\n')

i = 0

#Codigo IBGE '2927408'

driver.get(url1)

time.sleep(4)

# Setas - clicar 5x
print('Click button (1x)..')
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='embedWrapperID']/div[2]/logo-bar/div/div/div/logo-bar-navigation/span/button[2]")))
driver.execute_script("""arguments[0].click();""", element) 

print('Click button (2x)..')
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='embedWrapperID']/div[2]/logo-bar/div/div/div/logo-bar-navigation/span/button[2]")))
driver.execute_script("""arguments[0].click();""", element) 

print('Click button (3x)..')
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='embedWrapperID']/div[2]/logo-bar/div/div/div/logo-bar-navigation/span/button[2]")))
driver.execute_script("""arguments[0].click();""", element) 

print('Click button (4x)..')
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='embedWrapperID']/div[2]/logo-bar/div/div/div/logo-bar-navigation/span/button[2]")))
driver.execute_script("""arguments[0].click();""", element) 

print('Click button (5x)..')
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='embedWrapperID']/div[2]/logo-bar/div/div/div/logo-bar-navigation/span/button[2]")))
driver.execute_script("""arguments[0].click();""", element) 

# Clicar nas opções
print('Clica Opções Estados')
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[4]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div/i")))
driver.execute_script("""arguments[0].click();""", element) 
time.sleep(1)

#  Desmarcar todos os estads
print('Desmarcar todos os estads')
driver.execute_script("""document.getElementsByClassName("glyphicon checkbox checkboxOutline")['0'].click();""") 
time.sleep(3)


lst_Detalhamento_Icms = []
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[1]/div/span")
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[2]/div/span")
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[3]/div/span")
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[4]/div/span")
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[5]/div/span" )
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[6]/div/span" )
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[7]/div/span" )
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[8]/div/span" )
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[8]/div/span" )
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[9]/div/span" )
lst_Detalhamento_Icms.append( "//*[@id='slicer-dropdown-popup-72305621-91bc-51e6-297c-feb258e60ea5']/div[1]/div/div[2]/div/div[1]/div/div/div[10]/div/span" )


#Estado
# Loop de rolagems do dropList
inicio = 1
fim = 11
inicioDetalhes = 13
fimDetalhes = 22
flgDetalheICMS = True


#Acha alguns valores
Ano = ""
# Clicar nas opções
element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[3]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div/div")))
Ano = element.text

for l in range(3):
  


  # Sincroniza o proximo item da lista apos rolagem
  if l == 0:
    inicio = 1
    fim = 11
  if l == 1:
    inicio = 1
    fim = 8
  if l == 2:
    inicio = 1
    fim = 11

  print("Inicio: ", inicio, "  e  fim: ", fim)
  print("----------------------------------------------------------------------------------------------------------------------------")
  # Loop de leitura dos itens visíveis
  for i in range(inicio,fim):

    # Selecina o Estado
    tmp =   """document.getElementsByClassName("glyphicon checkbox checkboxOutline")['""" +  str(i) +  """'].click();"""
    print(tmp)
    driver.execute_script(tmp) 
    time.sleep(5)



    #Acessa Detalhamento do ICMS
    # Clicar nas opções
    print('Clica Opções Detalhamento de ICMS')
    element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[6]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div/i")))
    driver.execute_script("""arguments[0].click();""", element) 
    time.sleep(1)




    # Garante que desmarque a opção todos uma unica vez
    if flgDetalheICMS:
      # Clica as opções de ICMS
      print('Zera Opções de detalhamento de ICMS')
      driver.execute_script("""document.getElementsByClassName("glyphicon checkbox checkboxOutline")['12'].click();""", element) 
      #time.sleep(1)
      # Desarma o flag para executar apenas 1 vez
      flgDetalheICMS = False 


    #Acha alguns valores
    Estado = ""
    element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[4]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div/div")))
    Estado = element.text



    # Alguns estados tem detalhes a menos

    if Estado == "Piauí":
      inicioDetalhes = 12
      fimDetalhes = 21
    if Estado == "Rio de Janeiro":
      fimDetalhes = 20
    if Estado == "Rio de Janeiro":
      fimDetalhes = 21
    


    # Varre o detalhamento de ICMS
    for k in range(inicioDetalhes,fimDetalhes):

      print('Opções de detalhamento de ICMS', k)
      tmp2 =   """document.getElementsByClassName("glyphicon checkbox checkboxOutline")['""" +  str(k) +  """'].click();"""
      print("             ----------------------> " +      tmp2)
      driver.execute_script( tmp2, element) 
      time.sleep(1)

      #Pega os outros valores
      SubitemICMS = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[6]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div/div")))
      SubitemICMS = element.text


      Jan = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[1]/div[3]")))
      Jan = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Fev = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[2]/div[3]")))
      Fev = element.text.replace("R$ ","").replace(".","").replace(",",".")


      Mar = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[3]/div[3]")))
      Mar = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Abr = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[4]/div[3]")))
      Abr = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Mai = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[5]/div[3]")))
      Mai = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Jun = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[6]/div[3]")))
      Jun = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Jul = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[7]/div[3]")))
      Jul = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Ago = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[8]/div[3]")))
      Ago = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Set = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[8]/div[3]")))
      Set = element.text.replace("R$ ","").replace(".","").replace(",",".")

      Out = ""
      element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[15]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div[1]/div[4]/div/div[9]/div[3]")))
      Out = element.text.replace("R$ ","").replace(".","").replace(",",".")


      print('Ano        : ', Ano)
      print('Estados    : ', Estado)
      print('SubitemICMS: ',SubitemICMS)
      print('Jan        : ', Jan)
      print('Fev        : ', Fev)
      print('Mar        : ', Mar)
      print('Abr        : ', Abr)
      print('Mai        : ', Mai)
      print('Jun        : ', Jun)
      print('Jul        : ', Jul)
      print('Ago        : ', Ago)
      print('Set        : ', Set)
      print('Out        : ', Out)

      # Grava Registros para cada mês
      tmp = ""
      Data = "15/01/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Jan
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/02/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Fev
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/03/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Mar
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/04/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Abr
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/05/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Mai
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/06/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Jun 
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/07/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Jul
      # Grava n arquivo
      arq_csv.write(tmp + '\n')

      tmp = ""
      Data = "15/08/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Ago 
      # Grava n arquivo
      arq_csv.write(tmp + '\n') 

      tmp = ""
      Data = "15/09/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Set 
      # Grava n arquivo
      arq_csv.write(tmp + '\n')
  
      tmp = ""
      Data = "15/10/" +str(Ano) 
      tmp = tmp + str(Ano) + ","  + Data + "," + Estado + "," + SubitemICMS + "," + Out 
      # Grava n arquivo
      arq_csv.write(tmp + '\n')


   
   
    # Clicar nas opções
    print('Clica Opções Detalhamento de ICMS')
    element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='pvExplorationHost']/div/div/exploration/div/explore-canvas/div/div[2]/div/div[2]/div[2]/visual-container-repeat/visual-container[6]/transform/div/div[2]/div/visual-modern/div/div/div[2]/div/i")))
    driver.execute_script("""arguments[0].click();""", element) 
    time.sleep(1)

  
  # Rola a terra
  print('Rola a janela dropList')
  driver.execute_script("""document.getElementsByClassName("glyphicon checkbox checkboxOutline")['10'].scrollIntoView();""") 
  time.sleep(1)

  # document.getElementsByClassName("slicerItemContainer")
  # document.getElementsByClassName("dropdown-chevron powervisuals-glyph chevron-up")['1'];





# fecha csv
arq_csv.close()
# fecha driver
driver.close
