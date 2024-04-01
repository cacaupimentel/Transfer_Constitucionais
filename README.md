
# Transfer_Constitucionais
Construção de dataset para Seleção de variáveis usando abordagens de filtro com wrapper, com aplicação de algorítmos de Aprendizagem de Máquina 



## 


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/) 
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

## 
## Autores

- [@cacaupimentel](https:/www.github.com/cacaupimentel)
- [@jacobjr](https://www.github.com/jacobjr)



## Apêndice

Considera-se a previsão de receitas de grande relevância para os tomadores de decisão, bem como para o planejamento. Quando se trata do campo de aplicação prática, voltado para o Setor Público, no que tange a esfera municipal, observam-se distorções entre os valores orçados e previsto, mesmo aplicando as regras previstas na legislação vigente. A questão tem sido investigada por pesquisadores com uma trajetória de avanços de métodos de regressão estatísticos e as aplicações de técnicas de aprendizagem de máquina, mas a problemática das divergências das previsões continua e a legislação exige justificativas. Neste contexto, faz-se necessário investigar se os efeitos preço e quantidade podem ser identificados por técnicas de aprendizagem de máquina e os erros de previsão das receitas poderiam ser mitigados se as variáveis fossem usadas pelo regime de competência do ingresso de recursos. Neste sentido, esta pesquisa tem o objetivo de realizar um estudo de caso, com os dados de São Luís, para escolher as variáveis que atendam as prerrogativas legais, adotando a metodologia CRISP-DM, por meio da comparação da lista de importância de algoritmos ensembles, Random Forests, Gradient Boosting e XGBoost, com um modelo combinado das abordagens de filtro com wrapper, submetendo-os aos mesmos algoritmos para escolher as variáveis com menores métricas de avaliação dentro de uma sequência de menores erros das receitas transferidas. No documento foram relatadas, como um comparativo de execução, as etapas e tarefas do CRISP-DM em sua primeira iteração, utilizando os dados dos Portais da Transparência, no período de 2010 a 2021. Nos resultados, comparou-se dois conjuntos de dados, um com todos os repasses, incluindo os valores extraordinários e outro com apena as cotas oficiais. O Modelo Combinado obteve, na maioria dos resultados, as melhores métricas, especialmente, nos repasses extraordinários, corroborando com o estado da arte que já consagra esta abordagem, mas a aplicação do teste de Friedman não descartou a hipótese nula, pois as métricas dos dois conjuntos não apresentaram diferenças significativas. Na modelagem a RNN foi complexa obteve a melhor métrica, todavia, com exceção dos recursos da Mineração, a diferença dos valores foi melhor em outros algoritmos e o teste de Fridman também não teve diferenças significativas. Como resposta a questão de pesquisa foi possível identificar com clareza o efeito quantidade nos dois conjuntos de dados, mas o do preço não foi tão evidente nos resultados, aparecendo mais quando se testou apenas as cotas oficiais.


## Licença

[MIT](https://choosealicense.com/licenses/mit/)


## Referência

 - [PIMENTEL, Cláudia Patrícia Silva. Aplicação de técnicas de aprendizagem de máquina com seleção de variáveis na previsão de receitas públicas de capitais brasileiras: estudo de caso das receitas transferidas de São Luís. 2023. 148f. (Dissertação).Mestrado Profissional em Engenharia da Computação e Sistemas, Centro de Ciências e Tecnológicas, Universidade Estadual do Maranhão, São Luís, 2023. ](https://repositorio.uema.br/handle/123456789/2471?mode=full)
