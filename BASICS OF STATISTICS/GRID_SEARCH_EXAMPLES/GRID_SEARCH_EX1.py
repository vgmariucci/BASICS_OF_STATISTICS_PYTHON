###########################################################################################################
#
# A maioria dos modelos de aprendizado de máquina contém parâmetros que podem ser ajustados para variar 
# como o modelo aprende. Por exemplo, o modelo de regressão logística, do sklearn, possui um parâmetro C 
# que controla a regularização, o que afeta a complexidade do modelo.
#
# Como escolhemos o melhor valor para C? 
# O melhor valor depende dos dados usados para treinar o modelo.
#
#############################################################################################################

# Como funciona?
# 
# Um método é experimentar valores diferentes e depois escolher o valor que dá a melhor pontuação. 
# Essa técnica é conhecida como pesquisa em grade (Grid Search). Se tivéssemos que selecionar os valores 
# para dois ou mais parâmetros, avaliaríamos todas as combinações dos conjuntos de valores formando assim 
# uma grade de valores.

# Antes de entrarmos no exemplo, é bom saber o que o parâmetro que estamos alterando faz. 
# 
# Valores mais altos de C informam ao modelo que os dados de treinamento se assemelham a informações 
# do mundo real, colocam um peso maior nos dados de treinamento. Enquanto valores mais baixos de C 
# fazem o oposto.

# Primeiro vamos ver que tipo de resultados podemos gerar sem uma pesquisa de grade usando apenas 
# os parâmetros básicos. Para começar, devemos primeiro carregar o conjunto de dados com o qual 
# estaremos trabalhando.

# Importando as bibliotecas
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


########################################################################
#
#    EXPLORANDO A EXISTÊNCIA DE ALGUMA RELAÇÃO ENTRE OS DADOS 
#
########################################################################

# A seguir, para criar o modelo precisamos de um conjunto de variáveis independentes X 
# e uma variável dependente y
iris = datasets.load_iris()

X = iris['data']
y = iris['target']


print(X)
print(y)


# Usaremos o método de regressão  logística para classificar os dados

########################################################################
#
#                CONSTRUÇÃO DO MODELO DE REGRESSÃO LOGÍSITICA
#
########################################################################

# Nessa primeira etapa iremos escolher o valor máximo de iterações para garantir que processo de regressão
# consiga obter um resultado aceitável.
# Lembrando que o valor padrão para o parâmetro C na regressão logística é 1, pois vamos comparar com outros
# valore mais adiante.

# Neste exemplo, o objetivo é analisar os valores dos dados de iris e tentar treinar o modelo com diferente 
# valores do parâmetro C para a regressão logística.

modelo_de_regressao_logistica = LogisticRegression(max_iter = 10000)

# Após criar o modelo, podemos tentar ajustar o mesmo aos dados reais
print(modelo_de_regressao_logistica.fit(X,y))

print(modelo_de_regressao_logistica.score(X,y))

# Com a configuração padrão de C = 1, alcançamos uma pontuação de 0.973.
# Vamos ver se podemos fazer melhor implementando uma pesquisa de grade com valores diferentes de 0.973.

# Implementando a pesquisa de grade

# Seguiremos os mesmos passos de antes, exceto que desta vez definiremos um intervalo de valores para C.
# Saber quais valores definir para os parâmetros pesquisados exigirá uma combinação de conhecimento e prática do domínio.
# Como o valor padrão para C é 1, definiremos um intervalo de valores em torno dele.
C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# Em seguida criaremos um laço for para trocar os valores de C e realizar o ajuste usando regressão logística para cada valor 
# selecionado de C. Mas primeiro, vamos criar uma array vazia para armazenar os valos das pontuações para cada valor de C.

scores = []

for i in C:
    modelo_de_regressao_logistica.set_params(C = i)
    modelo_de_regressao_logistica.fit(X,y)
    scores.append(modelo_de_regressao_logistica.score(X,y))

# Uma vez que registramos os valores das pontuçãos (scores) para cada valor de C, podemos avaliar qual deles é o melhor:
print(scores)

# Explicando os Resultados

# Podemos ver que os valores mais baixos de C tiveram um desempenho pior do que o parâmetro base 1. 
# No entanto, à medida que aumentamos o valor de C para 1.75, o modelo experimentou maior precisão.
# Outra observação importante é que, aumentar C além de 1.75 não resulta num aumento da precisão do modelo,
# pois quando C = 2 o resultado do score permanece igual ao valor obtido para C = 1.75.


