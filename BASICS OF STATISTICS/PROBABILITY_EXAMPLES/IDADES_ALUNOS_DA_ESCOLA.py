############################################################################################
#   Em uma determinada escola de ensino médio, alguns alunos foram selecionados aleatoriamente
#   para saber qual idade eles tem. Abaixo temos a amostra das idades desse alunos.
#
#   Calcule a média e o desvio padrão, e elabore os gráficos de boxplot, histograma e a 
#   curva de distribuição normal.   
##############################################################################################

# Importando as bibliotecas
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

#Criando o dataframe ou dataset (conjunto de dados)
dados = {'Idades':[14, 17, 18, 15, 15, 16, 17, 15, 16, 16, 15, 17, 15, 16,
                      16, 18, 18, 19, 17, 16, 17, 15, 16, 17, 17, 19, 20, 18,
                      17, 16, 15, 16, 16, 17, 18, 18, 17, 17, 15, 16, 16, 15]
                     }

#Visualizando a distribuição dos dados
plt.hist(dados['Idades'])
plt.show()


# Calculando a média e o desvio padrão
desvio_padrao = np.std(dados['Idades'], ddof = 1)
media = np.mean(dados['Idades'])

# Visualizando a curva da distribuição normal
dominio = np.linspace(np.min(dados['Idades']), np.max(dados['Idades']))
plt.plot(dominio, norm.pdf(dominio, media, desvio_padrao))
plt.show()

dataframe = pd.DataFrame(data = dados)


print("\n Média: ", dataframe['Idades'].mean())
print("\n Mediana: ", dataframe['Idades'].median())
print("\n Moda: ", dataframe['Idades'].mode())
print("\n Desvio Padrão Amostral das Idades: ", dataframe['Idades'].std()) 

# Plotando o boxplot
plt.boxplot(dados['Idades'])
plt.show()