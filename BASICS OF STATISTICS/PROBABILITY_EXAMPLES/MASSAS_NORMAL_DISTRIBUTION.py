################################################################################################################################
#
#   Obtenção da curva norma para as densidades das massas de pastel.
#
###############################################################################################################################

# Importando as bibliotecas
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

#Criando o dataframe ou dataset (conjunto de dados)
dados = {'Densidade':[22.02   ,   23.83  ,   26.67  ,    25.38  ,    25.49   ,   23.50  ,    25.90   ,  24.89,
                      21.49   ,   22.67  ,    24.62 ,     24.18 ,     22.78  ,    22.56 ,     24.46  ,  23.79,
                      20.33   ,   21.67  ,    24.67 ,     22.45 ,     22.29  ,    21.95 ,     20.49  ,   21.81]
                     }

#Visualizando a distribuição dos dados
plt.hist(dados['Densidade'])
plt.show()


# Calculando a média e o desvio padrão
desvio_padrao = np.std(dados['Densidade'], ddof = 1)
media = np.mean(dados['Densidade'])

# Visualizando a curva da distribuição normal
dominio = np.linspace(np.min(dados['Densidade']), np.max(dados['Densidade']))
plt.plot(dominio, norm.pdf(dominio, media, desvio_padrao))
plt.show()