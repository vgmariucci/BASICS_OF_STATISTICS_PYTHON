###############################################################################
#   Script usado para verificar se um dado é viciado ou não
#
############################################################################

# importando as bibliotecas
import random 
import pandas as pd

#Criando uma lista com números de 1 a 6 com 200 elementos (tamanho igual a 200) 
lista_numeros_dado_6_faces = [random.randint(1,6) for i in range(200)]

#Passando a lista para um dataframe
df = pd.DataFrame(data=lista_numeros_dado_6_faces, columns=['Face'])

#Colocando os dados em um histograma
df.plot.hist(align = 'right', rwidth = 0.9, bins = 6, legend = False)
