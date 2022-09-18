# Importando as bibliotecas
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('bmh')


# Definindo as variáveis de entrada aleatoriamente no intervalo delimitado pelo quadrado de lado 1
# x = np.random.uniform(-1, 1)
# y = np.random.uniform(-1, 1)

# if np.sqrt(x**2 + y**2) < 1:
#     print("\n O ponto está dentro do círculo")

N = 1000 # Total de pontos que serão usados para a simulação

nc = []  # Variável para contar quantos pontos foram gerados dentro do círculo

# Contagem da quantidade de pontos dentro do círculo:  
for i in range(N):
    # Definindo as variáveis de entrada aleatoriamente no intervalo delimitado pelo quadrado de lado 2R
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    
    # Se algum ponto for gerado dentro do círculo, registra a coordenada (x, y) do mesmo na variável nc = []
    if np.sqrt(x**2 + y**2) < 1:
        nc.append((x, y))
        
plt.figure(figsize = (10,10))

plt.scatter(
    [x[0] for x in nc], 
    [x[1] for x in nc], 
    marker = ".", 
    alpha = 0.5)

pi = 4 * float( len(nc) / N)

print("\n Valor estimado de pi: ", pi)