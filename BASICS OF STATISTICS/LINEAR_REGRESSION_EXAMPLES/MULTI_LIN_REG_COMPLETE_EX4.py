################################################################################################
# Podemos prever a emissão de CO2 de um carro com base no volume do motor, mas com a regressão 
# múltipla podemos incluir mais variáveis, como o peso do carro, para tornar a previsão mais precisa. 
# Considerando a base de dados do arquivo .csv, que reúne informações sobre algumas marcas e modelos 
# de carros, desenvolva um script em Python para realizar uma regressão linear de múltiplas 
# variáveis e verificar a relação entre a emissão de CO2 com o peso do carro e o volume do motor. 

# Neste exemplo iremos analisar algumas propriedades importantes estatisticamente, 
# que geralmente são verificadas durante a construção  de um modelo de aprendizado de máquina.

################################################################################################

# Importando as bibliotecas básicas
import pandas as pd
import numpy as np


# Bibliotecas usadas para a construção de um modelo de aprendizado de máquina
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

# Bibliotecas para construção dos gráficos
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['figure.figsize'] = (7,7)
plt.style.use('ggplot')

# Abrir arquivo .csv com os dados para construção do modelo
df = pd.read_csv('LINEAR_REGRESSION_EXAMPLES\ARQUIVOS_CSV\CO2_&_CARROS_1.csv', sep = ';')

print(df)

########################################################################
#
#    EXPLORANDO A EXISTÊNCIA DE ALGUMA RELAÇÃO ENTRE OS DADOS 
#
########################################################################

# Visualizando os dados usando gráficos de espalhamento e histogramas
sn.set_palette('colorblind')
sn.pairplot(data = df, height = 3)
plt.show()

# Podemos notar a partir dos gráficos gerados com a função  da biblioteca
# seaborn que existe uma relação de tendência positiva entre os dados das
# colunas Volume_Motor e Peso_Carro, bem como entre as colunas CO2 com Volume_Motor 
# e CO2 com Peso_Carro. 

########################################################################
#
#                CONSTRUÇÃO DO MODELO DE REGRESSÃO LINEAR
#
########################################################################

# Gerando uma lista das variáveis preditoras (variáveis independentes) e nomeando-as de X [x1, x2, ..., xn]
X = df[['Volume_Motor', 'Peso_Carro']]  # É comum nomear a lista de variáveis preditoras com letras maiúsculas

# Definindo a varável resposta (variável dependente)
y = df[['CO2']]    # É comum nomear a lista de variáveis respostas com letras minúsculas               

ajuste_modelo = smf.ols(formula = 'CO2 ~ Volume_Motor + Peso_Carro', data = df)
equacao_ajustada = ajuste_modelo.fit()
equacao_ajustada.params

# Adiciona uma constante para cada variável preditora
X = sm.add_constant(X)

# Realiza a regressã linear usando a função OLS() --> Ordinary Least Squares
modelo = sm.OLS(y, X).fit()

# Apresenta os resultados da regressão linear
print(modelo.summary())

########################################################################
#
#                PREVENDO VALORES DE EMISSÃO DE CO2 COM O MODELO
#
########################################################################
def calcula_nivel_CO2(x1,x2):
        
    CO2 = 79.6947 + 0.0078 * x1 + 0.0076 * x2

    return CO2
    

try:
    Volume_Motor = int(input(" Informe o volume do motor do veículo: \n"))
    Peso_Carro = int(input(" Informe o peso do veículo: \n"))
    print("\n Nível de CO2 emitido pelo vaículo: \n", calcula_nivel_CO2(Volume_Motor, Peso_Carro))

except ValueError:
    print("\n Por favor, informe apenas os valores numéricos inteiros de cada grandeza: \n Ex: Volume do Motor = 1200 \n Peso do Veículo =  2000")


########################################################################
#
#                VISUALIZAÇÃO DOS DADOS E DO MODELO AJUSTADO
#
########################################################################

# Preparando os dados para visualização
x_surf, y_surf = np.meshgrid(np.linspace(df.Volume_Motor.min(), df.Volume_Motor.max(), 100), np.linspace(df.Peso_Carro.min(), df.Peso_Carro.max(), 100))
onlyX = pd.DataFrame({'Volume_Motor': x_surf.ravel(), 'Peso_Carro': y_surf.ravel()})
fittedY = equacao_ajustada.predict(exog = onlyX)

# Converte os resultados preditos em uma array
fittedY = np.array(fittedY)

# Construção do gráfico para a regresssão linear com múltiplas variáveis
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['Volume_Motor'], df['Peso_Carro'], df['CO2'], c = 'red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='blue', alpha=0.3)
ax.set_xlabel('Volume_Motor (cm^3)')
ax.set_ylabel('Peso_Carro (kg)')
ax.set_zlabel('CO2 (g/kg)')
plt.show()


########################################################################
#
#                           VALIDAÇÃO DO MODELO
#
########################################################################
# Depois de construir o modelo é importante para nós validar o seu desempenho. 
# Podemos avaliar um modelo observando seu coeficiente de determinação ( R2 ), 
# teste F, teste t e também os resíduos. 
# 
# O resumo do modelo contém muitos valores importantes que podemos usar 
# para avaliar nosso modelo.

# O coeficiente de determinação R2 é a parcela da variação total da variável 
# dependente que é explicada pela variação da variável independente.

# Com a bib statsmodel podemos obter o valor R2 do nosso modelo acessando o atributo 
# .rsquared do nosso modelo.
print("\n Coeficiente de determinação R2: ", modelo.rsquared)

# R2 varia entre 0 e 1, onde R2=0 significa que não há relação linear entre as variáveis 
# e R2 = 1 mostra uma relação linear perfeita. No nosso caso, obtivemos uma pontuação 
# R2 de cerca de 0.3765, o que significa que 37.65% de nossa variável dependente pode 
# ser explicada usando nossas variáveis independentes.


# Teste F (ANOVA)
#
# O teste F ou ANOVA (Analysis of Variance - Análise de Variância) em regressão multilinear 
# pode ser usado para determinar se nosso modelo complexo tem um desempenho melhor do que um 
# modelo mais simples (por exemplo, modelo com apenas uma variável independente). 
# Com o teste F podemos avaliar a significância do nosso modelo calculando a probabilidade 
# de observar uma estatística F que seja pelo menos tão alta quanto o valor que nosso modelo obteve. 
# Semelhante à pontuação R2, podemos obter facilmente a estatística F e a probabilidade da referida 
# estatística F acessando o atributo .fvalue e .f_pvalue do nosso modelo conforme abaixo:
print("\n F-statistic (ANOVA): ", modelo.fvalue)
print("\n Probabilidade (f_pvalue) de observar valores maiores que F-statistic: ", modelo.f_pvalue)

# Como nosso f_pvalue é menor que 0.05, podemos concluir que nosso modelo tem um desempenho melhor 
# do que outro modelo mais simples (por exemplo, sem considerar a influência da variáveis independentes).


# Teste T
# 
# O parâmetro t-statistic é o coeficiente linear dividido pelo seu erro padrão. 
# O erro padrão é uma estimativa do desvio padrão do coeficiente, a quantidade que varia entre os casos. 
# Pode ser pensado como uma medida da precisão com que o coeficiente de regressão é medido. 
# Igual ao teste F, o valor p mostra a probabilidade de ver um resultado tão extremo quanto o nosso modelo que temos. 
# Também podemos obter o valor p para todas as nossas variáveis chamando o atributo .pvalues no modelo.
print(modelo.pvalues)

# Ambas variáveis independentes, Volume_Motor e Peso_Carro, têm o p_value maior a 0.05, 
# isso significa que não há evidências suficientes de que Volume_Motor e Peso_Carro afetam 
# os níveis de emissão de CO2.

########################################################################
#
#                        TESTES DE SUPOSIÇÕES
#
########################################################################

# Em seguida, vamos validar nosso modelo fazendo análise de resíduos, 
# abaixo está a lista de testes ou suposições que faremos para verificar a validade do nosso modelo:

# Linearidade
# Normalidade
# Multicolinearidade
# Autocorrelação
# Homocedasticidade

# Residual é a diferença entre o valor observado e o valor previsto do nosso conjunto de dados. 
# Com statsmodel podemos facilmente obter o valor residual do nosso modelo simplesmente acessando 
# o atributo .resid do modelo e então podemos mantê-lo em uma nova coluna chamada 'residual' 
# em nosso dataframe df.
df['Nivel_CO2_Predito'] = modelo.predict(X)
df['Residual'] = modelo.resid
print(df) 


# Linearidade
# 
# Pressupõe que existe uma relação linear entre as variáveis independentes e a variável dependente. 
# No nosso caso, como temos 2 variáveis independentes, podemos fazer isso usando um gráfico de dispersão 
# para ver nossos valores previstos versus os valores reais.

# Construindo o gráfico dos valores reais vs. valores preditos
sn.lmplot(x = 'CO2', y = 'Nivel_CO2_Predito', data = df, fit_reg = False)

# Contrução da linha diagonal
coordenadas_linha = np.arange(df[['CO2','Nivel_CO2_Predito']].min().min()-10,
                              df[['CO2','Nivel_CO2_Predito']].max().min()+10)

plt.plot(coordenadas_linha, coordenadas_linha, # ponts X e y
         color = 'darkorange', linestyle='--')

plt.ylabel('Nivel de CO2 Predito', fontsize= 14)
plt.xlabel('Valor Real de CO2', fontsize= 14)
plt.title('Suposição de Linearidade', fontsize=14)
plt.show()

# Os gráficos de dispersão mostram pontos residuais distribuídos não uniformes o bastante ao redor da linha diagonal, 
# de modo que não podemos supor que existe uma relação linear entre nossas variáveis independentes e dependentes.
# Essa é uma avaliação importante e fácil de ser entendida neste exemplo específico, pois o peso do carro é 
# fortemente influenciado pelo peso do motor, e consequentemente pelo tamanho do mesmo, ou seja o vaolume do motor!


# Normalidade
# 
# Isso pressupõe que os termos de erro do modelo são normalmente distribuídos. 
# Examinaremos a normalidade dos resíduos plotando-os no histograma e analisando o p_value do teste 
# de Anderson-Darling para normalidade. Usaremos a função normal_ad() do statsmodel para calcular nosso p_value 
# e depois compará-lo com o limite de 0.05. 
# Se o p_value obtido for maior que o limite, podemos assumir que nosso resíduo é distribuído normalmente.
p_value = normal_ad(df['Residual'])[1]
print("\n p_value obtido a partir do teste de Anderson-Darling:", p_value)
print("\n valores de p_value < 0.05 significa uma distribuição não normal")

# Plotando a distribuição dos resíduos
plt.subplots(figsize = (8,4))
plt.title('Distribuição dos resíduos', fontsize = 18)
sn.distplot(df['Residual'])
plt.show()

# Análise sobre a normalidade dos resíduos
if p_value < 0.05:
    print("\n O resíduos não estão distribuídos normalmente")
else:
    print("\n Resíduos estão distribuídos normalmente")
    
# Do código acima, obtivemos um p_value = 0.1816, que pode ser considerado normal porque está acima do limite de 0.05. 
# O gráfico do histograma também mostra uma distribuição normal (apesar de parecer um pouco distorcida porque temos a poucas observações 
# em nosso conjunto de dados). De ambos os resultados, podemos supor que nossos resíduos são normalmente distribuídos.


# Multicolinearidade
# 
# Isso pressupõe que os preditores usados na regressão não estão correlacionados entre si. 
# Para identificar se há alguma correlação entre nossos preditores, podemos calcular o coeficiente de correlação de Pearson 
# entre cada coluna em nossos dados usando a função corr() do dataframe do Pandas. 
# Em seguida, podemos exibi-lo como um mapa de calor usando a função heatmap() do Seaborn.
corr = df[['Volume_Motor', 'Peso_Carro', 'CO2']].corr()
print("\nMatriz dos coeficientes de correlação de Pearson para cada variável:\n", corr)

# Gera uma máscara para os elementos da diagonal principal da matriz
mascara_diagonal = np.zeros_like(corr, dtype = np.bool)
np.fill_diagonal(mascara_diagonal, val = True)

# Define o tamanho da figura para a construção da matriz de correlação
fig, ax = plt.subplots(figsize = (4,3))

# Gera um mapa de cores personalizado para diferenciar valores extremos num certo intervalo 
mapa_de_cores = sn.diverging_palette(220, 10, as_cmap = True, sep = 100)
mapa_de_cores.set_bad('grey')

# Constrói o mapa de cores com a máscara da matriz de correlação e tamanho definidos acima
sn.heatmap(corr, mask = mascara_diagonal, cmap = mapa_de_cores, vmin = -1, vmax = 1, center = 0, linewidths=.5)
fig.suptitle('Matriz dos coeficientes de correlação de Pearson', fontsize= 24)
ax.tick_params(axis='both', which='major', labelsize = 10)
plt.show()

# A imagem da matriz mostra que há forte relação positiva entre Volume_Motor e Peso_Carro 
# e uma relação positiva mais fraca entre CO2 com Volume_Motor e CO2 com Peso_Carro. 
# Isso significa que ambas as variáveis independentes estão afetando uma à outra e que existe 
# multicolinearidade em nossos dados.


# Autocorrelação

# Autocorrelação é a correlação dos erros (resíduos) ao longo do tempo. 
# Usado quando os dados são coletados ao longo do tempo para detectar se a autocorrelação está presente. 
# A autocorrelação existe se os resíduos em um período de tempo estiverem relacionados aos resíduos em outro período. 
# Podemos detectar a autocorrelação realizando o teste de Durbin-Watson para determinar se há correlação positiva ou negativa. 
# Nesta etapa, usaremos a função durbin_watson() do statsmodel para calcular nossa pontuação Durbin-Watson e, em seguida, 
# avaliar o valor com a seguinte condição:

# Se a pontuação de Durbin-Watson for menor que 1.5, então há uma autocorrelação positiva e a suposição não é satisfeita
# Se a pontuação de Durbin-Watson estiver entre 1.5 e 2.5, então não há autocorrelação e a suposição é satisfeita
# Se a pontuação de Durbin-Watson for superior a 2.5, há uma autocorrelação negativa e a suposição não é satisfeita
durbinWatson = durbin_watson(df['Residual'])

print('Durbin-Watson: ', durbinWatson)

if durbinWatson < 1.5:
    print("\n Sinal de autocorrelação positiva,")
    print("\n Suposição não satisfeita")
elif durbinWatson > 2.5:
    print("\n Sinal de autocorrelação negativa,")
    print("\n Suposição não satisfeita")
else:
    print("\n Sinal de autocorrelação baixo,")
    print("\n Suposição satisfeita")

# Nosso modelo obteve uma pontuação de Durbin-Watson de cerca de 0.94, que está abaixo de 1.5, 
# então podemos supor que há autocorrelação em nosso resíduo.

# Homocedasticidade
# 
# Isso pressupõe homocedasticidade, que é a mesma variância dentro de nossos termos de erro. 
# A heterocedasticidade, a violação da homocedasticidade, ocorre quando não temos uma variação 
# uniforme entre os termos de erro. Para detectar a homocedasticidade, podemos plotar nosso resíduo 
# e ver se a variância parece ser uniforme.

# Plotando os resíduos
plt.subplots(figsize = (8,4))
plt.scatter(x = df.index, y = df.Residual, alpha=0.8)
plt.plot(np.repeat(0, len(df.index)+2), color = 'darkorange', linestyle = '--')

plt.ylabel('Residual', fontsize = 14)
plt.xlabel('Número da Amostra', fontsize = 14)
plt.title('Teste de Homocedasticidade', fontsize = 16)
plt.show()

# Apesar de ter poucos pontos de dados, nosso resíduo parece ter variância constante e uniforme, 
# então podemos supor que ele satisfez a suposição de homocedasticidade.

# Conclusão
# 
# Nossos modelo não obteve sucesso em todos os testes nas etapas de validação do modelo, 
# entretanto, podemos concluir que o mesmo pode ter um bom desempenho bom para prever níveis de emissão de CO2 
# as duas variáveis independentes, Volume do Motor e Peso do Carro. 
# Mas ainda assim, nosso modelo só tem pontuação R2 de 37.65%, 
# o que significa que ainda há cerca de 62% de fatores desconhecidos que estão afetando as emissões de CO2.
