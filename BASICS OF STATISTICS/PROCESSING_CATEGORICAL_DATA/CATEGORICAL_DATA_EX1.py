############################################################################################################
# 
# Neste exemplo veremos como trabalhar com dados categóricos, ou seja, dados como nomes em geral, símbolos
# não numéricos e outras formas de classificadores que precisamos lidar com frequecia em aprendizado 
# de máquina e análise de dados  
#
############################################################################################################

# Quando seus dados têm categorias representadas por strings, será difícil usá-los para treinar modelos de 
# aprendizado de máquina que geralmente aceitam apenas dados numéricos. Em vez de ignorar os dados categóricos 
# e excluir as informações do nosso modelo, você pode transformar os dados para que possam ser usados em seus modelos.
#
# Neste exemplo iremos usar a base de dados do arquivo CO2_&_CARROS_1.csv ao trabalhar com dados categóricos como 
# por exmeplo os nomes do carros ou as marcas dos mesmos

# Importando as bibliotecas
import pandas as pd

df = pd.read_csv('PROCESSING_CATEGORICAL_DATA\ARQUIVOS_CSV\CO2_&_CARROS_1.csv', sep=';')

# Visualizando os dataframe gerado após ler o arquivo .csv
print(df)

# One Hot Encoding

# Não podemos usar a coluna Carro ou Modelo em nossos dados, pois eles não são numéricos. 
# Uma relação linear entre uma variável categórica, Carro ou Modelo, e uma variável numérica, CO2, não pode ser determinada.
# Para corrigir esse problema, devemos ter uma representação numérica da variável categórica. 
# Uma maneira de fazer isso é ter uma coluna representando cada grupo na categoria.

# Para cada coluna, os valores serão 1 ou 0, onde 1 representa a inclusão do grupo e 0 representa a exclusão. 
#  Essa transformação é chamada de codificação a quente do inglês Hot Encoding

# O módulo  Pandas tem uma função chamada get_dummies() que faz uma "codificação quente".

marca_carros = pd.get_dummies(df[['Marca']])

print(marca_carros)

# O resultado do hot-encoding  para a coluna 'Marca' do dataframe do arquivo .csv
# é a criação de uma coluna para cada Marca de carro.

