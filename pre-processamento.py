# Importação das libs (bibliotecas) necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregamento do Conjunto de Dados (dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)

# Análise Exploratória de Dados (EDA)
# Visualizar as primeiras linhas do dataset
# print(titanic_data.head())

# Verificar informações sobre os tipos de dados e valores ausentes
# print(titanic_data.info())

# Tratamento de Dados Ausentes
# Preencher valores ausentes na coluna 'Age' com a média
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Remover linhas com valores ausentes na coluna 'Embarked'
titanic_data.dropna(subset=['Embarked'], inplace=True)

# Aplicação da codificação one-hot na coluna 'Embarked'
# One-hot: técnica para representar variáveis categóricas como variáveis binárias
# Cada coluna indica a presença(1) ou a ausência(0) 
# Exemplo: Fruta [Maçã, Banana, Laranja] => CODIFICAÇÃO ONE-HOT => Fruta_Maçã(0), Fruta_Banana(1), Fruta_Laranja(0)
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

# print('Antes:')
# print(titanic_data['Fare'])

# Normalização dos Dados Numéricos
# Usar o StandarScaler para normalizar a coluna 'Fare'
scaler = StandardScaler()
titanic_data['Fare'] = scaler.fit_transform(titanic_data['Fare'].values.reshape(-1, 1))

# print('Depois:')
# print(titanic_data['Fare'])