import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


dados = pd.read_csv("df_filt.csv")
teste = pd.read_csv("df_test.csv")

dados_group = dados.groupby("Subject").count()
# print(dados_group)
# print(dados_group.describe())
dados_group = dados.groupby("Position").count()
# print(dados_group)
# print(dados_group.describe())
# Body shake tem muito menos dados que o restante das categorias. Por outro lado, standing foi a categoria de maior quantidade de dados. A média geral é de 60380. 
#print(dados.info())

y_train = dados["Position"]
x_train = dados.drop(columns=["Position", "Timestamp", "Type", "Breed", "Subject"])

y_teste = teste["Position"]
x_teste = teste.drop(columns=["Position", "Timestamp", "Type", "Breed", "Subject"])

rf = RandomForestClassifier(
    n_estimators=100,       # número de árvores
    random_state=42,        # para reprodutibilidade
    n_jobs=-1               # usa todos os núcleos disponíveis
)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_teste)
print("Acurácia:", accuracy_score(y_teste, y_pred))
print(classification_report(y_teste, y_pred))
