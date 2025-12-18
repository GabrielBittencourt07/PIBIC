import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


dados = pd.read_csv("df_filt.csv")
teste = pd.read_csv("df_test.csv")

# dados_group = dados.groupby("Subject").count()
# # print(dados_group)
# # print(dados_group.describe())
# dados_group = dados.groupby("Position").count()
# print(dados_group)
# print(dados_group.describe())
# Body shake tem muito menos dados que o restante das categorias. Por outro lado, standing foi a categoria de maior quantidade de dados. A média geral é de 60380.
#print(dados.info())

# # Preparando os dados de treino
# y_completo = dados["Position"]
# x_completo = dados.drop(columns=["Position", "Timestamp", "Type", "Breed", "Subject"])

# # Separando 20% dos dados de treino para validação
# x_train, x_val, y_train, y_val = train_test_split(
#     x_completo, y_completo,
#     test_size=0.20,
#     random_state=42,
#     stratify=y_completo
# )

# print(f"Tamanho do conjunto de treino: {x_train.shape[0]}")
# print(f"Tamanho do conjunto de validação: {x_val.shape[0]}")
# print("="*60)

# Preparando os dados de teste
y_teste = teste["Position"]
x_teste = teste.drop(columns=["Position", "Timestamp", "Type", "Breed", "Subject"])

x_treino = dados.drop(columns=["Position", "Timestamp", "Type", "Breed", "Subject"])
y_treino = dados["Position"]
# Definindo os hiperparâmetros para Grid Search (3 valores para cada)
# param_grid = {
#     'n_estimators': [40, 80, 120],           # número de árvores
#     'max_depth': [15, 25, 35],                # profundidade máxima
#     'min_samples_leaf': [10, 20, 30]         # mínimo de amostras por folha
# }

# Criando o modelo base
rf = RandomForestClassifier(
    n_estimators=300,        # número de árvores
    max_depth=12,          # deixe None inicialmente
    min_samples_split=2,
    min_samples_leaf=1,     # padrão bom para classificação
    bootstrap=True,
    n_jobs=6,
    random_state=42
)

rf.fit(x_treino, y_treino)

y_pred = rf.predict(x_teste)

acc = accuracy_score(y_teste, y_pred)
print(f"Acurácia: {acc:.4f}")

print(classification_report(y_teste, y_pred))

cm = confusion_matrix(y_teste, y_pred)
print(cm)





# Grid Search com Cross-Validation de 5 folds
# print("Iniciando Grid Search com 5-fold Cross-Validation...")
# print(f"Total de combinações: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf'])}")
# print("="*60)

# grid_search = GridSearchCV(
#     estimator=rf_base,
#     param_grid=param_grid,
#     cv=5,                    # 5 folds
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2,
#     return_train_score=True
# )

# Treinando com Grid Search no conjunto de treino
# grid_search.fit(x_train, y_train)

# Mostrando os melhores parâmetros
# print("\n" + "="*60)
# print("Melhores hiperparâmetros encontrados:")
# print(grid_search.best_params_)
# print(f"Melhor score no CV: {grid_search.best_score_:.4f}")
# print("="*60)

# # Avaliando no conjunto de validação
# y_pred_val = grid_search.predict(x_val)
# acuracia_val = accuracy_score(y_val, y_pred_val)
# print(f"\nAcurácia no conjunto de validação: {acuracia_val:.4f}")
# print("\nRelatório de classificação (Validação):")
# print(classification_report(y_val, y_pred_val))

# # Treinando o modelo final com treino + validação
# print("\n" + "="*60)
# print("Treinando modelo final com treino + validação...")
# rf_final = RandomForestClassifier(
#     **grid_search.best_params_,
#     random_state=42,
#     n_jobs=-1
# )

# Combinando treino e validação
# rf_final.fit(x_completo, y_completo)

# # Avaliando no conjunto de teste final
# y_pred = rf_final.predict(x_teste)
# acuracia_teste = accuracy_score(y_teste, y_pred)

# print(f"\nAcurácia no conjunto de TESTE final: {acuracia_teste:.4f}")
# print("="*60)
# print("\nRelatório de classificação (Teste Final):")
# print(classification_report(y_teste, y_pred))

# # Matriz de confusão
# cm = confusion_matrix(y_teste, y_pred, normalize='true')
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_final.classes_)
# disp.plot(cmap='Blues')
# plt.title(f'Matriz de Confusão - Acurácia: {acuracia_teste:.4f}')
# plt.tight_layout()
# plt.show()
