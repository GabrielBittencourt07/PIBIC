import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import Tentativa_de_fazer_com_OO as mod

dados = mod.DadosIMU(pd.read_csv("df_filt.csv"))
teste = mod.DadosIMU(pd.read_csv("df_test.csv"))

dados["Timestamp"] = pd.to_datetime(dados["Timestamp"])
dados["segundo"] = dados["Timestamp"].dt.floor("s")

dados_num = dados.drop(columns=["Type", "Position", "Breed", "Subject", "Timestamp"])

grouped = dados_num.groupby("segundo")

features = pd.DataFrame()

for col in dados_num.columns[:-1]:
    _ = grouped[col].describe()
    features[f"mean_{col}"] = _["mean"]
    features[f"std_{col}"] = _["std"]

#=====================================================================================

def get_axis(ax:str, dataframe:pd.DataFrame) -> pd.DataFrame: 
    """Filtra o Eixo de um dataframe"""
    dataframe_ax = pd.DataFrame()
    for col in dataframe.columns: 
        if col.endswith(ax):
            dataframe_ax[col] = dataframe[col] 
    return dataframe_ax

def get_axis_mean(ax: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Retorna um Dataframe com com a média das médias e dos desvios padrões de um eixo específico"""
    dataframe_ax_mean = pd.DataFrame()
    dataframe_ax_std = pd.DataFrame()

    for col in dataframe: 
        if col.endswith(ax):
            if col.startswith("mean"):
                dataframe_ax_mean[col] = dataframe[col]
            else: 
                dataframe_ax_std[col] = dataframe[col]
    
    dataframe_ax_mean = dataframe_ax_mean.mean(axis = 1 )
    dataframe_ax_std = dataframe_ax_std.mean(axis = 1 )
        
    return pd.DataFrame({f"mean_media_{ax}": dataframe_ax_mean, f"mean_Desviopadrao_{ax}": dataframe_ax_std})

def get_axis_sum(ax: str, dataframe: pd.DataFrame) -> pd.DataFrame: 
    """Faz a soma do desvio padrão que acontece em um eixo em específico"""
    dataframe_ax_std = pd.DataFrame()

    for col in dataframe: 
        if col.endswith(ax):
            if col.startswith("std"):
                dataframe_ax_std[col] = dataframe[col]
            
    dataframe_ax_sum_std = dataframe_ax_std.sum(axis = 1 )

    return pd.DataFrame({f"soma_std_{ax}":dataframe_ax_sum_std})

def get_sensor(sensor: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_sensor = pd.DataFrame()
    for col in dataframe.columns: 
        if sensor in col: 
            dataframe_sensor[col] = dataframe[col]

    return dataframe_sensor

# =========================================================================

standing = dados[dados["Position"] == "standing"]
conjunto_sensores = set()
for col in dados_num.columns: 
    conjunto_sensores.add(col[:-2])

eixos = ("X", "Y", "Z")

dict_sensores = {}
for sensor in conjunto_sensores: 
    df_sensor = get_sensor(sensor, standing)
    dict_sensor_valores = {}
    
    for ax in eixos: 
        mean_standing_ax_sensor =  get_axis(ax, df_sensor).mean()
        dict_sensor_valores[f"mean_{ax}"] = mean_standing_ax_sensor

    dict_sensores[sensor] = dict_sensor_valores

medias_ax_sensor = pd.DataFrame.from_dict(dict_sensores, orient= "index")

print(medias_ax_sensor)


           
# 
# sum_std_ax = pd.DataFrame()
# mean_std = pd.DataFrame()
# for ax in eixos:
#     mean_std = get_axis_mean(ax, features).join(mean_std)
#     sum_std_ax = get_axis_sum(ax, features).join(sum_std_ax)
# print(sum_std_ax)






