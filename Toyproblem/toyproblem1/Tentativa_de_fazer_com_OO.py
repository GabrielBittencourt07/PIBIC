import pandas as pd 

class DadosIMU(pd.DataFrame): 
    def __init__(self, data):
        super().__init__(data)
        pass 

    def get_axis(self, axis: str): 
        list_cols = ["Timestamp"]
        for col in self.columns: 
            if col.endswith(axis):
                list_cols.append(col)
        return self[list_cols]

    def X(self): 
        return self.get_axis("X")
    
    def Y(self): 
        return self.get_axis("Y")

    def Z(self): 
        return self.get_axis("Z")
    
    def amostras_caes(self): #Retorna um dicionário cuja as chaves são os ID's de identificação dos cães e os valores a contagem de amostras de cada um  
        return self.groupby("Subject").size().to_dict()
        
    def amostras_posições(self): #Retorna um dicionário cuja as chaves são as posturas caninas e os valores são a frequência que cada uma aparece no dataframe
        return self.groupby("Position").size().to_dict()
    
    def dados_num(self): 
        return self.drop(columns=["Type", "Position", "Breed", "Subject", "Timestamp"], errors= "ignore")
    
    def dados_teste(self): 
        self.index = self["Subject"]
        return DadosIMU(self.loc[["04-02-LE","04-03-LD","05-13-LE","05-14-LE"]])
    
    def dados_filt(self): 
        self.index = self["Subject"]
        return DadosIMU(self.loc[['11-24-IK', '11-21-IK', '10-13-LK', '10-15-LK', '11-26-IL', '11-02-IM', '11-22-IM', '11-23-IT']])





        
