# -*- coding: utf-8 -*-
#Archivo data_split
#Aplico el split estratificado:
from sklearn.model_selection import StratifiedShuffleSplit

def split_train_test_stratified(data, cat_column, test_ratio=0.2, random_seed=42):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    
    #split.split genera los indices numericos de las filas.
    #Uso .loc para extraer las filas completas basandome en esos indices
    #y crear los nuevos DataFrames de entrenamiento y testeo.
    for train_index, test_index in split.split(data, data[cat_column]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    
    return strat_train_set, strat_test_set

