# -*- coding: utf-8 -*-
#data_clean -> Queda absorbido por transformers.py
import numpy as np
from sklearn.impute import SimpleImputer

def data_cleaning(data):
    
    #Copio para no modificar el original por referencia
    data_c = data.copy()
    
    #Paso de strings a números (Engine, Max Power, Max Torque)
    #Me quedo solo con los dígitos y puntos decimales
        
    data_c["Engine"] = data_c["Engine"].str.extract("(\d+)").astype(float)
    data_c["Max Power"] = data_c["Max Power"].str.extract("(\d+\.?\d*)").astype(float)
    data_c["Max Torque"] = data_c["Max Torque"].str.extract("(\d+\.?\d*)").astype(float)

    #Divido data_c en variables numericas y categoricas (luego de la conversión solo Drivetrain es string)
    columnas_num = data_c.select_dtypes(include=[np.number]).columns
    columnas_cat = data_c.select_dtypes(exclude=[np.number]).columns
    
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    
    num_imputer.fit(data_c[columnas_num])
    cat_imputer.fit(data_c[columnas_cat])       
    
    #Sobreescribo la copia del data frame
    data_c[columnas_num] = num_imputer.transform(data_c[columnas_num])
    data_c[columnas_cat] = cat_imputer.transform(data_c[columnas_cat])

    #num_imputer y cat_imputer van a ser útiles cuando limpie el test_set.
    return data_c, num_imputer, cat_imputer

