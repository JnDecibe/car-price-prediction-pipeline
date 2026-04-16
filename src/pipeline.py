# -*- coding: utf-8 -*-
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys
import os

#Agregamos la carpeta raíz al path de búsqueda de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transformers import StringNumericExtractor, CombinedAttributesAdder

def build_full_pipeline(num_attribs, cat_attribs):
    #Pipeline para números
    num_pipeline = Pipeline([
        #Recibe string, devuelve floats:
        ('extractor', StringNumericExtractor()), #Limpieza de strings primero
        
        #Se Rellenan los valores faltantes con la mediana (Limpieza)
        ('imputer', SimpleImputer(strategy="median")),          #acá ocurre automaticamente num_imputer.fit(data_c[columnas_num])
        
        #Agrega las columnas de edad, tanque y asientos:
        ('attribs_adder', CombinedAttributesAdder()),   
        
        #StandardScaler asegura que todas las variables tengan media 0 y varianza 1
        ('std_scaler', StandardScaler()),
    ])

    #Pipeline para categoricos
    cat_pipeline = Pipeline([
        #Rellena los nulos con most_frequent:
        ('imputer', SimpleImputer(strategy="most_frequent")),   #acá ocurre automaticamente cat_imputer.fit(data_c[columnas_cat])    
        
        #Se Aplica OneHotEncoder a la columna de texto/categoría:
        ('one_hot', OneHotEncoder(handle_unknown='ignore')),
    ])

    #ColumnTransformer permite aplicar diferentes transformaciones a diferentes columnas:
    full_pipeline = ColumnTransformer([
        #(Nombre, Transformador/Pipeline, Lista de columnas)
        
        #Se Aplica la pipeline numérica (limpieza, adder, escalado) a las columnas de números:
        ("num", num_pipeline, num_attribs),
        
        #Se Aplica la pipeline categórica
        ("cat", cat_pipeline, cat_attribs),
    ])
    
    return full_pipeline