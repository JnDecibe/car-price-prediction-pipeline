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
    #1. Pipeline para números
    num_pipeline = Pipeline([
        ('extractor', StringNumericExtractor()), #Limpieza de strings primero
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    #2. Pipeline para categorías
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('one_hot', OneHotEncoder(handle_unknown='ignore')),
    ])

    #3. Unión de ambos caminos
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
    
    return full_pipeline