#Archivo data_transform
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class StringNumericExtractor(BaseEstimator, TransformerMixin):
    """Extrae números de strings (Engine, Power, Torque)"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # X es un DataFrame aquí porque será el primer paso del pipeline
        X_copy["Engine"] = X_copy["Engine"].str.extract("(\d+)").astype(float)
        X_copy["Max Power"] = X_copy["Max Power"].str.extract("(\d+\.?\d*)").astype(float)
        X_copy["Max Torque"] = X_copy["Max Torque"].str.extract("(\d+\.?\d*)").astype(float)
        return X_copy


#Definimos los índices de las columnas para trabajar directamente con arrays de NumPy.
#Es importante notar que Scikit-Learn suele convertir DataFrames a matrices, 
#por lo que perdemos los nombres de las columnas y debemos usar sus posiciones.

#variable_indice, son la "posicion" de la col. (desde 0).

year_ix = 0
seating_capacity_ix = 5
fuel_tank_capacity_ix = 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para agregar atributos combinados al dataset.
    Hereda de BaseEstimator y TransformerMixin para obtener métodos como fit_transform()
    y permitir su uso automático en una Pipeline de Scikit-Learn.
    """
    
    def __init__(self, add_age_categories = True): #sin *args ni **kargs
        self.add_age_categories = add_age_categories
    
    def fit(self, X, y=None):
        #Aprendo los cuantiles del conjunto de entrenamiento
        #X[:, fuel_tank_capacity_ix] es la columna del tanque
        self.q1, self.q2, self.q3 = np.percentile(X[:, fuel_tank_capacity_ix], [25, 50, 75])
        return self
    
    def transform(self, X, y=None):
        #Agrupo cantidad de asientos \in {5,  7,  4,  8,  2,  6}:
        seat_cat = X[:, [seating_capacity_ix]]
    
        
        #Agrupo por capacidad del tanque    
        cat_tank = [0,1,2,3] #0:Chico, 1: Mediano, 2: Grande, 3: Muy grande.
        cap = X[:,fuel_tank_capacity_ix]
        cond_tank = [
        (cap <= self.q1),
        (cap > self.q1) & (cap <= self.q2),
        (cap > self.q2) & (cap <= self.q3),
        (cap > self.q3)
        ]
        tank_cat = np.select(cond_tank, cat_tank, default=2).reshape(-1, 1) #Con reshape "transpongo", es para pegar la columna
    
        #Agrupo los años:
        if(self.add_age_categories):
            #Categorias 0 = Nuevo, 1 = Semi Nuevo, 2 = Usando, 3 = Antiguo.
            cat_age = [0,1,2,3]
            age = 2026 - X[:,year_ix]
            
            cond_age = [
            (age <= 3),
            (age >= 4) & (age <= 7),
            (age >= 8) & (age <= 12),
            (age >= 13)
            ]
            age_cat = np.select(cond_age, cat_age, default=2).reshape(-1, 1)
            
            # Agregamos la nueva columna a la matriz X original
            # np.c_ es como un "append" de columnas
            return np.c_[X, seat_cat, tank_cat, age_cat]
        else:
            return np.c_[X, seat_cat, tank_cat]