# -*- coding: utf-8 -*-
#OBTENCION DE DATOS:
import os
from six.moves import urllib

#Configuracion de rutas y URLs:

#DOWNLOAD_URL: Es el link RAW
DOWNLOAD_URL = "https://raw.githubusercontent.com/JnDecibe/car-price-prediction-pipeline/refs/heads/main/data/car_details_v4.csv"

#CARS_PATH: La carpeta donde se va a guardar en mi PC
DATASET_PATH = os.path.join("datasets", "cars")

def fetch_cars_data(url=DOWNLOAD_URL, path=DATASET_PATH):
    '''
    Crea el directorio local y descarga el CSV directamente.
    '''
    #Crea la carpeta "datasets/cars" si no existe
    if not os.path.isdir(path):
        os.makedirs(path)
    
    #Define la ruta local completa para el archivo descargado   
    csv_path = os.path.join(path, "car_details_v4.csv")
    
    #Descarga el archivo desde la URL y lo guarda en tgz_path
    urllib.request.urlretrieve(url, csv_path)
    

#CARGA DE DATOS:
import pandas as pd

def load_cars_data(path=DATASET_PATH):
    
    #Carga el archivo CSV en un objeto DataFrame de pandas para su análisis.
    
    #Construye la ruta completa al archivo: "datasets/cars/cars.csv"
    csv_path = os.path.join(path, "car_details_v4.csv")
    
    #Lee el archivo y lo devuelve como un DataFrame (la estructura principal de pandas)
    return pd.read_csv(csv_path)

