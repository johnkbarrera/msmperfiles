
# LIBRERIAS
import numpy as np
import datetime
from datetime import date
import json
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import normalize

# ARCHIVO DE CONFIGURACION
import configparser
Config = configparser.ConfigParser()
Config.read("Config.conf")
Config.sections()


def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


# RUTAS
path_out=ConfigSectionMap("ruta")['salidas']
nombre = ConfigSectionMap("file")['data'][:-4]

# RUTAS DE ARCHIVOS ENTRADAS 
footprint="%s%s.footprint" %(path_out,nombre)
individual_clusters="%s%s.individual_footprint.clusters" %(path_out,nombre)
individual_labels="%s%s.individual_footprint.labels" %(path_out,nombre)  
collective_clusters="%s%s.collective_footprint.clusters" %(path_out,nombre)
collective_labels="%s%s.collective_footprint.labels" %(path_out,nombre)  


# CARGANDO FILE
print("Loading Data ...")

b = pd.read_csv(collective_labels, sep=";", header=0, index_col=False, low_memory=False)[['CO_ID', 'INDIVIDUAL_CLUSTER', 'COLLECTIVE_CLUSTER']] 
a = pd.read_csv(individual_labels, sep=";", header=0, index_col=False, dtype={'WEEK': str,'YEAR': str}, low_memory=False)[['CO_ID', 'YEAR', 'WEEK', 'INDIVIDUAL_CLUSTER']]

print("Procesing Data ...")
b = pd.merge(a, b, on=['CO_ID', 'INDIVIDUAL_CLUSTER'])
del(a)
a = pd.read_csv(footprint, sep=",", header=0, dtype={'YEAR': str,'WEEK': str}, low_memory=False)   ## read file
a = pd.merge(a, b, on=['CO_ID','YEAR', 'WEEK'])
del(b)
len(a)


# RUTAS DE ARCHIVOS SALIDA 
print("Saving Results ...")
path_res="%s%s_results.csv" %(path_out,nombre)
a.to_csv(path_res,index=False)
del a
print('Done')

