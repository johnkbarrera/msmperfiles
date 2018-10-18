
# LIBRERIAS
import numpy as np
import datetime
from datetime import date
import json
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import os, sys


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
path_in=ConfigSectionMap("ruta")['entradas']
path_out=ConfigSectionMap("ruta")['salidas']
conf_header=ConfigSectionMap("file")['header']
conf_data= ConfigSectionMap("file")['data']
nombre = conf_data[:-4]

# RUTAS DE ARCHIVOS ENTRADAS 
p_header="%s%s" %(path_in,conf_header)
p_data="%s%s" %(path_in,conf_data)


# CARGANDO FILE
print("Loading Data ...")
header = pd.read_csv(p_header)
data = pd.read_csv(p_data , header = None, low_memory=False)
data.columns = list(header)


print("Procesing Data ...")
data['AÑO'] = data['F_TRAFICO'].apply(lambda fecha: int(fecha[6:]))
data.head()

# DEFINIMOS LA LISTA DE CLIENTES
clientes =  data.groupby('CO_ID').CO_ID.count().index
print("Numero de Clientes: ",len(clientes))

#  DEFININIMOS FUNCION PROCESAR_U 
def procesar_u(user, tipo_eth = False):    
    uid=list(user['CO_ID'])[0]              # Cliente_id
    years = set(list(user['AÑO']))              # Lista los años en que se tiene TXs registradas
    anni = {year:{} for year in list(years)}    # definimos anni como una lista 
    
    # para cada fila, es decir, cada TXs del cliente)
    for dat in  range(0,len(user)):
        año = int(user.iloc[dat]['AÑO'])
        fecha = user.iloc[dat]['F_TRAFICO']
        fecha = pd.to_datetime(fecha, format='%d/%m/%Y', errors='coerce')
        mes = fecha.month
        dia = fecha.day        
        turn = user.iloc[dat]['HORA']
        
        week=str(datetime.datetime(año,mes,dia).isocalendar()[1])
        if len(week)==1:
            week = '0' + week
        weekday=datetime.datetime(año,mes,dia).weekday()
        
        # Si la semana no existe en el año
        if not(week in anni[año]):
            anni[año][week] = {}
        # Si el billcycle no existe en la semana y año
        if not (weekday in anni[año][week]):
            anni[año][week][weekday]={}  #
        # Si el turno no existe en el mccg,semana y año
        
        anni[año][week][weekday][turn]=list(user.iloc[dat,6:-1]) 
                
    return uid,anni


# PROCESAMOS U DE CADA CLIENTE    
profiles={}           # Creamos lista de prefiles
contador=0 
print("Number of rows "+str(len(data))) 

# Para cada cliente
for cliente in clientes:
    cliente_i= data[data['CO_ID'] == cliente]       # filtramos dataset solo para el cliente i
    results=procesar_u(cliente_i, tipo_eth=False)          # procesamos u del usuario i
    profiles[results[0]]=results[1]                     # cargamos lista de indice "uid" con la data del cliente(json)
    contador += 1
    if contador % 5000 == 1:
        print("vamos en el cliente ",contador)


# Creamos la cabecera dinámica donde se guardaran todos los footprints generados
cabecera = 'CO_ID,YEAR,WEEK,PROFILE_ID,SIZE'
for i in range(7):      # numero de dias
    for j in range(4):                # numero de turnos
        for k in range(38):            # numero de planes
            cabecera = cabecera+','+'D'+str(i)+'T'+str(j)+'P'+str(k)
cabecera = cabecera+'\n'


# RUTAS DE ARCHIVOS SALIDAS 
individual_footprint="%s%s.footprint" %(path_out,nombre)


# GUARDANDO ARCHIVOS
fw=open(individual_footprint,'w')  

fw.write(cabecera)                    # Escribimos la cabecera

# Para cada uid (cliente)
footprints=0
for uid in profiles:   
    profile_id=0
    # En cada año
    for year in profiles[uid]:       
        # Por cada semana
        for week in profiles[uid][year]:    
                             
            temp=np.zeros(7*4*38) 
            # Por cada semana
            for weekday in profiles[uid][year][week]:
                temp2=np.zeros(4*38) 
                # Por cada turno
                for turno in profiles[uid][year][week][weekday]:                        
                    # print(uid,year,week,weekday,turno,len(profiles[uid][year][week][weekday][turno]))
                    temp2[turno*38:(turno+1)*38] = profiles[uid][year][week][weekday][turno]
                temp[weekday*len(temp2):(weekday+1)*len(temp2)] = temp2
          
            # Escribimos los datos del primer comportamiento (Tensor)    
            txt = ''+str(uid)+','+str(year)+','+str(week)+','+str(profile_id)+','+str(sum(temp))
            for i in range(len(temp)):
                txt = txt +','+str(temp[i])
            fw.write(txt +'\n')

            profile_id += 1   
            footprints += 1  
            
    fw.flush()
fw.close()               
print ("number of footprint: "+str(footprints))



print("Done")

