
# LIBRERIAS
import numpy as np
import datetime
from datetime import date
import json
# import pylab
import pandas as pd
import os, sys
from sklearn.preprocessing import normalize

# import matplotlib.pylab as plt
import matplotlib
matplotlib.use('agg')
import pylab as plt

matplotlib.pyplot.switch_backend('agg')

plt.switch_backend('agg')
# import matplotlib.pylab as plt


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
path_in=ConfigSectionMap("ruta")['entradas']
conf_header=ConfigSectionMap("file")['header']
nombre = ConfigSectionMap("file")['data'][:-4]


# RUTAS DE ARCHIVOS ENTRADAS 
p_header="%s%s" %(path_in,conf_header)

footprint="%s%s.footprint" %(path_out,nombre)
individual_clusters="%s%s.individual_footprint.clusters" %(path_out,nombre)
individual_labels="%s%s.individual_footprint.labels" %(path_out,nombre)  
collective_clusters="%s%s.collective_footprint.clusters" %(path_out,nombre)
collective_labels="%s%s.collective_footprint.labels" %(path_out,nombre)  
path_res = "%s%s_results.csv" %(path_out,nombre)  

# CARGANDO FILE
print("Loading Data ...")
header = pd.read_csv(p_header)
result = pd.read_csv(path_res, sep=",", header=0,  dtype={'week': str,'year': str}, low_memory=False)
print("Datos Cargados ..")
print(len(result))
result.head(5)

# Variables de Apoyo - Clientes
clientes = result.CO_ID.unique()      # Lista de Clientes


# DISTRIBUCION DE CLIENTES POR CLUSTER

# Por Cluster Collectivo
print(" 1th plot ...")

temporal= result.groupby(['COLLECTIVE_CLUSTER'], as_index=False, sort=True)['CO_ID'].count()
print("=>  ",str(len(temporal))," clusters")
fig = plt.figure()
fig.patch.set_alpha(0.5)
ax = fig.add_subplot(111)
ax.patch.set_alpha(0.5)
data = ax.bar(temporal.index, temporal['CO_ID'], align='center')
ax.grid(color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Cluster Colectivo')
ax.set_xticks(temporal.index)
ax.set_ylabel('Número de Comportamiento')
tag =[]
for j in range(len(temporal)):
    nn = 'C '+ str(j)+''
    tag.append(nn)
del(nn)
ax.set_xticklabels(tag, rotation=70)
title = "Distribición de Clusters"
ax.set_title(title)
grafico="%s%s_Collective_Clusters.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
del(temporal)
print('Done')


# Por Cluster  Individual
print(" 2th Plot ...")

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.00*height,
                '%d' % int(height),
                ha='center', va='bottom',  rotation=80, fontweight='bold', color='Black')


#   -------------------------------------------------------------------------------------------------
temporal= result.groupby(['INDIVIDUAL_CLUSTER'], as_index=False, sort=True)['CO_ID'].count()
print("=>  ",str(len(temporal))," clusters")
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111, frameon=True)
data = ax.bar(temporal.index, temporal['CO_ID'], align='center')
ax.grid(color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Cluster Individual')
ax.set_xticks(temporal.index)
ax.set_ylabel('Número de Comportamiento')
tag =[]
for j in range(len(temporal)):
    nn = 'C '+ str(j)+''
    tag.append(nn)
    del(nn)

ax.set_xticklabels(tag, rotation=70)
autolabel(data)
title = "Distribición de Cluster"
ax.set_title(title)
grafico="%s%s_Individual_Clusters.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
del(temporal)
print('Done')


# CLUSTERS INDIVIDUALES PER COLLECTIVES

print(" 3th Plot ...")
# matrix of clusters

temporal= result.groupby(['COLLECTIVE_CLUSTER','INDIVIDUAL_CLUSTER'], as_index=False, sort=True)['CO_ID'].count()

matrix = np.zeros((len(temporal.COLLECTIVE_CLUSTER.unique()), len(temporal.INDIVIDUAL_CLUSTER.unique())))
for x in temporal.COLLECTIVE_CLUSTER.unique():
    temporal_i = temporal[(temporal.COLLECTIVE_CLUSTER == x)]
    temporal_i = temporal_i.groupby(['INDIVIDUAL_CLUSTER'], as_index=False, sort=True)['CO_ID'].sum().T
    a = list(temporal_i.values[1])
    for i in range(len(a)):
        matrix[x][i]=a[i]
    del(a,temporal_i)
temporal = pd.DataFrame(matrix)
# temporal.head()

# Variables de Apoyo - Planes
H = matrix
fig = plt.figure(figsize=(10, 30))
ax = fig.add_subplot(111)
ax.set_title('colorMap')
im = plt.imshow(H, cmap="Blues")
ax.set_aspect('equal')

for i in range(len(temporal)):
    for j in range(len(temporal.iloc[1])):
        text = ax.text(j, i, round(matrix[i, j],3), ha="center", va="center", color="Black", rotation=45)

ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)
ax.patch.set_alpha(0)
ax.set_frame_on(True)
plt.colorbar(orientation='vertical')
grafico="%s%s_Both_Clusters.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
print('Done')

# Comportamiento de los clusters en el Tiempo
print(" 4th plot ...")

from datetime import datetime
def fromisocalendar(y,w,d):
    date = datetime.strptime( "%04dW%02d-%d"%(y,w-1,d), "%YW%W-%w")
    aa = date.year
    mm = date.month
    dd = date.day
    return str(aa)+"-"+str(mm)+"-"+str(dd)

import matplotlib._color_data as mcd
box = dict(facecolor='#cccc00', pad=4, alpha=0.8)


longitud = len(result.groupby(['COLLECTIVE_CLUSTER'], as_index=False, sort=True)['CO_ID'].count())
temporal = result.groupby(['YEAR','WEEK','COLLECTIVE_CLUSTER'], as_index=False, sort=True)['CO_ID'].count()
temporal['fecha'] = temporal['YEAR'].astype(str).str.cat(temporal['WEEK'].astype(str), sep='-')
temporal = temporal[['fecha','COLLECTIVE_CLUSTER','CO_ID']]
helper = temporal.groupby(['fecha']).count()
helper['fecha'] = helper.index

fig = plt.figure(figsize=(18,8))
ax = fig.add_subplot(111, frameon=True)
x = list(temporal['fecha'].drop_duplicates())
x1 = x
for j in range(len(x)):
    # x1[j] = fromisocalendar(int(x[j][:4]),int(x[j][5:len(x[j])]),2)
    x1[j] = x[j]

ax.set_xticklabels(x1, rotation=80)

# ------------ COLORES ---------------------
overlap = mcd.XKCD_COLORS
colores = ["" for x in range(len(overlap))]
for j, n in enumerate(sorted(overlap, reverse=True)):
    cor = mcd.XKCD_COLORS[n].upper()
    colores[j] = cor

#legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

from datetime import datetime

for k in range(longitud):
    cluster_k = temporal[temporal['COLLECTIVE_CLUSTER']==k]
    cluster_k1 = helper.set_index('fecha').join(cluster_k.set_index('fecha'), lsuffix='_x', rsuffix='_y')
    cluster_k1 = cluster_k1[['CO_ID_y']]
    cluster_k1 = cluster_k1.fillna(0)
    y = cluster_k1['CO_ID_y']
    #x = cluster_k1.index
    ax.plot(x,y,'-',color=colores[k*9],label="C "+str(k))



ax.legend(loc='upper right', shadow=True, fontsize='x-small')
title = 'Variación en el Tiempo del Comportamiento de los Clientes (Cluster collectivo)'
ax.set_title(title)
ax.set_xlabel('Año - Semana (Fecha)', bbox=box)
ax.set_ylabel('Cantidad de Clientes', bbox=box)

grafico="%s%s_Collective_Clusters_Time.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)

del(temporal,helper)
print('Done')


# Distribucion de Semanas con "n" Clientes
print(" 5th plot ...")

temp= result.groupby(['PROFILE_ID'], as_index=False, sort=True)['CO_ID'].count()

print("=>  ",str(len(temp))," clusters")

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot(111, frameon=True)
ax.bar(temp.index, temp['CO_ID'], align='center')
ax.grid(color='gray', linestyle='--', linewidth=1)

ax.set_xlabel('Numero de Semanas que tiene Actividad')
ax.set_xticks(temp.index)

ax.set_ylabel('Num. Clientes')
tag =[]
for j in range(len(temp)):
    nn = ''+ str(j)+''
    tag.append(nn)
    del(nn)

ax.set_xticklabels(tag, rotation=80)

title = "Semanas en las que hay clientes "
ax.set_title(title)
grafico="%s%s_Actividades_Clientes_time.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
del(temp)
print('Done')


# Frecuencia  de clientes en cada semana
print(" 6th plot ...")

temp = result.groupby(['CO_ID'], as_index=False, sort=True)['SIZE'].count()
temp = temp.groupby(['SIZE'], as_index=False, sort=True).count()

print("=>  ",str(len(temp))," clusters")

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot(111, frameon=True)
ax.bar(temp.index, temp['CO_ID'], align='center')
ax.grid(color='gray', linestyle='--', linewidth=1)

ax.set_xlabel('Numero de Semanas que tiene Actividad')
ax.set_xticks(temp.index)
"""
ax.set_ylabel('Num. Clientes')
tag =[]
for j in range(len(temp)):
    nn = ''+ str(j)+''
    tag.append(nn)
    del(nn)

ax.set_xticklabels(tag, rotation=80)
    """
title = "Histograma "
ax.set_title(title)
grafico="%s%s_Frecuencia_Cliente.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
#plt.show()
del(temp)
print('Done')


# CARGANDO CENTROIDES

print(" 7th plot ...")
# Abrimos el File para cargar los centroides
centroides = pd.read_csv(collective_clusters, sep=";", header=0,  dtype={'WEEK': str,'YEAR': str})


# Seleccionamos Centroide o cluster deseado
num_cluster = int(ConfigSectionMap("detalles")['centroide'])
# extraemos
centroides_i = centroides[(centroides.COLLECTIVE_CLUSTER == num_cluster)]
centroides_i


# In[16]:


# Definimos parametros
title = centroides_i.iloc[:,1:].columns
datos = centroides_i.iloc[:,1:].values
datos = datos[0]
turnos = ["Mad", "Mañ", "Tar", "Noc"]
dias = ["L", "M", "M", "J", "V", "S", "D"]
planes_tag = header.columns
planes_tag = planes_tag[6:len(planes_tag)]

# In[17]:


td = np.zeros((len(turnos),len(dias)))
tm = np.zeros((len(turnos),38))
dm = np.zeros((len(dias),38))

# Agrupamos por Planes
contador = 0
for m in range(38):
    for t in range(len(turnos)):        # filas
        for d in range(len(dias)):    #columnas

            td[t][d] += datos[contador]
            contador += 1

# Agrupamos por Dias
contador = 0
for m in range(38):
    for t in range(len(turnos)):        # filas
        for d in range(len(dias)):    #columnas
            tm[t][m] += datos[contador]
            contador += 1

# Agrupamos por Turnos
contador = 0
for m in range(38):
    for t in range(len(turnos)):        # filas
        for d in range(len(dias)):    #columnas
            dm[d][m] += datos[contador]
            contador += 1


# Plot de centroide i

from mpl_toolkits.axes_grid1 import make_axes_locatable
box = dict(facecolor='#ff9999', pad=4, alpha=0.2)

fig, axes = plt.subplots(3, 1, figsize=(15,30))
#fig, axes = plt.subplots(3, 1)
ax1, ax2 , ax3= axes.flatten()
plt.subplots_adjust(top=0.8, bottom=0.5, left=0.10, right=0.95, hspace=0.5, wspace=0.1)

title_i = "Centroide"+str(num_cluster)
plt.title(title_i)

im1 = ax1.imshow(td, cmap="YlGn")
ax1.set_xticks(np.arange(len(dias)))
ax1.set_yticks(np.arange(len(turnos)))
ax1.set_xticklabels(dias)
ax1.set_yticklabels(turnos)
for i in range(len(turnos)):
    for j in range(len(dias)):
        text = ax1.text(j, i, round(td[i, j],3), ha="center", va="center", color="Black", rotation=45)
ax1.set_title('Cantidad de TXs mas Frecuentes')
ax1.set_xlabel('Dias', bbox=box)
ax1.set_ylabel('Turnos', bbox=box)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="10%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
ax1.xaxis.set_visible(True)

im2 = ax2.imshow(tm, cmap="YlGn")
ax2.set_xticks(np.arange(38))
ax2.set_yticks(np.arange(len(turnos)))
ax2.set_xticklabels(range(38))
ax2.set_xticklabels(planes_tag, rotation=70)
ax2.set_yticklabels(turnos)
for i in range(len(turnos)):
    for j in range(38):
        text = ax2.text(j, i, round(tm[i, j],3), ha="center", va="center", color="Black", rotation=45)
ax2.set_title('Cantidad de TXs mas Frecuentes')
ax2.set_xlabel('Planes', bbox=box)
ax2.set_ylabel('Turnos', bbox=box)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="10%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)
ax2.xaxis.set_visible(True)

im3 = ax3.imshow(dm, cmap="YlGn")
ax3.set_xticks(np.arange(38))
ax3.set_yticks(np.arange(len(dias)))
ax3.set_xticklabels(range(38))
ax3.set_xticklabels(planes_tag, rotation=70)
ax3.set_yticklabels(dias)
for i in range(len(dias)):
    for j in range(38):
        text = ax3.text(j, i, round(dm[i, j],3), ha="center", va="center", color="Black", rotation=45)
ax3.set_title('Cantidad de TXs mas Frecuentes')
ax3.set_xlabel('Planes', bbox=box)
ax3.set_ylabel('Dias', bbox=box)
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="10%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3)
ax3.xaxis.set_visible(True)

grafico="%s%s_Centroide_%s.png" %(path_out,nombre,num_cluster)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
print('Done')