

# ANALISIS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
#import matplotlib.pylab as plt
import datetime
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import os


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

file_path = "../resultados/"
directory = os.path.dirname(file_path)
try:
    os.stat(directory)
except:
    os.mkdir(directory)  

# RUTAS DE ARCHIVOS ENTRADAS 
p_header="%s%s" %(path_in,conf_header)
p_data="%s%s" %(path_in,conf_data)


# CARGANDO FILE
print("Loading Data ...")
header = pd.read_csv(p_header)
data = pd.read_csv(p_data , header = None, dtype={'GB_TOTAL': float,'HORA': int}, low_memory=False)
#data = pd.read_csv(p_data , header = None,  dtype=None, low_memory=False)
data.columns = list(header)

print("Procesing Data ...")
data["BILLCYCLE"] = data["BILLCYCLE"].fillna(9)
data['F_TRAFICO'] = pd.to_datetime(data['F_TRAFICO'], format='%d/%m/%Y', errors='coerce')
data['WEEK_DAY'] = data['F_TRAFICO'].dt.dayofweek


print("Ploting...")
fig, axes = plt.subplots(1, 4, figsize=(14,2), squeeze = True)
ax1, ax2, ax3, ax4 = axes.flatten()
plt.subplots_adjust(top=1,wspace = 0.3)

print(" 1th plot ...")
y = data.groupby(['CO_ID']).count()      # Us por cliente
y = y["BILLCYCLE"].tolist()
mu = np.mean(y)
median = np.median(y)
sigma = np.var(y)
text = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
ax1.hist(y,10)
ax1.set_title('Sesiones por Cliente')
ax1.set_xlabel('Sessions')
ax1.set_ylabel('N° de Clientes')
ax1.text(0.15, 0.85, text, transform=ax1.transAxes, fontsize=10, verticalalignment='top')

print(" 2th plot ...")
tag =[' ','Madrugada','Mañana','Tarde','Noche']  
y = data.groupby(['HORA']).sum()['GB_TOTAL']
x = y.index
y = y.values
# ax2.grid(True)
ax2.bar(x,y,0.7,color='green')
ax2.set_xticklabels(data['HORA'].unique())
ax2.set_xticklabels(tag, rotation=70)
ax2.set_title('Consumo por Turnos')
ax2.set_xlabel('Horas')
ax2.set_ylabel('GBs')

print(" 3th plot ...")
ax3.grid(b=None, which='minor', axis='both')
tag =[' ','Lun','Mie','Vie','Dom'] 
y = data.groupby(['WEEK_DAY']).sum()['GB_TOTAL']
ax3.bar(list(y.keys()),list(y.values),  align='center', color='Orange')
ax3.plot(y.index,y.values,'-',color='#ff0000')
ax3.plot(y.index,y.values,'ob',color='#990000')
ax3.set_xticklabels(tag, rotation=70)
ax3.set_title('Consumo Mobil por Dia')
ax3.set_xlabel('Dias')
ax3.set_ylabel('GBs')

print(" 4th plot ...")
y = data.groupby(['BILLCYCLE']).sum()['GB_TOTAL']
ax4.bar(list(y.index),y.values,color='Red',label="y.index")
ax4.set_title('Consumo Mobil por Billcycle')
ax4.set_xlabel('Billcycle')
ax4.set_ylabel('GBs')

print("Saving ...")
grafico="%s%s_Analisis.png" %(path_out,nombre)
plt.savefig(grafico,dpi = 1000, bbox_inches='tight')
del(grafico)
print("Done")

