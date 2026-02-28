import numpy as np
import matplotlib.pyplot as plt
from data import DATA_ALL
from data_profesional import DATA_PROF
import matplotlib as mpl
import matplotlib
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['pdf.use14corefonts'] = False

matplotlib.rcParams.update({'font.size': 12})

def get_vals_and_stds(data, parameter):
    """
    Calcula valores y devuelve listas completas para encontrar 
    máximos y mínimos con sus respectivos desvíos.
    """
    vals = []
    stds = []
    
    for dic in data:
        # Asumimos que la lista es [media, desvio]
        vals.append(dic[parameter][0]) 
        stds.append(dic[parameter][1]) 
    
    return np.array(vals), np.array(stds)

# --- CONFIGURACIÓN DE PARÁMETROS ---
param = "T30"
unit = "[seg]"
data = DATA_ALL 

# 1. Obtener Medias y Desvíos de todas las variantes
vals, stds = get_vals_and_stds(data, param)

# Encontrar índices del mínimo y máximo valor medio
min_idx = np.argmin(vals)
max_idx = np.argmax(vals)

# Extraer valores (media) y sus errores (desvío) correspondientes
min_val = np.round(vals[min_idx], 2)
min_std = stds[min_idx]

max_val = np.round(vals[max_idx], 2)
max_std = stds[max_idx]

# 2. Obtener Base ITW (Media y Desvío)
raw = {
        'STOI': [0.99, 0.01],
        'PESQ': [2.82, 0.72],
        'SI-SDR': [19.08, 8.93],
        'T30': [0.98, 0.57],
        'C50': [15.86, 5.48],
        'D50': [90.77, 10.79],
        'SNR': [19.08, 8.93],
        'F0': [200.01, 103.57],
        }

base_val = raw[param][0]
base_std = raw[param][1] # Desvío estándar de la base

# 3. Obtener Profesional (Media y Desvío)
data_prof = DATA_PROF 
prof_val = data_prof[param][0]
prof_std = data_prof[param][1] # Desvío estándar profesional

# Datos para graficar
labels = ['Dataset\nbase (ITW)', 'Peor variante\nde la cadena', 'Mejor Variante\nde la cadena', 'Dataset\nprofesional']
values = [base_val, max_val, min_val, prof_val]
errors = [base_std, max_std, min_std, prof_std] # Lista de desvíos para las barras de error

# Definición de Colores (Hex codes o nombres)
# Color 1: Base, Color 2: Variantes, Color 3: Profesional
bar_colors = ['#1f77b4', '#ff7f0e', '#ff7f0e', '#2ca02c'] 

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 5)) # Aumenté un poco el alto para los labels de 2 líneas

# Gráfico de barras con Colores y Barras de Error
# yerr: define el error hacia arriba y abajo
# capsize: pone el "tope" horizontal en la línea de error
bars = ax.bar(range(len(labels)), values, 
              yerr=errors, 
              capsize=5, 
              color=bar_colors,
              alpha=0.9) # Un poco de transparencia opcional

# Etiquetas de eje y título
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel(f'Valor medio de {param} {unit}')

# Mostrar los valores sobre cada barra
for bar in bars:
    height = bar.get_height()
    
    # xy=(x_centro, height): La coordenada base es el centro superior de la barra (la media)
    # xytext=(12, 0): Desplazamos el texto 12 puntos a la derecha
    # ha='left': Alineamos el texto a la izquierda para que empiece desde ese punto
    # va='center': Centrado verticalmente respecto a la línea de la media
    
    ax.annotate(f'{height}', 
                xy=(bar.get_x() + bar.get_width() / 2, height + 0.06),
                xytext=(12, 0),  
                textcoords="offset points",
                ha='left', va='center')

# plt.ylim(1, 4.2)
plt.tight_layout()

out_path = os.path.join("graph_finales", f"Comparación tipos datasets {param}.pdf")
plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
plt.show()