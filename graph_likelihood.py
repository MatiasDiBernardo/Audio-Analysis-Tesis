import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

def plot_kde_matplotlib(archivos_csv, etiquetas, output_filename="kde_loglikelihood_2.pdf"):
    """
    Grafica la estimación de densidad KDE usando matplotlib puro y scipy.
    
    Parámetros:
    - archivos_csv: Lista de rutas a los archivos CSV consolidados.
    - etiquetas: Lista de nombres para la leyenda del gráfico.
    - output_filename: Nombre del archivo de salida.
    """
    
    # Configuración base de la figura
    plt.figure(figsize=(10, 6))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, Naranja, Verde
    
    # Para asegurar que el eje X tenga sentido, guardamos los mínimos y máximos globales
    x_min_global = float('inf')
    x_max_global = float('-inf')

    # Diccionario temporal para guardar los datos y calcular el eje X global después
    datos_extraidos = []

    # 1. Leer los datos de todos los archivos
    for archivo, etiqueta, color in zip(archivos_csv, etiquetas, colores):
        ruta = Path(archivo)
        
        if not ruta.exists():
            print(f"Error: No se encontró el archivo {ruta}")
            continue
            
        try:
            df = pd.read_csv(ruta)
            if 'logp' not in df.columns:
                print(f"Advertencia: El archivo {ruta.name} no tiene la columna 'logp'.")
                continue
                
            datos = df['logp'].dropna().values
            
            # Actualizar límites globales para el eje X (con un margen extra)
            x_min_global = min(x_min_global, datos.min() - 0.5)
            x_max_global = max(x_max_global, datos.max() + 0.5)
            
            datos_extraidos.append((datos, etiqueta, color))
            print(f"Procesado {ruta.name}: {len(datos)} audios.")
            
        except Exception as e:
            print(f"Error al procesar {ruta.name}: {e}")

    # Eje X común para evaluar todas las densidades (500 puntos de resolución)
    x_grid = np.linspace(x_min_global, x_max_global, 500)

    # 2. Calcular KDE y graficar
    for datos, etiqueta, color in datos_extraidos:
        # Inicializar el estimador de densidad Gaussiana
        kde = gaussian_kde(datos)
        
        # Evaluar la densidad en nuestro eje X común
        y_kde = kde(x_grid)
        
        # Graficar la línea principal
        plt.plot(x_grid, y_kde, label=etiqueta, color=color, linewidth=2.5)
        
        # Rellenar el área bajo la curva
        plt.fill_between(x_grid, y_kde, alpha=0.15, color=color)

    # 3. Configuraciones estéticas (Emulando el estilo limpio académico)
    ax = plt.gca()
    
    # Cuadrícula suave
    ax.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
    
    # Eliminar los bordes superior y derecho para mayor limpieza visual
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    # Etiquetas y título
    plt.xlabel('Verosimilitud (Log-Likelihood)', fontsize=14, color='#333333')
    plt.ylabel('Densidad de Probabilidad', fontsize=14, color='#333333')
    
    plt.xticks(fontsize=12, color='#333333')
    plt.yticks(fontsize=12, color='#333333')
    
    # Leyenda
    plt.legend(title='Variante de Preprocesamiento', title_fontsize=12, fontsize=11, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.xlim(0.85, 4)
    
    # Exportar y mostrar
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"\nGráfico guardado exitosamente como '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    # Ajusta estas rutas a las de tus archivos consolidados
    archivos_a_comparar = [
        "./Likelihood/Demucs_NISQA_3_consolidado.csv",
        # "./Likelihood/NoDenoising_NISQA_4-2_consolidado.csv",
        "./Likelihood/DeepFilterNet_DNSMOS_2-7_consolidado.csv",
        "./Likelihood/NoDenoising_DNSMOS_2-7_consolidado.csv",
    ]
    
    nombres_variantes = [
        "Demucs (Filtro NISQA 3)",
        "Sin Denoising (Filtro DNSMOS 2.7)",
        "DeepFilterNet (Filtro DNSMOS 2.7)"
    ]
    
    plot_kde_matplotlib(archivos_a_comparar, nombres_variantes)