import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_kde_comparativo(archivos_csv, etiquetas, output_filename="kde_loglikelihood.pdf"):
    """
    Grafica la estimación de densidad KDE para múltiples archivos de resultados.
    
    Parámetros:
    - archivos_csv: Lista de rutas a los archivos CSV consolidados.
    - etiquetas: Lista de nombres para la leyenda del gráfico.
    - output_filename: Nombre del archivo de salida (recomendado .pdf para LaTeX).
    """
    
    # Configuración del estilo de seaborn para gráficos académicos
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6))
    
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, Naranja, Verde
    
    # Leer y graficar cada variante
    for archivo, etiqueta, color in zip(archivos_csv, etiquetas, colores):
        ruta = Path(archivo)
        
        if not ruta.exists():
            print(f"Error: No se encontró el archivo {ruta}")
            continue
            
        try:
            # Leer el CSV
            df = pd.read_csv(ruta)
            
            if 'logp' not in df.columns:
                print(f"Advertencia: El archivo {ruta.name} no tiene la columna 'logp'.")
                continue
                
            # Extraer los datos y limpiar posibles NaNs
            datos_logp = df['logp'].dropna()
            
            # Graficar el KDE
            sns.kdeplot(
                data=datos_logp,
                label=etiqueta,
                color=color,
                linewidth=2.5,
                fill=True,
                alpha=0.15 # Transparencia del relleno
            )
            
            print(f"Procesado {ruta.name}: {len(datos_logp)} audios.")
            
        except Exception as e:
            print(f"Error al procesar {ruta.name}: {e}")

    # Configuraciones visuales del gráfico
    plt.xlabel('Verosimilitud (Log-Likelihood)', fontsize=14, fontweight='bold')
    plt.ylabel('Densidad de Probabilidad', fontsize=14, fontweight='bold')
    
    # Ajustar el tamaño de los números en los ejes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Configurar la leyenda
    plt.legend(title='Variante de Preprocesamiento', title_fontsize=12, fontsize=11, loc='upper right')
    
    # Ajustar márgenes
    plt.tight_layout()
    
    # Guardar gráfico en formato vectorial
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"\nGráfico guardado exitosamente como '{output_filename}'.")
    
    # Mostrar el gráfico en pantalla
    plt.show()

if __name__ == "__main__":
    # Rutas a los CSVs consolidados generados por el script de métricas anterior
    # Asegúrate de que las rutas coincidan con la ubicación real de tus archivos
    archivos_a_comparar = [
        "./results/Demucs_NISQA_4-2_consolidado.csv",
        "./results/NoDenoising_DNSMOS_2-7_consolidado.csv",
        "./results/DeepFilterNet_DNSMOS_2-7_consolidado.csv"
    ]
    
    # Nombres limpios que aparecerán en la leyenda del gráfico en tu tesis
    nombres_variantes = [
        "Demucs (Filtro NISQA)",
        "Sin Denoising (Filtro DNSMOS)",
        "DeepFilterNet (Filtro DNSMOS)"
    ]
    
    plot_kde_comparativo(archivos_a_comparar, nombres_variantes)