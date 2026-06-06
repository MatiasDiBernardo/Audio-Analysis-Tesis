# Itera sobre TTS samples y evalúa la calidad de los samples (MOS)
# Armar un diccionario de MOS resultante por configuración
# Gráfico de comparaciones entre configuraciones y MOS

import matplotlib.pyplot as plt
import numpy as np
from data_mos import config_TTS_MOS, config_TTS_MOS_es
from data_computed_datasets import config_obj_metrics

# Variable to select which objective metric to plot
obj_metrics = 'CS'  # Options: 'RD', 'CS', 'CA', 'DH', 'TOT'

def plot_mos_vs_objective_metric(obj_metrics_type='RD'):
    """
    Plot scatter plot comparing objective metrics against TTS MOS results.
    
    Parameters:
    -----------
    obj_metrics_type : str
        Type of objective metric to plot. Options: 'RD', 'CS', 'CA', 'DH', 'TOT'
    
    The plot shows:
    - X-axis: Objective metric value (sorted)
    - Y-axis: MOS mean value
    - Error bars: MOS standard deviation
    """
    
    # Extract common configurations that exist in both dictionaries
    common_configs = set(config_TTS_MOS_es.keys()) & set(config_obj_metrics.keys())
    
    # Prepare data: extract metric values and MOS values
    metric_values = []
    mos_means = []
    mos_stds = []
    config_names = []
    
    for config in common_configs:
        metric_val = config_obj_metrics[config][obj_metrics_type]
        mos_mean, mos_std = config_TTS_MOS_es[config]
        
        metric_values.append(metric_val)
        mos_means.append(mos_mean)
        mos_stds.append(mos_std)
        config_names.append(config)
    
    # Sort by metric values (X-axis)
    sorted_indices = np.argsort(metric_values)
    metric_values_sorted = [metric_values[i] for i in sorted_indices]
    mos_means_sorted = [mos_means[i] for i in sorted_indices]
    mos_stds_sorted = [mos_stds[i] for i in sorted_indices]
    config_names_sorted = [config_names[i] for i in sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot scatter with error bars
    ax.errorbar(metric_values_sorted, mos_means_sorted, yerr=mos_stds_sorted, 
                fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
                ecolor='darkblue', color='dodgerblue', alpha=0.7, label='MOS ± STD')
    
    # Add labels for each point (configuration names)
    for i, config_name in enumerate(config_names_sorted):
        ax.annotate(config_name, 
                   (metric_values_sorted[i], mos_means_sorted[i]),
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center', 
                   fontsize=8,
                   alpha=0.7)
    
    ax.set_xlabel(f'Objective Metric: {obj_metrics_type}', fontsize=12, fontweight='bold')
    ax.set_ylabel('TTS MOS (Mean ± STD)', fontsize=12, fontweight='bold')
    ax.set_title(f'Correlation: {obj_metrics_type} vs MOS\n(sorted by metric value)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display correlation coefficient
    correlation = np.corrcoef(metric_values_sorted, mos_means_sorted)[0, 1]
    print(f"Correlation coefficient ({obj_metrics_type} vs MOS): {correlation:.4f}")
    print(f"Number of configurations plotted: {len(common_configs)}")

# Example usage:
if __name__ == "__main__":
    # Change obj_metrics variable to plot different objective metrics
    plot_mos_vs_objective_metric(obj_metrics)
