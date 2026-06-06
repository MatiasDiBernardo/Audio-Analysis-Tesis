import numpy as np
from data import DATA_NISQA, DATA_ALL

def calculate_metrics(pro):
    """
    Calculo de métricas globales a partir de las métricas individuales.
    
    :param pro: Diccionario con métricas del dataset prosesado.
    """

    # Original dataset values
    raw =  {'HOURS': 24.30,
            'STOI': [0.99, 0.01],
            'PESQ': [2.82, 0.72],
            'SI-SDR': [19.08, 8.93],
            'T30': [0.98, 0.57],
            'C50': [15.86, 5.48],
            'D50': [90.77, 10.79],
            'SNR': [19.08, 8.93],
            'F0': [200.01, 103.57],
            }

    # Pesos de la métrica global
    w1 = 0.1  # DR
    w2 = 0.5  # SQ
    w3 = 1.2  # AP
    w4 = 4  # SD
    
    # Cálculo de métrica compuesta
    DR = 1 - (pro['HOURS']/raw['HOURS'])
    SQ = (raw['PESQ'][0]/pro['PESQ'][0]) + (raw['SI-SDR'][0]/pro['SI-SDR'][0]) + (raw['SNR'][0]/pro['SNR'][0])
    AP = (pro['T30'][0]/raw['T30'][0]) + (raw['C50'][0]/pro['C50'][0])
    SD = np.abs(1 - (pro['F0'][1]/raw['F0'][1])) + (pro['MCD'][0]/10)
    TOT = (DR * w1 + SQ * w2 + AP * w3 + SD * w4)

    return [DR * w1, SQ * w2, AP * w3, SD * w4, TOT]

def calculate_all_configs(dics_vals):
    """
    Calcula las métricas a partir de una lista de diccionarios con metricas por configuración.
    
    :param dics_vals: Lista de diccionarios con información.
    """

    rta = {}
    
    for dic in dics_vals:
        mos_type = dic['MOS Type']
        thshold = dic['MOS Threshold']
        den = dic["Denoising"]
        config = f"{mos_type} = {thshold} | Den = {den}"
        
        metric_vals = calculate_metrics(dic) 
        rta[config] = metric_vals
    
    return rta

# Cálculo de métricas globales para cada configuración 
data = DATA_ALL 
dic_rta = calculate_all_configs(data)
items_sorted = sorted(dic_rta.items(), key=lambda kv: kv[1][4])
max_len = max(len(k) for k, _ in items_sorted) 
val_width = 4

dr_list = []
sq_list = []
ap_list = []
sd_list = []

print(" ")
print(f"Las métricas globales para cada configuración son:")
for k, v in items_sorted:
    format_vals = f"DR: {np.round(v[0], 2)} | SQ: {np.round(v[1], 2)} | AP: {np.round(v[2], 2)} | SD: {np.round(v[3], 2)}| TOT: {np.round(v[4], 2)}"
    print(f"{k:<{max_len}}: {format_vals}")

    dr_list.append(v[0])
    sq_list.append(v[1])
    ap_list.append(v[2])
    sd_list.append(v[3])

print("Variaciones")
print(f"DR std: {np.std(np.array(dr_list))} | Delta Max: {max(dr_list) -  min(dr_list)}")
print(f"SQ std: {np.std(np.array(sq_list))} | Delta Max: {max(sq_list) -  min(sq_list)}")
print(f"AP std: {np.std(np.array(ap_list))} | Delta Max: {max(ap_list) -  min(ap_list)}")
print(f"SD std: {np.std(np.array(sd_list))} | Delta Max: {max(sd_list) -  min(sd_list)}")
