import numpy as np
from data import DATA_NISQA, DATA_DNSMOS, DATA_ALL

def calc_porcentual_diff(data, parameter, up_parameter, complement, mean=0):
    """Calculate the porcentual difference for the given parameter respect to the original
    an return an array with the order and the corresponding configuration.

    Args:
        data (list): List of dictionaries with data.
        parameter (string): Parameter to evaluete the distance.
        up_patameter(boolean): Si es verdadero espero que el parámetro after pipeline suba. Falso
        es lo contrario.
        mean(int): Para modificar si ver la media (0) o el desvío (1).
    """
    raw = {     'HOURS': 24.30,
                'STOI': [0.99, 0.01],
                'PESQ': [2.82, 0.72],
                'SI-SDR': [19.08, 8.93],
                'T30': [0.98, 0.57],
                'C50': [15.86, 5.48],
                'D50': [90.77, 10.79],
                'SNR': [19.08, 8.93],
                'F0': [200.01, 103.57],
                }
    
    rta = {}
    
    for dic in data:
        mos_type = dic['MOS Type']
        thshold = dic['MOS Threshold']
        den = dic["Denoising"]
        config = f"{mos_type} = {thshold} | Den = {den}"

        if parameter == "HOURS":
            diff_x = (1 - (dic[parameter]/raw[parameter]))
        elif parameter == "MCD":
            diff_x = dic[parameter][mean]/10
        else:
            if up_parameter:
                if complement:
                    diff_x = (1 - (raw[parameter][mean]/dic[parameter][mean]))
                else:
                    diff_x = raw[parameter][mean]/dic[parameter][mean]
            else:
                if complement:
                    diff_x = np.abs(1 - (dic[parameter][mean]/raw[parameter][mean]))
                else:
                    diff_x = np.abs(dic[parameter][mean]/raw[parameter][mean])
            
        rta[config] = diff_x
    
    return rta

param = "PESQ"
data = DATA_ALL 
up_param = True  # Espero que el parametro suba
complement = True  # Veo el complemento de la probabilidad
dic_rta = calc_porcentual_diff(data, param, up_param, complement, mean=0)
items_sorted = sorted(dic_rta.items(), key=lambda kv: kv[1])
max_len = max(len(k) for k, _ in items_sorted) 
val_width = 4

calc_std = []
print(" ")
print(f"El porcentaje de diferencie de {param} es:")
for k, v in items_sorted:
    print(f"{k:<{max_len}}: {v:>{val_width}}")
    calc_std.append(v)

print(" ")
print("La mejora tiene una variación (std) de: ", np.std(np.array(calc_std)))
print("La diferencia máxima es de: ", max(calc_std) - min(calc_std))
print(" ")
