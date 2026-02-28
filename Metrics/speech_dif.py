from pymcd.mcd import Calculate_MCD
import numpy as np
import pesto
import torchaudio
import tqdm
import os

def calculate_MCD(audio_based, audio_denoised):
    """Calculate MCD for paired audios.

    Args:
        audio_based (str): Path to reference
        audio_denoised (str): Path to altered version

    Returns:
        np.float: Value of MCD
    """
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd_val = mcd_toolbox.calculate_mcd(audio_based, audio_denoised)
    return mcd_val

def calculate_F0(path_audio, threshold_confidence):
    """Calculate F0 mean and F0 std for single audio file.

    Args:
        path_audio (str): Path audio to evaluate
        threshold_confidence (float): Value (from 0 to 1) of confidence 
        where the F0 value is considered to calculate the average and std.


    Returns:
        float, float: F0 mean, F0 std
    """

    x, sr = torchaudio.load(path_audio)
    if x.dim() > 1:
        x = x.mean(dim=0)  # Mono conversion

    timesteps, pitch, confidence, activations = pesto.predict(x, sr)
    pitch_vals = pitch.numpy()
    confidence_vals = confidence.numpy()
    
    pitch_vals_filtered = pitch_vals[confidence_vals > threshold_confidence]
    
    F0_mean = np.mean(pitch_vals_filtered)
    F0_std = np.std(pitch_vals_filtered)

    return F0_mean, F0_std

def calculate_speech_dif(folder_path):
    """Calculate all metrics for audio quality evaluation.
    Computes PESQ, SI-SDR and SNR mean for batch of audios.

    Args:
        folder_path (str): Path to folder with audios to analyze

    Returns:
        float, float, float: pesq_mean, sisder_mean, snr_mean 
    """
    MCD = []
    F0_STD = []
    f0_threshold = 0.9

    for folders in tqdm(os.listdir(folder_path)):
        for audio_path in os.listdir(os.path.join(folder_path, folders)):
            file_name = os.path.join(folder_path, folders, audio_path)
            _, f0_std_val = calculate_F0(file_name, f0_threshold)

            F0_STD.append(f0_std_val)

    f0_std_mean = np.mean(F0_STD)

    return f0_std_mean
