from Metrics.hours_duration import calculate_time_hours
from Metrics.audio_quality import calculate_audio_quality
from Metrics.acoustic_parameters import calculate_acoustic_parameters
from Metrics.speech_dif import calculate_speech_dif, calculate_MCD

def calculate_composite_metric(folder_path_original, folder_path_processed):    
    """Calculate composite metric for audio quality evaluation.
    Combines HOURS, PESQ, SI-SDR, SNR, T30, C50, F0 std and MCD.

    Args:
        folder_path_original (str): Path to folder with the original dataset
        folder_path_processed (str): Path to folder with the processed dataset
    
    Returns:
        (float): Composite metric value
    """
    hours_r = calculate_time_hours(folder_path_original)
    hours_p = calculate_time_hours(folder_path_processed)

    pesq_r, sisdr_r, snr_r = calculate_audio_quality(folder_path_original)
    pesq_p, sisdr_p, snr_p = calculate_audio_quality(folder_path_processed)
    
    T30_r, C50_r = calculate_acoustic_parameters(folder_path_original)
    T30_p, C50_p = calculate_acoustic_parameters(folder_path_processed)

    f0_std_r = calculate_speech_dif(folder_path_original)
    f0_std_p = calculate_speech_dif(folder_path_processed)
    mcd_p = calculate_MCD(folder_path_original, folder_path_processed)

    # Individual metrics 
    DR = (1 - hours_p/hours_r)
    SQ = (pesq_r/pesq_p) + (sisdr_r/sisdr_p) + (snr_r/snr_p)
    AP = (T30_p/T30_r) + (C50_r/C50_p)
    SD = abs(1 - (f0_std_p/f0_std_r)) + (mcd_p/5)

    # Composite metric calculation (weights can be adjusted)
    composite_metric = DR + SQ + AP + SD
    
    return composite_metric
