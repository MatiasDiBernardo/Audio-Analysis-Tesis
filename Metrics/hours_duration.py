import os
from pydub import AudioSegment

def calculate_time_hours(folder_path):
    total_time_min = 0
    for path in os.listdir(folder_path):
        audio_path = os.path.join(folder_path, path)
        audio = AudioSegment.from_file(audio_path)
        minutes = len(audio) / 1000 / 60
        total_time_min += minutes

    return total_time_min / 60
