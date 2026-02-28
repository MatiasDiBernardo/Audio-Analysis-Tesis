# Itera sobre small variants y genere un muestra de TTS zero shot con los audios de cada carpeta, y las pone en una
# Carpeta llamada TTS_samples (manteniendo el nombre delas subcarpeta con la configuración al igual que Small_variants)
# Colab con Qwen TTS
# Hay cosas a corregir pero lo dejo como ejemplo

import os
import torch
import soundfile as sf
import whisper
from qwen_tts import Qwen3TTSModel

# 1. Setup Devices & Models
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading Whisper model...")
# 'small' is a good baseline, but you might want 'medium' for higher transcription accuracy
whisper_model = whisper.load_model("small").to(device)

print("Loading Qwen3-TTS model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 2. Configuration
reference_folder = "ref_audios"
output_folder = "output_clones"
os.makedirs(output_folder, exist_ok=True)

# The list of sentences to generate for each voice

# 3. Pipeline Execution
valid_extensions = ('.wav', '.mp3', '.flac')
# Safely grab only audio files from the directory
audio_files = [f for f in os.listdir(reference_folder) if f.lower().endswith(valid_extensions)]

for audio_file in audio_files:
    ref_audio_path = os.path.join(reference_folder, audio_file)
    print(f"\n--- Processing: {audio_file} ---")
    
    # Step A: Transcribe the reference audio with Whisper
    # Note: If testing an Argentine Spanish corpus, you can enforce the language using:
    # transcription = whisper_model.transcribe(ref_audio_path, language="es")
    transcription = whisper_model.transcribe(ref_audio_path)
    ref_text = transcription["text"].strip()
    print(f"Transcribed Text: {ref_text}")
    
    # Step B: Iterate over the target sentences
    for idx, sentence in enumerate(sentences):
        print(f" -> Generating sentence {idx+1}...")
        
        # Generate cloned audio
        wavs, sr = model.generate_voice_clone(
            text=sentence,
            # Adjust to "Spanish" if generating non-English text
            language="Spanish", 
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
        
        # Step C: Save output with a clear naming convention
        base_name = os.path.splitext(audio_file)[0]
        output_filename = f"{base_name}_target_{idx+1}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        sf.write(output_path, wavs[0], sr)
        print(f"    Saved: {output_path}")

print("\nBatch generation complete.")


sentences = [
    "¿Sabías que el cuerpo humano posee sensores? ¡Sí! ¡Como los robots!",
    "Esos sensores se encargan de capturar la energía que nos rodea.",
    "La transforman y la transportan como sensaciones que llegan al cerebro.",
    "Los sensores más conocidos detectan formas y colores.",
    "También detectan sonidos, olores y gustos.",
    "Además producen muchas sensaciones en la piel, como frío y calor.",
    "¿Hay otros sensores? ¡Claro que sí!",
    "Muchos más son menos conocidos, pero también importantes.",
    "Algunos sensores nos mantienen en equilibrio. Otros actúan cuando comemos algo picante.",
    "Cuando las sensaciones llegan al cerebro, se encuentran con conocimientos guardados en la memoria.",
    "Esos conocimientos se acumulan durante años .Entonces se produce la percepción.",
    "La percepción nos dice qué significa la sensación. También nos dice si hay un mensaje.",
    "Desde pequeños aprendemos los sonidos de nuestra lengua.",
    "Si oímos una palabra que nunca escuchamos, tratamos de acercarla a una conocida.",
    "Si eso falla, podemos decir: ¡no la entendí!"
]

folder_variants = "Small_variants"
output_folder = "TTS_samples"
config_folders = os.listdir(folder_variants)
count_sentence = 0

for config in config_folders:
    os.makedirs(os.path.join(output_folder, config), exist_ok=True)
    audio_files = os.listdir(os.path.join(folder_variants, config))
    for audio in audio_files:
        # Define parameters for Qwen inference
        infer_sentence = sentences[count_sentence % len(sentences)]
        count_sentence += 1
        ref_audio_path = os.path.join(folder_variants, config, audio)
        transcription = whisper_model.transcribe(ref_audio_path)
        ref_text = transcription["text"].strip()

        # Inference
        wavs, sr = model.generate_voice_clone(
            text=infer_sentence,
            language="Spanish", 
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
        
        # Save output with the same naming convention
        name_wav = audio.split(".")[0] + ".wav"
        output_path = os.path.join(output_folder, config, name_wav)
        sf.write(output_path, wavs[0], sr)
        

    

