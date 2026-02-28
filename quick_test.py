import os

folder_variants = "Small_variants"
output_folder = "TTS_samples"

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

config_folders = os.listdir(folder_variants)
count_sentence = 0

for config in config_folders:
    os.makedirs(os.path.join(output_folder, config), exist_ok=True)
    audio_files = os.listdir(os.path.join(folder_variants, config))
    for audio in audio_files:
        # Define parameters for Qwen inference
        infer_sentences = sentences[count_sentence % len(sentences)]
        count_sentence += 1
        ref_audio_path = os.path.join(folder_variants, config, audio)
        
        print("Transcribe y genera el audio")

        # Step C: Save output with a clear naming convention
        name_wav = audio.split(".")[0] + ".wav"
        output_path = os.path.join(output_folder, config, name_wav)
        