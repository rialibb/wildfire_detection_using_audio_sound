import librosa
import glob
import numpy as np
import soundfile as sf
import os
import pandas as pd
import random


def extract_classes(path):
    df = pd.read_csv(os.path.join(path, 'meta/esc50.csv'))
    list_fire = df[df.target==12]['filename'].tolist()
    list_no_fire = df[df.target!=12]['filename'].tolist()
    return list_fire, list_no_fire



def add_fire_audio(path ,list_fire):

    for name_audio in list_fire :
        audio, sr = librosa.load(os.path.join(path,'audio', name_audio), sr=None)

        out_path = os.path.join("audio_data/augmented_fire_audio", name_audio)
        sf.write(out_path, audio, sr)




def add_non_fire_audio(path, list_no_fire):
    for name_audio in list_no_fire :
        audio, sr = librosa.load(os.path.join(path,'audio', name_audio), sr=None)

        out_path = os.path.join("audio_data/non_fire_audio", name_audio)
        sf.write(out_path, audio, sr)



def delete_audio():

    ###### delete fire audio #####
    # Path to the folder containing the .wav files
    folder_path = "audio_data/augmented_fire_audio"

    # Use glob to find all .wav files in the folder and delete them
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

    for file_path in wav_files:
        os.remove(file_path)

    #### delete non fire audio ####  
    # Path to the folder containing the .wav files
    folder_path = "audio_data/non_fire_audio"

    # Use glob to find all .wav files in the folder and delete them
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

    for file_path in wav_files:
        os.remove(file_path)


    




def add_background(path, main_path, bg_path, bg_volume=0.1):
    
    # Load audio files
    main_audio, sr_main = librosa.load(os.path.join(path,'audio', main_path), sr=None)
    bg_audio, sr_bg = librosa.load(os.path.join(path,'audio', bg_path), sr=None)

    # Ensure sample rates match
    if sr_main != sr_bg:
        raise ValueError("Sample rates of main and background audio must match")

    # Ensure both audio signals are the same length
    if len(main_audio) > len(bg_audio):
        # Repeat or truncate the background audio to match the length of the main audio
        bg_audio = np.tile(bg_audio, int(np.ceil(len(main_audio) / len(bg_audio))))[:len(main_audio)]
    elif len(bg_audio) > len(main_audio):
        # Truncate background audio if it's longer
        bg_audio = bg_audio[:len(main_audio)]
    
    # Scale the background audio
    bg_audio = bg_audio * bg_volume
    
    # Combine the main audio and background audio
    mixed_audio = main_audio + bg_audio
    
    # Normalize the mixed audio to prevent clipping
    max_amplitude = np.max(np.abs(mixed_audio))
    if max_amplitude > 1.0:
        mixed_audio = mixed_audio / max_amplitude

    out_path = os.path.join("audio_data/augmented_fire_audio", main_path[:-4]+bg_path[:-4]+'.wav')

    # Save the mixed audio
    sf.write(out_path, mixed_audio, sr_main)





def generate_fire_data(path):

    delete_audio()
    
    list_fire, list_no_fire = extract_classes(path)

    add_fire_audio(path, list_fire)

    add_non_fire_audio(path, list_no_fire)


    # fire audio augmentation
    for i in range(40*48):
        while True :
            main_path = random.choice(list_fire)
            bg_path = random.choice(list_no_fire)

            # check that the couple (main_path, bg_path) is not taken before
            file_path = os.path.join("audio_data/augmented_fire_audio", main_path[:-4]+bg_path[:-4]+'.wav')
            if not os.path.isfile(file_path) :
                break
        
        add_background(path, main_path, bg_path)

generate_fire_data('data/esc50')
