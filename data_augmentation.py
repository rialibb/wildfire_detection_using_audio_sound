import librosa
import glob
import numpy as np
import soundfile as sf
import os
import pandas as pd
import random
from copy import copy



def delete_audio():
    """Delete audio files in audio_data/audio folder.
    """
    ###### delete audio #####
    # Path to the folder containing the .wav files
    folder_path = "audio_data/audio"

    # Use glob to find all .wav files in the folder and delete them
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

    for file_path in wav_files:
        os.remove(file_path)
        
        
        
def extract_classes(path):
    """Returns the list of fire and non-fire names from ESC50 dataset

    Parameters
    ----------
    path: str
        the path to ESC50 dataset

    Returns
    -------
    list
        the list of fire names in ESC50
    list
        the list of non-fire names in ESC50
    """
    df = pd.read_csv(os.path.join(path, 'meta/esc50.csv'))
    list_fire = df[df.target==12]['filename'].tolist()
    list_no_fire = df[df.target!=12]['filename'].tolist()
    return list_fire, list_no_fire



def add_audio(path ,list_inp, i_start, class_type):
    """add audio files of fire and non-fire from ESC50 in audio_data/audio folder

    Parameters
    ----------
    path: str
        the path to ESC50 dataset
    list_inp: list
        the input list, can contain either fire or no-fire names
    i_start: int
        the starting index of insertion
    class_type: int
        the label of the input list list_inp
    """
    i = i_start
    for name_audio in list_inp :
        audio, sr = librosa.load(os.path.join(path,'audio', name_audio), sr=None)

        out_path = os.path.join("audio_data/audio", str(i)+'.wav')
        sf.write(out_path, audio, sr)
        i+=1
    
    df = pd.read_csv("audio_data/meta/esc2.csv")
    df1 = pd.DataFrame({'filename': [str(j)+('.wav') for j in range(i_start,len(list_inp)+i_start)],
                        'target' : [class_type for j in range(len(list_inp))] })
    df = pd.concat([df, df1])
    df.to_csv("audio_data/meta/esc2.csv", index=False)



def add_background(vall, start_val, path, main_path, bg_path, bg_volume=0.1):
    """Generate and add the augmented file into audio_data/audio

    Parameters
    ----------
    vall: int
        index of the added file
    start_val: int
        starting index of the added file
    path: str
        the path to ESC50 dataset
    main_path: str
        name of the main audio file 
    bg_path: str
        name of the background file
    bg_volume: float
        the decrease rate of the background 
    """
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

    out_path = os.path.join("audio_data/audio", str(vall+start_val)+'.wav')

    # Save the mixed audio
    sf.write(out_path, mixed_audio, sr_main)


    df = pd.read_csv("audio_data/meta/esc2.csv")
    df1 = pd.DataFrame({'filename': [str(vall+start_val)+'.wav'],
                        'target' : [1] })
    df = pd.concat([df, df1])
    df.to_csv("audio_data/meta/esc2.csv", index=False)



def generate_data(path):
    """Generate the training and val/test data for binary classification

    Parameters
    ----------
    path: str
        the path to ESC50 dataset
    """
    # initialize the pandas dataframe
    df= pd.DataFrame(columns =['filename','target'])
    df.to_csv('audio_data/meta/esc2.csv', index=False)
    #initialize the audio folder by deleting existing data
    delete_audio()
    #extract fire and no fire data from ESC50
    list_fire, list_no_fire = extract_classes(path)
    # shuffle both classes
    random.shuffle(list_fire)
    random.shuffle(list_no_fire)
    # split fire data into train and val/test data
    list_fire_train , list_fire_val_test = list_fire[: int(0.7 *len(list_fire))], list_fire[int(0.7 *len(list_fire)):]
    # split non fire data into train and val/test data :
    list_non_fire_train , list_non_fire_val_test = list_no_fire[:int(0.7 *len(list_no_fire))], list_no_fire[int(0.7 *len(list_no_fire)):]
    
    
    ## training data ##
    
    # add ESC50 training fire data with label 1
    add_audio(path ,list_fire_train, 0, 1)
    # add ESC50 training non fire data with label 0
    add_audio(path ,list_non_fire_train, len(list_fire_train), 0)
    # training fire audio augmentation
    ls_main = copy(list_fire_train)
    ls_bg = copy(list_non_fire_train)
    track= []
    
    for vall in range(len(list_non_fire_train) - len(list_fire_train)):
        while True :
            main_path = random.choice(ls_main)
            bg_path = random.choice(ls_bg)
            
            if (main_path, bg_path) not in track : 
                track.append((main_path, bg_path))
                break
            
        start_val =  len(list_fire_train+list_non_fire_train)   
        add_background(vall, start_val, 'data/esc50', main_path, bg_path)
        
        
    ## val/test data ##
    
    # add ESC50 val/test fire data with label 1
    add_audio(path ,list_fire_val_test, 2*len(list_non_fire_train) , 1)
    # add ESC50 val/test non fire data with label 0
    add_audio(path ,list_non_fire_val_test, 2*len(list_non_fire_train) +len(list_fire_val_test), 0)
    # val/test fire audio augmentation
    ls_main = copy(list_fire_val_test)
    ls_bg = copy(list_non_fire_val_test)
    track= []
    
    for vall in range(len(list_non_fire_val_test) - len(list_fire_val_test)):
        while True :
            main_path = random.choice(ls_main)
            bg_path = random.choice(ls_bg)
            
            if (main_path, bg_path) not in track : 
                track.append((main_path, bg_path))
                break
            
        start_val =  2*len(list_non_fire_train) + len(list_fire_val_test+list_non_fire_val_test)   
        add_background(vall, start_val, 'data/esc50', main_path, bg_path)
        

generate_data('data/esc50')
