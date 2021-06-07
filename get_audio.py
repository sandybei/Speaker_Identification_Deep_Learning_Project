import librosa
from preprocess_files import train_files
import pandas as pd
from preprocess_files import new_train_files


def get_speaker_audio(dict):
    """
    :param id: the id of the speaker
    :return: a dictionary of sampling frequencies and signals of audio files
    """  
    audio_signals_dict = {}
    sampling_rates_dict = {}      
    for id in dict.keys():
        signals = []
        sampling_rates = []
        for file in dict[id]:            
            samples, sampling_rate = librosa.load(file, sr=44000) 
            sampling_rates.append(sampling_rate)
            signals.append(samples)
        audio_signals_dict[id] = signals
        sampling_rates_dict[id] = sampling_rate
    return sampling_rates_dict, audio_signals_dict


# create training set dictionary
sampling_rates_dict, audio_signals_dict = get_speaker_audio(new_train_files)

# get training set as a list of numpy arrays
#print(audio_signals_dict.values())

