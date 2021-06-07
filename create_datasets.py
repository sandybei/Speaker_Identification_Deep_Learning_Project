import librosa
from preprocess_files import train_files
import pandas as pd
from scipy.io import wavfile 
import matplotlib.pyplot as plt


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


# select 10 speakers for classification
"""
probably better select speakers by sorting them by their total audio lengths 
instead of their total number of audio files ?? 
"""
files_num = pd.Series(dtype=float)
new_dict = {}
ids = train_files.keys()
for id in ids:
    files_num[id] = len(train_files[id])
files_num = files_num.sort_values()
files_num = files_num.iloc[1:10] # [8:18]
ids = files_num.index
# dict_keys(['id10650', 'id10310', 'id10327', 'id10684', 'id10614', 'id11229', 'id11074', 'id10784', 'id10958'])


# create new dictionaries
new_train_files = {id: train_files[id] for id in ids}
#new_test_files = {id: test_files[id] for id in ids}

# create training set dictionary
sampling_rates_dict, audio_signals_dict = get_speaker_audio(new_train_files)

# get training set as a list of numpy arrays
#print(audio_signals_dict.values())

