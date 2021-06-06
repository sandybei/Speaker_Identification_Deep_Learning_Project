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
            sampling_rate, samples = wavfile.read(file) 
            sampling_rates.append(sampling_rate)
            signals.append(samples)
        audio_signals_dict[id] = signals
        sampling_rates_dict[id] = sampling_rate
    return audio_signals_dict


# select 10 speakers for classification
files_num = pd.Series(dtype=float)
new_dict = {}
ids = train_files.keys()
for id in ids:
    files_num[id] = len(train_files[id])
files_num = files_num.sort_values()
files_num = files_num.iloc[1:10] # [8:18]
ids = files_num.index

# create new dictionaries
new_train_files = {id: train_files[id] for id in ids}
print(new_train_files.keys())
#new_test_files = {id: test_files[id] for id in ids}

# create training set dictionary
training_set = get_speaker_audio(new_train_files)

# get training set as a list of numpy arrays
X = []
for id in training_set.keys():
    for signals in training_set[id]:
        for audio_signal in signals:
            X.append(audio_signal)
print(type(X))
print(type(X[0]))

