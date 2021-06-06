
from scipy.io import wavfile 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np
import librosa
from create_datasets import training_set


def get_spectogram(sample_rate, audio_signal):
    """
    get spectogram of an audio file
    """
    frequencies, times, spectrogram = signal.spectrogram(audio_signal, sample_rate)
    return spectrogram


def show_spectogram(spectogram):
    """
    show spectogram of an audio file
    """
    plt.imshow(spectogram)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()

if __name__ == "__main__":
    # get training set as a list of numpy arrays
    X = []
    for id in training_set.keys():
        for signals in training_set[id]:
            for audio_signal in signals:
                X.append(audio_signal)
    





