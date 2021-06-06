
from scipy.io import wavfile 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np
import librosa
from create_datasets import audio_signals_dict, sampling_rates_dict
from pyAudioAnalysis.audioBasicIO import stereo_to_mono, read_audio_file



def get_spectogram(sample_rate, audio_signal):
    """
    get spectogram of an audio file
    """
    frequencies, times, spectrogram = audio_signal.spectrogram(audio_signal, sample_rate)
    return spectrogram


def show_spectogram(spectogram):
    """
    show spectogram of an audio file
    """
    plt.imshow(spectogram)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()


def get_maximum_duration():
    """
    return maximum duration in seconds of all audio files in training set
    to get the duration in seconds, we divide the number of samples by sampling rate
    """
    durations = []
    for key in audio_signals_dict.keys():
        for audio_signal in audio_signals_dict[key]:
            duration = audio_signal.shape[0] / float(sampling_rates_dict[key])
            durations.append(duration)
    durations = np.asarray(durations)
    max_duration = durations.max()
    min_duration = durations.min()
    return max_duration, min_duration


def audio_trim(sample_rate, audio_signal):
    signal_length = audio_signal.shape[0] / sample_rate
    if signal_length > 10:
        audio_signal = audio_signal[:10]        
    return audio_signal, sample_rate



if __name__ == "__main__":
    
    # all have same length
    '''
    print(sampling_rates_dict)
    for key in sampling_rates_dict.keys():
        print(sampling_rates_dict[key])
    '''
    
    # get maximum length of audio signal
    max_length, min_length = get_maximum_duration()
    print(max_length, min_length)
    print(round((max_length + min_length)/ 2, 2))

    audio = audio_signals_dict['id10650'][1]
    print(audio)
    sr = sampling_rates_dict['id10650']
    print(audio_trim(sr, audio))

    # convert audio to mono (if stereo) (all were mono)
    mono_signal = stereo_to_mono(audio)
    
    






