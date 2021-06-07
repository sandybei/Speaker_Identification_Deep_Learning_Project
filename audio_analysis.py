from scipy.io import wavfile 
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np
import librosa
from scipy.signal.spectral import spectrogram
from get_audio import audio_signals_dict, sampling_rates_dict
from pyAudioAnalysis.audioBasicIO import stereo_to_mono, read_audio_file


def get_maximum_duration():
    """
    return maximum duration in seconds of all audio files in the training set
    to get the duration in seconds, we divide the number of samples by sampling rate
    """
    durations = []
    for key in audio_signals_dict.keys():
        for audio_signal in audio_signals_dict[key]:
            duration = audio_signal.shape[0] / float(sampling_rates_dict[key])
            durations.append(duration)
    durations = np.asarray(durations)
    max_duration = durations.max()
    return max_duration


def show_spectogram(spectogram):
    """
    show spectogram of an audio file
    """
    plt.imshow(spectogram)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()


def audio_resize(sample_rate, audio_signal, max_duration):
    """
    resizes audio signal to have the maximum signal length
    """
    signal_length = audio_signal.shape[0] 
    print('audio duration :', signal_length / float(sample_rate))
    max_length = sample_rate * max_duration
    if signal_length > max_length:
        audio_signal = audio_signal[:max_length]  
    elif signal_length < max_length:
        pad_length = max_length - signal_length
        padding = np.zeros(int(pad_length))
        audio_signal = np.append(audio_signal, padding)
        signal_length = audio_signal.shape[0] / sample_rate
        print('padded audio duration: ', audio_signal.shape[0] / float(sample_rate))        
    return audio_signal



if __name__ == "__main__":
        
    # all have same sampling rate / same #channels (mono)
    sampling_rates = []
    for key in sampling_rates_dict.keys():
        sampling_rates.append(sampling_rates_dict[key])
    print(sampling_rates)
    print('All audio files have the same sampling')
    
    # get maximum length of audio signal
    max_length = get_maximum_duration()
    print('The max length of audio files is: ', max_length)

    # get an audio file
    audio = audio_signals_dict['id10650'][4]
    print('audio length :', audio.shape[0])
    sample_rate = sampling_rates_dict['id10650']
    print('sample_rate :', sample_rate)
    
    # convert audio to mono (if stereo) 
    mono_signal = stereo_to_mono(audio)

    # resize audio
    audio = audio_resize(sample_rate, mono_signal, max_length)

    # get spectogram of audio in mel-scale
    spectogram = librosa.feature.melspectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128)






