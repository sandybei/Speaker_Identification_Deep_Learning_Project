from librosa.util import files
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from pyAudioAnalysis.audioBasicIO import stereo_to_mono
import matplotlib.pyplot as plt
from preprocess_files import all_files


def get_speaker_audio(dict):
    """
    :param id: the id of the speaker
    :return: a dictionary of sampling frequencies and signals of audio files
    """  
    signals = []
    for id in dict.keys():        
        for file in dict[id]:            
            samples, sampling_rate = librosa.load(file, sr=44000) 
            signals.append(samples)
    return signals


def min_max_duration(signals, sample_rate):
    """
    return maximum duration in seconds of all audio files in the training set
    to get the duration in seconds, we divide the number of samples by sampling rate
    """
    durations = []
    for sig in signals:
        duration = sig.shape[0] / float(sample_rate)
        durations.append(duration)
    durations = np.asarray(durations)
    return durations


def audio_resize(sample_rate, audio_signal, max_duration):
    """
    resizes audio signal to have the maximum signal length
    """
    signal_length = audio_signal.shape[0] 
    #print('audio length :', signal_length / float(sample_rate))
    max_length = sample_rate * max_duration
    # trim audio if length  more than 10 secs
    if signal_length > max_length:
        audio_signal = audio_signal[:max_length]  
    # add padding at the end of the audio if length less than 10 sec
    elif signal_length < max_length:
        pad_length = max_length - signal_length
        padding = np.zeros(int(pad_length))
        audio_signal = np.append(audio_signal, padding)
        #padded_length = audio_signal.shape[0] / sample_rate
        #print('padded audio length: ', padded_length.shape[0] / float(sample_rate))      
    return audio_signal


def spectrogram_image(spectrogram):
    fig = plt.figure()
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))
    return fig
    

def preprocess(file):
    """
    takes an audio file and returns a plot of the audio mel spectrogram 
    """
    # load audio file
    audio, sample_rate = librosa.load(file, sr=44000)

    # convert audio to mono (if stereo) 
    mono_signal = stereo_to_mono(audio)

    # resize audio 
    audio = audio_resize(sample_rate, mono_signal, 10)

    # get spectrogram as numpy array
    spectrogram = librosa.feature.melspectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128)

    # get plot figure of spectrogram
    figure = spectrogram_image(spectrogram)

    return figure


file = all_files[0]
image = preprocess(file) 
plt.savefig('image_2.png', bbox_inches='tight', pad_inches=0, transparent=True)

'''
#create training set dictionary
audio_signals = get_speaker_audio(files_dict)
# get signal lengths in sec
durations = min_max_duration(audio_signals, 44000)
print(durations)
'''
    