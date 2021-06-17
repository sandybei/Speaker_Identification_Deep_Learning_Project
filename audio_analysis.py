import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from pyAudioAnalysis.audioBasicIO import stereo_to_mono, read_audio_file
import matplotlib.pyplot as plt
from preprocess_files import dev_files, all_files
import librosa.display


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
    

def get_spectrogram(file):
    """
    takes an audio file and returns a numpy array of the spectrogram 
    """
    # load audio file
    sample_rate, samples = read_audio_file(file)
    # convert audio to mono (if stereo) 
    mono_signal = stereo_to_mono(samples)
    mono_signal = mono_signal.astype(np.float32)
    # get spectrogram as numpy array
    spectrogram = librosa.feature.melspectrogram(mono_signal, sample_rate, n_fft=2048, hop_length=512, n_mels=128) # larger hop length 2048
    # get plot figure of spectrogram
    # figure = get_image(spectrogram)
    # add padding to images
    return spectrogram


def optimal_image_width(plotShow):
    img_widths = []
    for file in all_files:                   
        spec = get_spectrogram(file) 
        img_widths.append(spec.shape[1])
    img_widths = np.asarray(img_widths)
    best_width = np.percentile(img_widths, 95)
    best_width = int(best_width)
    if plotShow:
        plt.hist(img_widths)
        plt.show()
    return best_width


def pad_spectrogram(spec, best_width):
    img_length = spec.shape[0] 
    img_width = spec.shape[1]
    max_width = best_width
    # trim image if length larger than best width
    if img_width > max_width:
        spec = spec[:, :max_width]  
    # add padding at the end of the image if width less than best width
    elif img_width < max_width:
        pad_width = max_width - img_width
        padding = np.zeros((img_length, pad_width))
        spec = np.column_stack((spec, padding))
    return spec


def get_image(spectrogram):
    fig = plt.figure()
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))
    return fig

def preprocess(file):
    spec = get_spectrogram(file)
    processed_spec = pad_spectrogram(spec, best_width)
    fig = get_image(processed_spec)
    return fig

# best width for spectrogram images
best_width = optimal_image_width(False)
#spec = get_spectrogram(dev_files['id11229'][2])
#spec = pad_spectrogram(spec, best_width)