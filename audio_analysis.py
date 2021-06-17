import matplotlib.pyplot as plt
import numpy as np
import librosa
from pyAudioAnalysis.audioBasicIO import stereo_to_mono, read_audio_file
import matplotlib.pyplot as plt
from preprocess_files import all_files
import librosa.display
import matplotlib

matplotlib.use('TkAgg')


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
    spectrogram = librosa.feature.melspectrogram(mono_signal, sample_rate, n_fft=2048, hop_length=2048, n_mels=128) # larger hop length 2048
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
    print('\n        Spectrogram Images Statistics      ')
    print('-------------------------------------------')
    print('Mininum width:       ', np.min(img_widths))
    print('Maximum width :      ', np.max(img_widths))
    print('95-Percentile width: ', best_width)
    return best_width


def get_sample_rates():
    sample_rates = []
    for file in all_files:                   
        sample_rate, _ = read_audio_file(file) 
        sample_rates.append(sample_rate)
    sample_rates = np.asarray(sample_rates)
    sample_rates = np.unique(sample_rates)
    if len(sample_rates) == 1:
        print(f'\nAll audios have the same sample rate: {sample_rates[0]} kHz')
    else:
        print("Audios don't have the same sample rates")
    return sample_rates


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


def preprocess(file):
    spec = get_spectrogram(file)
    processed_spec = pad_spectrogram(spec, best_width)
    fig = plt.figure()
    librosa.display.specshow(librosa.power_to_db(processed_spec, ref=np.max))
    return fig


# check if all audio files have the same sample rates
sample_rates = get_sample_rates()
# get best width for spectrogram images
best_width = optimal_image_width(False)

