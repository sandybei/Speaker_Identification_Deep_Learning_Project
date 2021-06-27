import matplotlib.pyplot as plt
import numpy as np
import librosa
from numpy.lib.arraypad import pad
from pyAudioAnalysis.audioBasicIO import stereo_to_mono, read_audio_file
from process_files import files_dict
import librosa.display
import os


def get_sample_rates():
    """
    checks if the sample rate of all audio files have the same or different sample rate

    :return: unique sample rates of all audio files
    """
    sample_rates = []
    for id in files_dict.keys(): 
        for file in files_dict[id]:  
            sample_rate, _ = read_audio_file(file) 
            sample_rates.append(sample_rate)
    sample_rates = np.asarray(sample_rates)
    sample_rates = np.unique(sample_rates)
    return sample_rate


def audio_to_spectrogram(file):
    """
    takes an audio file and returns its spectrogram 

    :param file: filepath of an audio file
    :return: spectrogram of the audio wave as a numpy array
    """
    # load audio file
    sample_rate, samples = read_audio_file(file)
    # convert audio to mono (if stereo) 
    mono_signal = stereo_to_mono(samples)
    mono_signal = mono_signal.astype(np.float32)
    # get spectrogram as numpy array
    spectrogram = librosa.feature.melspectrogram(mono_signal, sample_rate, n_fft=2048, hop_length=2048, n_mels=128) 
    return spectrogram


def optimal_image_width():
    """
    finds the optimal width of spectrogram images in pixels

    :return: the 95th percentile of all spectrogram images widths
    """
    # get best width
    img_widths = []
    for id in files_dict.keys():    
        for file in files_dict[id]:
            spec = audio_to_spectrogram(file) 
            img_widths.append(spec.shape[1])
    img_widths = np.asarray(img_widths)
    best_width = np.percentile(img_widths, 95)
    best_width = int(best_width)
    # plot histogram 
    plt.hist(img_widths)
    plt.title('Image Widths')
    plt.xlabel('Width in Pixels')
    plt.ylabel('Frequency')
    plt.axvline(x=best_width, color='r', linestyle='--', label='95th-percentile')
    plt.legend()
    plt.savefig(os.path.join('results', 'image_widths.png'))
    plt.close()
    return best_width
    

def pad_or_crop_spectrogram(spec, best_width):
    """
    - crops spectrogram image to have width equal to the 95th percentile of all widths if it has less
    - pads spectrogram image with zeros to have width equal to the 95th percentile of all widths if it has more

    :param spec: spectrogram as a numpy array
    :return: padded or cropped spectrogram
    """
    img_width = spec.shape[1]
    max_width = best_width
    if img_width > max_width:
        spec = spec[:, :max_width]  
    elif img_width < max_width:
        pad_width = max_width - img_width
        left_pad_width = int(pad_width / 2)
        right_pad_width = np.abs(pad_width - left_pad_width)
        spec = np.pad(spec, ((0,0), (left_pad_width,right_pad_width)))
    return spec


def preprocess(file):
    """
    gets audio file and returns its mel spectrogram

    :param file: filepath of an audio file
    :return: plot figure of the mel spectrogram of an audio wave
    """
    spec = audio_to_spectrogram(file)
    processed_spec = pad_or_crop_spectrogram(spec, best_width)
    fig = plt.figure()
    librosa.display.specshow(librosa.power_to_db(processed_spec, ref=np.max))
    return fig


# check if all audio files have the same sample rates
sample_rates = get_sample_rates()

# get best width for spectrogram images
best_width = optimal_image_width()
print('The optimal image width of spectrograms is: ', best_width)


