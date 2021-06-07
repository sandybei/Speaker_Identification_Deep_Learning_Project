from preprocess_files import all_files
from audio_analysis import *
from preprocess_files import speaker_ids

def preprocess(file):
    # load audio file

    audio, sample_rate = librosa.load(file, sr=44000)

    # convert audio to mono (if stereo) 
    mono_signal = stereo_to_mono(audio)

    # resize audio 
    audio = audio_resize(sample_rate, mono_signal, 10)

    # get spectogram of audio in mel-scale
    spectogram = librosa.feature.melspectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    #librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
    #plt.show()

    return spectogram


X_train = [] 
for file in all_files:
    image = preprocess(file) 
    X_train.append(image)
X_train = np.asarray(X_train)

y_train = [int(id[2:]) for id in speaker_ids]
y_train = np.asarray(y_train)
print(X_train.shape)
print(y_train.shape)
