from preprocess_files import all_files
from audio_analysis import preprocess
from preprocess_files import speaker_ids
import numpy as np

# get training set
X_train = [] 
for file in all_files:
    image = preprocess(file) 
    X_train.append(image)
X_train = np.asarray(X_train)

# get labels
y_train = [int(id[2:]) for id in speaker_ids]
y_train = np.asarray(y_train)
print(X_train.shape)
print(y_train.shape)
