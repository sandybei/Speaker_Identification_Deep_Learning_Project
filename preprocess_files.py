import os
import json
import pandas as pd


def map_files_to_labels(dir):
    dir_split = dir.split(os.sep)
    for word in dir_split:
        if 'id' in word:
            label = word
    audio_files = []
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        for file in os.listdir(folder_path):
            audio_file = os.path.join(folder_path, file)
            audio_files.append(audio_file)
    return label, audio_files


def create_file_dictionary(dir):
    audio_files_dict = {}
    labels = []
    audio_files_list = []
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        label, audio_files = map_files_to_labels(folder_path)
        labels.append(label)
        audio_files_list.append(audio_files)
        audio_files_dict[label] = audio_files
    return audio_files_dict



# get voxceleb training set files
train_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'wav'
train_files = create_file_dictionary(train_dir)

# select 10 speakers for classification
"""
probably should select speakers by sorting them by their total audio lengths 
instead of their total number of audio files ?? 
"""
files_num = pd.Series(dtype=float)
new_dict = {}
ids = train_files.keys()
for id in ids:
    files_num[id] = len(train_files[id])
files_num = files_num.sort_values()
files_num = files_num.iloc[1:10] # [8:18]
ids = files_num.index

# get classification lables
metadata_path = 'data' + os.sep + 'metadata' + os.sep + 'metadata.csv'
metadata = pd.read_csv(metadata_path, index_col=[0])
labels = metadata.loc[metadata['VoxCeleb1 ID'].isin(ids)]
print(labels)

# get audio files of training set
new_train_files = {id: train_files[id] for id in ids}
output_file_name = "data" + os.sep + "datasets_files" + os.sep + "training_set_files.json"
with open(output_file_name, 'w') as output_file:
    json.dump(new_train_files, output_file, indent=2) 


# have to create the training set according to the classes in the test set
'''
# get audio files of test set
test_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'vox1_test_wav' + os.sep + 'wav'
test_files = create_file_dictionary(test_dir)
print(test_files.keys())
new_test_files = {id: test_files[id] for id in ids}
'''







