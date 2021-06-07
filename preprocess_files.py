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


def get_audio_files(dictionary):
    files = []
    ids = []
    for id in dictionary.keys():
        for file in dictionary[id]:
            files.append(file)
            ids.append(id)
    return ids, files


# get voxceleb training set files
train_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'wav'
train_files = create_file_dictionary(train_dir)

# select 10 speakers for classification
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
files_with_ids = {id: train_files[id] for id in ids}
speaker_ids, all_files = get_audio_files(files_with_ids)
output_file_name = "data" + os.sep + "datasets_files" + os.sep + "all_files.json"
with open(output_file_name, 'w') as output_file:
    json.dump(all_files, output_file, indent=2) 

