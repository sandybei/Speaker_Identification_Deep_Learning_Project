import os
import json


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


def dataset_to_json(dir, file_name):
    files_dictionary = create_file_dictionary(dir)
    output_file_name = "data" + os.sep + "datasets_files" + os.sep + f"{file_name}.json"
    with open(output_file_name, 'w') as output_file:
        json.dump(files_dictionary, output_file, indent=4)   
    return files_dictionary


# get training set files
train_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'wav'
train_files = dataset_to_json(train_dir, 'train_files')


# get test set files
test_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'vox1_test_wav' + os.sep + 'wav'
test_files = dataset_to_json(test_dir, 'test_files')





