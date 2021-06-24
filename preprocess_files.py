import os
from numpy.lib.function_base import rot90
import pandas as pd
import matplotlib.pyplot as plt


def map_files_to_labels(dir):
    dir_split = dir.split(os.sep)
    for word in dir_split:
        if 'id' in word:
            label = word
    files = []
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        for file in os.listdir(folder_path):
            audio_file = os.path.join(folder_path, file)
            files.append(audio_file)
    return label, files


def create_file_dictionary(dir):
    files_dict = {}
    labels = []
    audio_files_list = []
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        label, audio_files = map_files_to_labels(folder_path)
        labels.append(label)
        audio_files_list.append(audio_files)
        files_dict[label] = audio_files
    return files_dict


# get voxceleb dev files
dev_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'wav'
dev_files = create_file_dictionary(dev_dir)

# get number of audio files per speaker id
files = pd.Series(dtype=float)
new_dict = {}
ids = dev_files.keys()
for id in ids:
    files[id] = len(dev_files[id])
# select 10 speakers for classification
files = files.sort_values(ascending=False)
files.iloc[:20].plot.barh(title='Number of files per speaker\n(for top 20 speakers with most audio files)')
plt.savefig(os.path.join('results', 'files_per_speaker.png'))
plt.close()
files_to_keep = files.iloc[4:14] 
ids = files_to_keep.index.to_list()

# get number of files for training, validation and test set for a 80/10/10 split
total_files = files_to_keep.sum()
test_files_num = int(total_files * 10 / 100)
val_files_num = int(total_files * 10 / 100)
train_files_num = int(total_files - (test_files_num + val_files_num))
files_per_id = round(test_files_num / files.shape[0])
# get file info plot
files_info = pd.Series({'Total': total_files, 'Training': train_files_num, 'Validation': val_files_num, 'Test': test_files_num})
files_info.plot.barh(title='Number of audio files')
plt.tight_layout()
plt.savefig(os.path.join('results', 'files_info.png'))
plt.close()

# get classification labels
metadata = pd.read_csv(os.path.join('data', 'vox1_meta.csv'), sep='\t')
labels = metadata.loc[metadata['VoxCeleb1 ID'].isin(ids)]
labels.reset_index(drop=True, inplace=True)
labels = labels.groupby(['Gender']).size()
fig = labels.plot.bar()
# get gender distribution plot
plt.title('Gender Distribution')
plt.savefig(os.path.join('results', 'genders.png'))
plt.close()

# get all audio files to be used 
files_dict = {id: dev_files[id] for id in ids}

# create training, validation and test set files from voxceleb data
train_files = {}
val_files = {}
test_files = {}
for id in ids:
    train_files[id] = dev_files[id][files_per_id*2:]
    val_files[id] = dev_files[id][:files_per_id-1]
    test_files[id] = dev_files[id][files_per_id:files_per_id*2]


