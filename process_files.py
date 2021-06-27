import os
import pandas as pd
import matplotlib.pyplot as plt
import json


def map_files_to_ids(dir):
    """
    This functions creates a dictionary of audio files with keys the speakers ids and values
    the audio files that correspond to them.

    :param dir: directory of audio files having the structure of voxceleb data folders
    :return: a dictionary that maps speakers ids with to their audio files
    """
    files_dict = {}
    for folder in os.listdir(dir):
        id = folder
        folder = os.path.join(dir,folder)
        files = []
        for subfolder in os.listdir(folder):
            subfolder = os.path.join(folder, subfolder)
            for file in os.listdir(subfolder):
                audio_file = os.path.join(subfolder, file)
                files.append(audio_file)
            files_dict[id] = files
    return files_dict

def get_metadata():
    # load voxceleb metadata
    metadata = pd.read_csv(os.path.join('data', 'vox1_meta.csv'), sep='\t')

    # get the metadata for the voxceleb subset we have selected  
    ids = files_dict.keys()
    metadata = metadata.loc[metadata['VoxCeleb1 ID'].isin(ids)]
    return metadata
    

# get voxceleb dev files
dev_dir = 'data' + os.sep + 'voxceleb_data' + os.sep + 'wav'
dev_files = map_files_to_ids(dev_dir)

# get number of audio files per speaker 
files = pd.Series(dtype=float)
new_dict = {}
ids = dev_files.keys()
for id in ids:
    files[id] = len(dev_files[id])

# select 10 speakers with large numbers of audio files
files = files.sort_values(ascending=False)
files.iloc[:20].plot.barh(title='Number of files per speaker\n(for top 20 speakers with most audio files)')
plt.savefig(os.path.join('results', 'files_per_speaker.png'))
plt.close()
files_to_keep = files.iloc[4:14] 
ids = files_to_keep.index.to_list()

# get all audio files to be used for classification
files_dict = {id: dev_files[id] for id in ids}

# get number of files for training, validation and test set for a 80/10/10 split
total_files = files_to_keep.sum()
print(total_files)
test_files_num = int(total_files * 10 / 100)
val_files_num = int(total_files * 10 / 100)
train_files_num = int(total_files - (test_files_num + val_files_num))
files_per_id = round(test_files_num / files.shape[0])

# get plot of number of files per dataset
files_info = pd.Series({'Total': total_files, 'Training': train_files_num, 'Validation': val_files_num, 'Test': test_files_num})
files_info.plot.barh(title='Number of audio files')
plt.tight_layout()
plt.savefig(os.path.join('results', 'files_numbers.png'))
plt.close()

# create training, validation and test set files from voxceleb dev dataset
train_files = {}
val_files = {}
test_files = {}
for id in ids:
    train_files[id] = dev_files[id][files_per_id*2:]
    val_files[id] = dev_files[id][:files_per_id-1]
    test_files[id] = dev_files[id][files_per_id:files_per_id*2]


