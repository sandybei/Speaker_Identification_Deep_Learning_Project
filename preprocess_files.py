from ntpath import join
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_metadata():
    """
    reads voxceleb metadata file and keeps speakers names and ids
    """
    # load file with metadata
    metadata = pd.read_csv(os.path.join('data', 'vox1_meta.csv'), sep='\t')
    # keep id and name columns
    metadata = metadata[['VoxCeleb1 ID', 'VGGFace1 ID']]
    return metadata


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
files_sum = pd.Series(dtype=float)
new_dict = {}
ids = dev_files.keys()
for id in ids:
    files_sum[id] = len(dev_files[id])

# select 10 speakers for classification
files_sum = files_sum.sort_values(ascending=False)
print('Number of files per speaker:')
print(files_sum.head(20)) 
# keep speakers with similar number of files
files_sum = files_sum.iloc[4:14] 
ids = files_sum.index.to_list()
files_sum.plot.barh()
plt.title('Number of files per speaker')
plt.show()

# get classification labels
metadata = get_metadata()
labels_df = metadata.loc[metadata['VoxCeleb1 ID'].isin(ids)]
labels_df.reset_index(drop=True, inplace=True)

# get number of files for training and test set for a 80/20 split
total_files = files_sum.sum()
test_files_num = int(total_files * 20 / 100)
train_files_num = int(total_files - test_files_num)
files_per_id = round(test_files_num / files_sum.shape[0])
files_sum = files_sum.iloc[:files_per_id] 

# get all audio files to be used for training / test
files_dict = {id: dev_files[id] for id in ids}

# get training and test set files
train_files = {}
test_files = {}
for id in ids:
    train_files[id] = dev_files[id][files_per_id:]
    test_files[id] = dev_files[id][:files_per_id]

# print file info
print('\n                   Files information                   ')
print('---------------------------------------------------------')
print(f'Total number of audio files: {total_files}')
print('Number of files to be used for training: ', train_files_num)
print('Number of files to be used for test: ', test_files_num)
print('Number of files for each speaker for test set: ', files_per_id)


