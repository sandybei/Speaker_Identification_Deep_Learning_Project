from preprocess_files import train_files, test_files
from audio_analysis import *
import os
import matplotlib.pyplot as plt


def create_directory(folder, files_dict):
    os.mkdir(folder)
    for id in files_dict.keys():
        subfolder = os.path.join(folder,id)
        os.mkdir(subfolder)
        for i, file in enumerate(files_dict[id]): 
            fig = preprocess(file)
            file_name = os.path.join(subfolder, str(i) + '.png')
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    return


# create directory for spectrogram images of each speaker
dir = os.path.join('data', 'images')
os.mkdir(dir)

# get best width for spectrogram images
best_width = optimal_image_width(False)

# create directory for training set
folder = os.path.join(dir, 'train')
create_directory(folder, train_files)

# create directory for test set
folder = os.path.join(dir, 'test')
create_directory(folder, test_files)


#spec = get_spectrogram(dev_files['id11229'][2])
