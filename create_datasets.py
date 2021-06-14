from preprocess_files import train_files, test_files
from audio_analysis import preprocess
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil


def create_directory(folder, files_dict):
    os.mkdir(folder)
    for id in files_dict.keys():
        subfolder = os.path.join(folder,id)
        os.mkdir(subfolder)
        for i, file in enumerate(files_dict[id]): 
            head, tail = os.path.split(file)
            #file_name = os.path.join(subfolder, tail)
            fig = preprocess(file)
            file_name = os.path.join(subfolder, str(i) + '.png')
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
    return


# create directory for spectrogram images of each speaker
dir = 'data' + os.sep + 'images'
os.mkdir(dir)

# create directory for training set
folder = os.path.join(dir, 'train')
create_directory(folder, train_files)

# create directory for test set
folder = os.path.join(dir, 'test')
create_directory(folder, test_files)


