from preprocess_files import train_files, test_files
from audio_analysis import preprocess
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def create_directory(folder, files_dict):
    os.mkdir(folder)
    for id in files_dict.keys():
        subfolder = os.path.join(folder,id)
        os.mkdir(subfolder)
        for i, file in enumerate(files_dict[id]): 
            fig = preprocess(file)
            file_name = os.path.join(subfolder, str(i) + '.png')
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            fig.clear()
            plt.close(fig)
    return


# create directory for training set
folder = os.path.join('data', 'train')
create_directory(folder, train_files)

# create directory for test set
folder = os.path.join('data', 'test')
create_directory(folder, test_files)

