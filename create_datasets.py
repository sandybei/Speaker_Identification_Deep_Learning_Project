from preprocess_files import train_files, val_files, test_files
from audio_analysis import preprocess
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def create_directory(folder, files_dict):
    """
    This function is used for the creation of the image directories of the training, validation and test set.
    It creates a directory with folders whose names are the speakers ids and it saves inside them the spectrogram 
    images that are extracted from their audio files by the 'preprocess' function.

    :param folder: folder name of directory to be created
    :param files_dict: dictionary that maps speaker ids to their audio files
    :return: 
    """
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


if __name__ == "__main__":
    # create directory for training set 
    folder = os.path.join('data', 'train')
    create_directory(folder, train_files)

    # create directory for validation set 
    folder = os.path.join('data', 'val')
    create_directory(folder, val_files)

    # create directory for test set
    folder = os.path.join('data', 'test')
    create_directory(folder, test_files)

