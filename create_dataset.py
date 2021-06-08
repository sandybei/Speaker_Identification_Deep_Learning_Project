from preprocess_files import files_dict
from audio_analysis import preprocess
import numpy as np
import os
import matplotlib.pyplot as plt


# create directory with spectrogram images for each speaker
folder = 'data' + os.sep + 'images'
os.mkdir(folder)
for id in files_dict.keys():
    subfolder = os.path.join(folder,id)
    os.mkdir(subfolder)
    for i, file in enumerate(files_dict[id]): 
        base = os.path.basename(file)
        file_name = os.path.join(subfolder, str(i) + '.png')
        fig = preprocess(file)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)

