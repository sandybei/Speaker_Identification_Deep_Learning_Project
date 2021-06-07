import pandas as pd
from pathlib import Path
import os

folder_path = 'data' + os.sep + 'metadata' + os.sep 

# load file with metadata
metadata = pd.read_csv(folder_path + 'vox1_meta.csv', sep='\t')
print(metadata.head())

# keep id and name columns
metadata = metadata[['VoxCeleb1 ID', 'VGGFace1 ID']]
print(metadata.head())

metadata.to_csv(folder_path + 'metadata.csv')


