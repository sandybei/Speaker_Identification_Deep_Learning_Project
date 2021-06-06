import pandas as pd
from pathlib import Path
import os

# load file with metadata
metadata = pd.read_csv('data' + os.sep + 'metadata' + os.sep + 'vox1_meta.csv', sep='\t')
print(metadata.head())

# keep id and name columns
metadata = metadata[['VoxCeleb1 ID', 'VGGFace1 ID']]
print(metadata.head())


