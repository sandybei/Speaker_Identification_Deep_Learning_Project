import pandas as pd
import matplotlib.pyplot as plt
from process_files import files_dict, get_metadata
import os


metadata = get_metadata()
# print speakers selected for classification
print('Selected Speakers:\n', metadata['VGGFace1 ID'])

# get plot of gender distribution
metadata.reset_index(drop=True, inplace=True)
group_gender = metadata.groupby(['Gender']).size()
fig = group_gender.plot.barh()
plt.title('Gender Distribution')
plt.savefig(os.path.join('results', 'genders.png'))
plt.close()

# get plot of nationality distribution
group_nation = metadata.groupby(['Nationality']).size()
fig = group_nation.plot.barh()
plt.title('Nationality Distribution')
plt.savefig(os.path.join('results', 'nationalities.png'))
plt.close()