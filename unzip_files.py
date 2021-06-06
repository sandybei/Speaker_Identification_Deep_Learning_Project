import os
import zipfile

data_directory = 'data'
for filename in os.listdir(data_directory):
    filepath = os.path.join(data_directory, filename)
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall('data')
            print('success')
    else:
        print('Not a zip file.')

