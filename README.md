# Deep_Learning_Project

## Team Members
Kyriaki Bei - sandybei058@gmail.com

## Subject
Speaker Identitication by Voice. 

## Dataset

[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. The dataset consists of two versions, VoxCeleb1 and VoxCeleb2. For this project the dataset [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) was selected, that contains over 100,000 utterances for 1,251 celebrities.

### Download Dataset
1. Download the following files in 'data' folder:
```
vox1_dev_wav_partaa
vox1_dev_wav_partab
vox1_dev_wav_partac
vox1_dev_wav_partad
```

2. Concatenate files using the following command:
```
$ cat vox1_dev* > vox1_dev_wav.zip
```

3. Unzip vox1_dev_wav.zip file inside 'data' folder:

### Download metadata
Download a file with full names, nationality and gender labels for all the speakers from by clicking on the following link:

https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv

## Installation

Clone the source of this library: 
```
git clone https://github.com/sandybei/Speaker_Identification_Deep_Learning_Project.git
```

Install the required dependencies using pip:
```
pip install matplotlib
pip install numpy 
pip install pandas
pip install librosa
pip install pyAudioAnalysis
pip install tensorflow
pip install keras
```

## How to Use 
Follow the next steps to generate results: 

### Dataset Creation
Run the following command to extract a VoxCeleb1 subset that will be used for classification:

```
$ python file_preprocess.py
```
### Data Exploration
Run the following command to visualize information about the dataset:
```
$ python data_explore.py
```
### Data Preprocessing
Run the following command to create directories of audio spectrogram images:
```
$ python create_directories.py
```

### Training
Run the following command to train the model:
```
$ python cnn.py
```

### Evaluation
5. Run the following command to evaluate the model's performance and make some predictions on test images:
```
$ python evaluate.py
```

## References
[1] [VoxCeleb: a large-scale speaker identification dataset](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf), A. Nagrani, J. S. Chung, A. Zisserman, INTERSPEECH, 2017.
      

