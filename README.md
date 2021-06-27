# Deep_Learning_Project

### Team Members
Kyriaki Bei - sandybei058@gmail.com

### The Project
Speaker Identitication by Voice. 

## Dataset

[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. The dataset consists of two versions, VoxCeleb1 and VoxCeleb2. For this project the dataset VoxCeleb1 was selected, that contains over 100,000 utterances for 1,251 celebrities.

A file with full names, nationality and gender labels for all the speakers in the dataset is also provided. From this file the full name will be used for the verification of the speaker.

**1. Download files:**
```
vox1_dev_wav_partaa
vox1_dev_wav_partab
vox1_dev_wav_partac
vox1_dev_wav_partad
```

**2. Concatenate files:**\
Concatenate all files using the following command:
```
$ cat vox1_dev* > vox1_dev_wav.zip
```

**3. Put all files in 'data' folder.**
**4. Unzip files:**\
Unzip vox1_dev_wav.zip inside data folder.

# Dependencies


## References
[1] [VoxCeleb: a large-scale speaker identification dataset](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf), A. Nagrani, J. S. Chung, A. Zisserman, INTERSPEECH, 2017.
      

