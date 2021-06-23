import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from preprocess_files import metadata
from glob import glob


def prepare_image(file):
    img = image.load_img(file, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img


# get datasets directories
test_dir = os.path.join('data', 'test')

# dataset loader parameters
batch_size = 64
img_height = 128
img_width = 147

# load test dataset
test_ds = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height,img_width),
    seed=123,
    batch_size=batch_size
)

# load model
model = load_model('model.h5')

# evaluate model on test set
score = model.evaluate(test_ds, verbose=1) 
test_loss = round(score[0],2)
test_accuracy = round(score[1],2)*100
print('Test loss:', test_loss) 
print(f'Test accuracy: {test_accuracy} %\n')

# make predictions
file_1 = test_dir + os.sep + 'id10715' + os.sep + '41.png' 
file_2 = test_dir + os.sep + 'id10397' + os.sep + '3.png'
file_3 = test_dir + os.sep + 'id10935' + os.sep + '11.png'
files = [file_1,file_2,file_3]
for i, file in enumerate(files):
    test_img = prepare_image(file)
    # get probability

    folders = glob("data/test/*") 
    print(folders)
    class_names = [os.path.basename(folder) for folder in folders]
    class_names = sorted(class_names) 
    y_prob = model.predict(test_img)
    y_class = np.argmax(y_prob, axis=-1)
    y_prob_max = np.max(y_prob)
    prob = round(y_prob_max,2) * 100

    # get predicted class
    y_class = y_class[0]
    id = class_names[y_class]
    speaker = metadata.loc[metadata['VoxCeleb1 ID'] == id]
    speaker = speaker['VGGFace1 ID'].item()


    true_id = os.path.basename(os.path.dirname(file))
    true_speaker = metadata.loc[metadata['VoxCeleb1 ID'] == true_id]
    true_speaker = true_speaker['VGGFace1 ID'].item()
    print('Prediction: ', i + 1)
    print(f'Predicted speaker {speaker} with probability: {prob} %')
    print('True speaker name: ', true_speaker)
    print('-------------------------------------')







