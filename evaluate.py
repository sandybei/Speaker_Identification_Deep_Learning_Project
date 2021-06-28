import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from process_files import get_metadata
from glob import glob


def prepare(file):
    """
    This function gets an image and prepares it to feed it to the CNN.

    :param file: filepath of image
    :return: 1 batch of the image as numpy array
    """
    img = image.load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    return img


# get datasets directories
test_dir = os.path.join('data', 'test')

# set dataset generator parameters
BATCH_SIZE = 64
IMG_HEIGHT = 128
IMG_WIDTH = 128
N_CHANNELS = 3

# load test dataset
img_gen = ImageDataGenerator(rescale=1./255)
test_ds = img_gen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    seed=0,
    batch_size=BATCH_SIZE
)

# load model
model = load_model('model.h5')
print('model loaded')

# evaluate model on test set
score = model.evaluate(test_ds, verbose=1, steps=len(test_ds)) 
test_loss = round(score[0],3)
test_accuracy = round(score[1],3)*100
print('Test loss:', test_loss) 
print(f'Test accuracy: {test_accuracy} %\n')

# get class names
folders = glob("data/test/*") 
class_names = [os.path.basename(folder) for folder in folders]
class_names = sorted(class_names) 

# make predictions on 3 images from the test set
files = [os.path.join(test_dir, 'id10343'+ os.sep + '41.png'),
    os.path.join(test_dir, 'id11184'+ os.sep + '6.png'),
    os.path.join(test_dir, 'id10945'+ os.sep + '10.png')
]

for i, file in enumerate(files):
    # prepare image
    test_img = prepare(file)
    # get softmax probability for each class
    prob = model.predict(test_img)
    index = np.argmax(prob)
    prob_max = np.max(prob)
    prob_max = round(float(prob_max),3) * 100
    # get predicted spaker name
    id = class_names[index]
    metadata = get_metadata()
    pred_speaker = metadata.loc[metadata['VoxCeleb1 ID'] == id]
    pred_name = pred_speaker['VGGFace1 ID'].item()
    # get true speaker name
    true_id = os.path.basename(os.path.dirname(file))
    true_speaker = metadata.loc[metadata['VoxCeleb1 ID'] == true_id]
    true_name = true_speaker['VGGFace1 ID'].item()
    # print results 
    print('Prediction: ', i + 1)
    print(f'Predicted speaker {pred_name} with probability {prob_max} %')
    print('True speaker name:', true_name)
    print('')



