import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess_files import metadata
from glob import glob


def prepare_image(file):
    img = image.load_img(file, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    return img


# get datasets directories
test_dir = os.path.join('data', 'test')

# dataset loader parameters
batch_size = 64
img_height = 128
img_width = 147

# load test dataset
img_gen = ImageDataGenerator(rescale=1./255)
test_ds = img_gen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    target_size=(img_height,img_width),
    seed=0,
    batch_size=batch_size
)

# load model
model = load_model('model.h5')
print('model loaded')

# evaluate model on test set
score = model.evaluate(test_ds, verbose=1, steps=len(test_ds)) 
test_loss = round(score[0],2)
test_accuracy = round(score[1],2)*100
print('Test loss:', test_loss) 
print(f'Test accuracy: {test_accuracy} %\n')


# get class names
folders = glob("data/test/*") 
class_names = [os.path.basename(folder) for folder in folders]
class_names = sorted(class_names) 

# make predictions on test images
files = [
    os.path.join(test_dir, 'id10343'+ os.sep + '41.png'),
    os.path.join(test_dir, 'id11184'+ os.sep + '6.png'),
    os.path.join(test_dir, 'id10945'+ os.sep + '10.png')
]

for i, file in enumerate(files):
    # prepare image
    test_img = prepare_image(file)
    # get softmax probability 
    prob = model.predict(test_img)
    index = np.argmax(prob)
    prob_max = np.max(prob)
    prob_max = round(float(prob_max),2) * 100
    # get predicted spaker name
    id = class_names[index]
    pred_speaker = metadata.loc[metadata['VoxCeleb1 ID'] == id]
    pred_name = pred_speaker['VGGFace1 ID'].item()
    # get true speaker name
    true_id = os.path.basename(os.path.dirname(file))
    true_speaker = metadata.loc[metadata['VoxCeleb1 ID'] == true_id]
    true_name = true_speaker['VGGFace1 ID'].item()
    # print results 
    print('Prediction: ', i + 1)
    print(f'Predicted speaker {pred_name} with probability: {prob_max} %')
    print('True speaker name: ', true_name)
    print('-------------------------------------')







