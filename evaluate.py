import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from preprocess_files import files_dict

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

# load image from test set
img_path = 'data' + os.sep + 'test' + os.sep + 'id11184' + os.sep + '1.png'
img = load_img(img_path)
img.show()

# convert image to numpy array
img = img.resize((128, 147))
img = img_to_array(img)
img = img.reshape(-1,128, 147,3)
print(img.shape)

# make prediction
model = load_model('model.h5')
y_prob = model.predict(img) 
y_classes = y_prob.argmax(axis=-1)
print(f'class {y_classes} with probability {y_prob}')







