from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

# get datasets directories
train_dir = os.path.join('data', 'train')
val_dir = os.path.join('data', 'val')

# dataset parameters
batch_size = 64
img_height = 128
img_width = 147
n_channels = 3


# load training dataset
train_ds = ImageDataGenerator(rescale=1./255)
train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height,img_width),
    seed=0,
    batch_size=batch_size
)

# load validation dataset
val_ds = ImageDataGenerator(rescale=1./255)
val_ds = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height,img_width),
    seed=0,
    batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# build model
model = Sequential([
  layers.Input(shape=(img_height, img_width, n_channels)),
  layers.Conv2D(16, (3,3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2,2)),
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.BatchNormalization(),
  layers.Flatten(),
  layers.Dense(300, activation='relu'),
  layers.BatchNormalization(),
  layers.Dropout((0.5)),
  layers.Dense(10, activation='softmax')
])

# print model structure
print(model.summary())

# compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# fit model to data
epochs=30
callback = EarlyStopping(monitor='loss', patience=5)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[callback]
)

# save model weights to HDF5
model.save("model.h5")

# plot training and validation loss
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot(title='Training vs Validation Loss')
plt.xlabel('Epochs')
plt.savefig(os.path.join('results', 'loss.png'))


