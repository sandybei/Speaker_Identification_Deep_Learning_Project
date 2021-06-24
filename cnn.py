import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import os


# get dataset directories
train_dir = os.path.join('data', 'train')
val_dir = os.path.join('data', 'val')

# dataset parameters
batch_size = 64
img_height = 128
img_width = 147
n_channels = 3

img_gen = ImageDataGenerator(rescale=1./255)

# load training dataset
train_ds = img_gen.flow_from_directory(
    train_dir,
    class_mode='categorical',
    target_size=(img_height,img_width),
    seed=0,
    batch_size=batch_size,
    shuffle=True
)

# load validation dataset
val_ds = img_gen.flow_from_directory(
    val_dir,
    class_mode='categorical',
    target_size=(img_height,img_width),
    seed=0,
    batch_size=batch_size,
    shuffle=True
)


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
opt = optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# fit model to data
epochs=20
#callback = EarlyStopping(monitor='val_loss',mode='auto')
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks=[callback]
)

# save model weights to HDF5
model.save("model.h5")


# plot training and validation loss
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot(title='Training vs Validation Loss')
plt.xlabel('Epochs')
plt.savefig(os.path.join('results', 'loss.png'))


