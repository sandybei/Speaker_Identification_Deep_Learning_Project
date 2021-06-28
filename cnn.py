from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import os


# get dataset directories
train_dir = os.path.join('data', 'train')
val_dir = os.path.join('data', 'val')

# set dataset loader parameters
BATCH_SIZE = 64
IMG_HEIGHT = 128
IMG_WIDTH = 128
N_CHANNELS = 3

img_gen = ImageDataGenerator(rescale=1./255)

# load training dataset
train_ds = img_gen.flow_from_directory(
    train_dir,
    class_mode='categorical',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    seed=0,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# load validation dataset
val_ds = img_gen.flow_from_directory(
    val_dir,
    class_mode='categorical',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    seed=0,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# build model
model = Sequential([
  layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.BatchNormalization(),
  layers.Flatten(),
  layers.Dense(300, activation='relu'),
  layers.BatchNormalization(),
  layers.Dropout((0.6)),
  layers.Dense(10, activation='softmax')
])

# print model structure
print(model.summary())

# compile model
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# fit model to data
epochs = 50
early_stopping = EarlyStopping(
    min_delta=0.001, 
    patience=10, 
    restore_best_weights=True,
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[early_stopping]
)

# save model weights to HDF5
model.save("model.h5")

# plot training vs validation loss and accuracy
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot(title='Training vs Validation Loss')
plt.xlabel('Epochs')
plt.savefig(os.path.join('results', 'loss.png'))

history_frame.loc[:, ['accuracy', 'val_accuracy']].plot(title='Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.savefig(os.path.join('results', 'accuracy.png'))




