import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model('models/snr_10_adam_smaller_model_kernel5.keras')

# Import datasets
dataset_directory = os.path.join(os.getcwd(), "DeepSig-Dataset-2018")

# Use these to load the entire dataset into memory
fft_dataset_train = np.load(os.path.join(dataset_directory, "fft_dataset_train.npy"))
fft_dataset_val = np.load(os.path.join(dataset_directory, "fft_dataset_val.npy"))
fft_dataset_test = np.load(os.path.join(dataset_directory, "fft_dataset_test.npy"))

labels_train = np.load(os.path.join(dataset_directory, "labels_train.npy"))
labels_val = np.load(os.path.join(dataset_directory, "labels_val.npy"))
labels_test = np.load(os.path.join(dataset_directory, "labels_test.npy"))

# Time the time it takes to train the model
start = time.time()

optimizer = "adam"

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define the callbacks and save the best model to a new file
callbacks = [keras.callbacks.ModelCheckpoint(filepath=f'models/{model.name}.keras', save_best_only=True, monitor='val_loss')]

# Train model
history = model.fit(fft_dataset_train, labels_train, epochs=200, batch_size = 256, validation_data = (fft_dataset_val, labels_val), callbacks=callbacks)

# Save pickle of history
with open(f"models/{model.name}.pickle", mode='wb') as f:
	pickle.dump(history.history, f)

# Evaluate model
_, accuracy = model.evaluate(fft_dataset_test, labels_test)

print("--- Model trained in %s seconds ---" % (time.time() - start))

# Write accuracy to file
f = open("models/accuracy_log.txt", "a")
f.write(f"Model {model.name} evaluated an accuracy of {accuracy}.\n")
f.write(f"Model {model.name} trained in {int(time.time() - start)} seconds.\n\n")
f.close()