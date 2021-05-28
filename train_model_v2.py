# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
import signal_processing as sp
import matplotlib.pyplot as plt
import time
import sklearn
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from sys import getsizeof


# %%
# Import datasets
dataset_directory = os.path.join(os.getcwd(), "DeepSig-Dataset-2018")

# Use these to load the entire dataset into memory
fft_dataset_train = np.load(os.path.join(dataset_directory, "fft_dataset_train.npy"))
fft_dataset_val = np.load(os.path.join(dataset_directory, "fft_dataset_val.npy"))
fft_dataset_test = np.load(os.path.join(dataset_directory, "fft_dataset_test.npy"))

labels_train = np.load(os.path.join(dataset_directory, "labels_train.npy"))
labels_val = np.load(os.path.join(dataset_directory, "labels_val.npy"))
labels_test = np.load(os.path.join(dataset_directory, "labels_test.npy"))

# # Use these to use a memory map to save RAM
# fft_dataset_train = np.load(os.path.join(dataset_directory, "fft_dataset_train.npy"), mmap_mode="r+")
# fft_dataset_val = np.load(os.path.join(dataset_directory, "fft_dataset_val.npy"), mmap_mode="r+")
# fft_dataset_test = np.load(os.path.join(dataset_directory, "fft_dataset_test.npy"), mmap_mode="r+")

# labels_train = np.load(os.path.join(dataset_directory, "labels_train.npy"), mmap_mode="r+")
# labels_val = np.load(os.path.join(dataset_directory, "labels_val.npy"), mmap_mode="r+")
# labels_test = np.load(os.path.join(dataset_directory, "labels_test.npy"), mmap_mode="r+")

# %% [markdown]
# Super crude hyper parameter tuning by creating multiple different models

# %%
# Create list of models
models = []


# %%
# Large model SNR 10
inputs = keras.Input(shape=(1024, 1))
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.Conv1D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(filters=512, kernel_size=3, activation='relu')(x)
x = layers.Conv1D(filters=512, kernel_size=3, activation='relu')(x)
x = layers.Conv1D(filters=512, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128)(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(24, activation='softmax')(x)

models.append(keras.Model(inputs=inputs, outputs=outputs, name = "snr_10_nadam"))
models.append(keras.Model(inputs=inputs, outputs=outputs, name = "snr_10_adagrad"))

models[-1].summary()


# %%
# Create list of optimizers to use
optimizers = [keras.optimizers.Nadam(learning_rate=0.005), keras.optimizers.Adagrad(learning_rate=0.005)]


# %%
# # Enable for quick testing
# fft_dataset_train = fft_dataset_train[:100]
# fft_dataset_val = fft_dataset_val[:100]
# fft_dataset_test = fft_dataset_test[:100]

# labels_train = labels_train[:100]
# labels_val = labels_val[:100]
# labels_test = labels_test[:100]


# %%
for optimizer in optimizers:
    for model in models:
        # Time the time it takes to train the model
        start = time.time()
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Define the callbacks and save the best model to a new file
        callbacks = [keras.callbacks.ModelCheckpoint(filepath=f'models/{model.name}.keras', save_best_only=True, monitor='val_loss'), 
            keras.callbacks.EarlyStopping(monitor="val_loss", min_delta = 0.002, patience = 10, verbose = 1, restore_best_weights = True)]
        # Train model
        history = model.fit(fft_dataset_train, labels_train, epochs=100, batch_size = 256, validation_data = (fft_dataset_val, labels_val), callbacks=callbacks)

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


