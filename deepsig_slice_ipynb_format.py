# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import h5py
import os
import numpy as np
import signal_processing as sp
import matplotlib.pyplot as plt
import time
import sklearn
from fastprogress.fastprogress import progress_bar
from sys import getsizeof


# %%
# Get location of DEEPSIG DATASET: RADIOML 2018.01A dataset
current_working_directory = os.getcwd()
file_name = os.path.join(current_working_directory, "datasets/large/DeepSig-Dataset-2018/GOLD_XYZ_OSC.0001_1024.hdf5")

# Load File
start = time.time()
print("Loading dataset...")
f = h5py.File(file_name, 'r')

# Get the dataset from each key 
x = f["X"] # Dataset
y = f["Y"] # Labels
z = f["Z"] # SNR Value

print(f"Dataset loaded in {time.time() - start:.6f} seconds.")


# %%
# It is significantly faster to reshape an array than to concatenate multiple arrays. 
# Start be creating an empty array of a known size, where the first axis is the slices of the dataset we are keeping
# Then reassign the elements of this empty array to be the slices of the larger array
# Finally, reshape the array to the final shape

print("Slicing dataset with SNR >= 0")
start = time.time()

# For our dataset, x, we want to pick out the data where the SNR is greater than or equal to zero
# There are 65,536 such samples for each of the 24 modulation types
# The datatype of the X dataset is float32. This is important.
# While the original datatype of Y is int64, use int16 to save a tiny bit of memory.
dataset_slices = np.empty((24, 65536, 1024, 2), dtype=np.float32)
labels_slices = np.empty((24, 65536, 24), dtype = np.int16)

# We can then iterate over each of the modulation types
# The numbers used in these are specifically chosen to pick out only SNR >= 0 values. 
for i in progress_bar(range(24)):
    n = 106496
    start = 40960+n*i
    end = 106496+n*i
    dataset_slices[i] = x[start:end]
    labels_slices[i] = y[start:end]

# We can then reshape this array to the final shape, which is signfically faster than concatenation
dataset = np.reshape(dataset_slices, (1572864, 1024, 2))
labels = np.reshape(labels_slices, (1572864, 24))

print(f"Datasets created in {time.time() - start:.6f} seconds.")

# For fun, I timed the process of this method versus the old method using concatenation, and it is about 3 times faster. Plus, we get to see the progress, whereas concatenation just runs with no progress indicators. Reshaping is nearly instant. 


# %%
# The next step is to convert our dataset to complex values.
# We can use our empty dataset method to speed things up to avoid concatenation
print("Converting dataset to complex values...")
start = time.time()

dataset_complex = np.empty((len(dataset), 1024), dtype = np.complex64)
for i in progress_bar(range(len(dataset))):
    # Flatten the dataset, to be in I/Q format that our signal_to_complex function expects
    dataset_complex[i] = sp.signal_to_complex(np.reshape(dataset[i], (-1, )))

print(f"Converted dataset to complex values in {time.time() - start:.6f} seconds.")

# We can then remove the original dataset from memory to save precious memory
del dataset


# %%
# Now we need to take a FFT of the complex dataset
print("Calculating FFT of dataset...")
start = time.time()

fft_dataset = np.empty((len(dataset_complex), 1024), dtype = np.float64)
for i in progress_bar(range(len(dataset_complex))):
    fft_dataset[i] = sp.fft_signal(dataset_complex[i])

print(f"Completed FFT calculaations in {time.time() - start:.6f} seconds.")

# Remove the complex dataset from memory
del dataset_complex


# %%



