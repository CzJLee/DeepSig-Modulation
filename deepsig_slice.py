import h5py
import os
import numpy as np
import signal_processing as sp
import matplotlib.pyplot as plt
import time

# Get location of DEEPSIG DATASET: RADIOML 2018.01A dataset
current_working_directory = os.getcwd()
file_name = os.path.join(current_working_directory, "datasets/large/DeepSig-Dataset-2018/GOLD_XYZ_OSC.0001_1024.hdf5")

# Load File
f = h5py.File(file_name, 'r')

# Get the dataset from each key 
x = f["X"] # Dataset
y = f["Y"] # Labels
z = f["Z"] # SNR Value

def slice_snr(arr, dtype):
	slice_snr_greater_than_zero = np.empty(0, dtype)
	for i in range(24):
		n = 106496
		start = 40960+n*i
		end = 106496+n*i
		slice_snr_greater_than_zero = np.array(arr[start:end])
		print(f"Imported section {i+1} of 24.", end = "\r")
		yield slice_snr_greater_than_zero

blah = sp.signal_to_complex(x[0])
print(x[0])

# print("Importing y")
# np_y_list = [blah for blah in slice_snr(y, dtype = np.int16)]
# print()
# print("Concatenating y")
# labels = np.concatenate(np_y_list, axis=0)

# print("Importing x")
# np_x_list = [blah for blah in slice_snr(x, dtype = np.float32)]
# print()
# print("Concatenating x")
# dataset = np.concatenate(np_x_list, axis=0)

# new_dataset = np.empty(0, dtype=np.complex32)