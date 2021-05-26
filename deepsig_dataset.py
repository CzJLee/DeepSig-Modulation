import h5py
import os
import numpy as np
# import tensorflow as tf
import signal_processing as sp
import matplotlib.pyplot as plt 

# Get location of DEEPSIG DATASET: RADIOML 2018.01A dataset
current_working_directory = os.getcwd()
file_name = os.path.join(current_working_directory, "datasets/large/DeepSig-Dataset-2018/GOLD_XYZ_OSC.0001_1024.hdf5")

# Load File
f = h5py.File(file_name, 'r')

# Print list of all dict keys in the file
print(list(f.keys()))

# Get the dataset from each key 
x = f["X"] # Dataset
y = f["Y"] # Labels
z = f["Z"] # SNR Value

print(x.shape) # (2555904, 1024, 2)
print(y.shape) # (2555904, 24)
print(z.shape) # (2555904, 1)

# Copy class labels list from classes.txt
classes = ['32PSK',
	'16APSK',
	'32QAM',
	'FM',
	'GMSK',
	'32APSK',
	'OQPSK',
	'8ASK',
	'BPSK',
	'8PSK',
	'AM-SSB-SC',
	'4ASK',
	'16PSK',
	'64APSK',
	'128QAM',
	'128APSK',
	'AM-DSB-SC',
	'AM-SSB-WC',
	'64QAM',
	'QPSK',
	'256QAM',
	'AM-DSB-WC',
	'OOK',
	'16QAM']

# # Create function to convert one-hot vector y to its class label
# def get_class_label(one_hot_vector):
# 	index = tf.argmax(one_hot_vector, axis=0)
# 	return classes[index]

# print(get_class_label(y[1]))

# Count number of SNR dB value occurrences
np_z = np.reshape(np.array(z), (-1, ))
from collections import Counter
count = Counter(np_z)
print(count)

# # print(x[0])
# # Convert to IQ modulation
# x0 = np.reshape(x[1700002], (-1, ))
# x1 = np.reshape(x[1700003], (-1, ))

# x2 = np.concatenate((x0, x1))
# # print(x2)
# xc = sp.signal_to_complex(x2)

# sp.plot_signal(xc[500:-500], 1e6, 900e6)

# import time
# start_time = time.time()
# i = 0
# for data in np_z:
# 	if i % 10000 == 0:
# 		print(i)
# 	i += 1
# print(f"{time.time() - start_time} seconds.")

import time

start_time = time.time()
np_z = np.reshape(np.array(z), (-1, ))
print(f"Time to convert z: {time.time() - start_time} seconds.")

start_time = time.time()
np_y = np.reshape(np.array(y[1000000:2000000]), (-1, ))
print(f"Time to convert y: {time.time() - start_time} seconds.")

# start_time = time.time()
# np_x = np.reshape(np.array(x), (-1, ))
# print(f"Time to convert x: {time.time() - start_time} seconds.")

# start_time = time.time()
# good = np.zeros(np.shape(np_z))
# for i, value in enumerate(np_z):
# 	if value >= 0:
# 		good[i] = 1
# print(f"{time.time() - start_time} seconds.")

# # plt.plot(good)
# # plt.show()

# npz2 = np.array(z[0:1000000, 1100000:2100000, 2200000:2300000])

# print(len(npz2))

# 983040 values we don't want with SNR < 0
# /24 -> 40960 per sample
# 106496 of each signal
# [40960:106496]

# n = 106496
# start = 40960+n*23
# end = 106496+n*23
# np_z = np.reshape(np.array(z[start:end]), (-1, ))
# from collections import Counter
# count = Counter(np_z)
# print(count)
# print(end)

start_time = time.time()
all = np.empty(0, dtype=np.int64)
for i in range(24):
	n = 106496
	start = 40960+n*i
	end = 106496+n*i
	all = np.append(all, np.array(z[start:end]))
print(f"{time.time() - start_time} seconds.")

from collections import Counter
count = Counter(all)
print(count)

print(type(z[0,0]))

def slice_snr(arr):
	all = np.empty(0, dtype=np.float32)
	for i in range(24):
		start_time = time.time()
		n = 106496
		start = 40960+n*i
		end = 106496+n*i
		all = np.array(arr[start:end])
		print(f"{time.time() - start_time} seconds")
		yield all
start_time = time.time()
np_x_list = [blah for blah in slice_snr(x)]
np_x = np.concatenate(np_x_list, axis=0)
print(f"{time.time() - start_time} seconds.")
print(np.shape(np_x))

# for blah in slice_snr(y):
# 	print(np.shape(blah))