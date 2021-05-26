import numpy as np
import matplotlib.pyplot as plt

def signal_to_complex(data):
	"""
	Converts an IQ datastream recording to complex numbers if not already. 

	Args:
		data (np.ndarray): Time domain data stream recording.

	Returns:
		np.ndarray: Complex valued time domain data stream recording.
	"""
	# Check if data is already complex values, or not. 
	# If not complex, then format is alternating Real and Imaginary
	if data.dtype == np.int16:
		RX_BITS = 16
		# By default, I am saving recording signals as dtype np.int16. We want to convert them to complex.
		# These numbers also are not normalize. Let's first normalize the values to between [1, 1].
		data = np.divide(data, np.power(2.0, RX_BITS-1))
		# Split scaled_data into real and imaginary components
		data = (data[::2] + 1j * data[1::2])

	# Handle if data type is float32, for example, in the DeepSig dataset
	elif data.dtype == np.float32:
		data = (data[::2] + 1j * data[1::2])
	
	return data


def fft_signal(data):
	"""
	Calculate the signal FFT of a single 1D datastream and returns the dB values in the frequency domain.

	If data is not already a complex time domain signal, it is converted to complex before the FFT.
	In some cases, the FFT may produce some zero values. To avoid returning a -Inf value as a result of taking the log of a zero value, any -Inf value is substituted with the value -150.

	Args:
		data (np.ndarray): Time domain data stream recording.

	Returns:
		np.ndarray: Returns a FFT of the given data. The returned numpy array is the same shape as the given data. 
	"""
	# Make data complex if not already.
	data = signal_to_complex(data)

	N = len(data)
	# Take the fourier transform of the signal and perform FFT Shift
	fft_shift = np.fft.fftshift(np.fft.fft(data, N) / N)
	# Take log to convert to dB
	# In some cases, fft_shift will contain a value of zero, which will throw an error since you can not take log(0)
	# In this case, replace the value -Inf with -150
		# Slow method: run np.nan_to_num on all values before returning
		# nan_to_num slows down this function by about 30% compared to no correction
		# np.where seems to be a bit faster, slowing the function down by about 15-20%. 
		# I think the best case would be to react only if numpy raises the divide by zero RuntimeWarning, but I couldn't figure out a way to react to the warning and modify the array without recalculating it. 
	# return 20 * np.log10(np.abs(fft_shift)) # No -Inf correction
	with np.errstate(divide = "ignore"):
		fft_log = 20 * np.log10(np.abs(fft_shift))
		return np.where(fft_log == np.NINF, -150, fft_log)

def fft_dataset(dataset):
	"""
	Calculate the FFT of a dataset of recordings. 

	Args:
		dataset (np.ndarray): 2D array of time domain signal recordings, where each row corresponds to a recording. 

	Returns:
		np.ndarray: Returns an array of the same shape as the give dataset, where a signal FFT is calculated for each row individually.
	"""
	# Dataset should be a 2D numpy array
	# Apply FFT on each row. 
	return np.apply_along_axis(fft_signal, 1, dataset)

def max_pooling(data, pool_size = 2):
	"""
	Apply a max pooling function to a 1D array. The data is divided into clusters of pool_size, and only the maximum value is kept. The returned array is of size len(data) / pool_size. The size of data must be divisible by pool_size.

	Args:
		data (np.ndarray): 1D array of data to be pooled.
		pool_size (int, optional): Size of sections of data to pool. A larger value will return a smaller. The size of data must be divisible by pool_size. Defaults to 2.

	Returns:
		np.ndarray: Returns a max pooled array where only the maximum value per pool section is kept. The returned array is of size len(data) / pool_size.
	"""
	data_reshape = np.reshape(data, (-1, pool_size))
	data_max_pool = data_reshape.max(1)
	return data_max_pool

def plot_fft(data, sample_rate, tuning_frequency):
	"""
	Plot the FFT of a given time domain signal recording with known sample rate and tuning frequency. 
	It is necessary to know the sample rate and tuning frequency in order to determing the x-axis for the FFT plot.

	Plot using matplotlib.

	Args:
		data (np.array): 1D array time domain signal recording
		sample_rate (int): Sample rate in Hz
		tuning_frequency (int): L0 tuning frequency in Hz
	"""
	N = len(data)
	# plt.figure(num=1, figsize=(11, 8.5))

	# Frequency Domain Plot
	# plt.subplot(212)
	f_ghz = (tuning_frequency + (np.arange(0, sample_rate, sample_rate/N) - (sample_rate/2) + (sample_rate/N))) / 1e9
	plt.plot(f_ghz, data)
	plt.xlim(f_ghz[0], f_ghz[-1])
	plt.ylim(-140, 0)
	plt.xlabel('Frequency (GHz)')
	plt.ylabel('Amplitude (dBFS)')
	plt.title("Frequency Domain Plot")
	plt.show()

def plot_signal(data, sample_rate, tuning_frequency):
	"""
	Plot the time domain signal and the FFT of a given time domain signal recording with known sample rate and tuning frequency.

	Plot using matplotlib.
	It is necessary to know the sample rate and tuning frequency in order to determing the x-axis for the FFT plot.

	Args:
		data (np.array): 1D array time domain signal recording
		sample_rate (int): Sample rate in Hz
		tuning_frequency (int): L0 tuning frequency in Hz
	"""
	# Make data complex if not already
	data = signal_to_complex(data)

	N = len(data)
	
	# Time Domain Plot
	plt.figure(num=1, figsize=(11, 8.5))
	plt.subplot(211)
	t_us = np.arange(N) / sample_rate / 1e-6
	plt.plot(t_us, data.real, 'k', label='I')
	plt.plot(t_us, data.imag, 'r', label='Q')
	plt.xlim(t_us[0], t_us[-1])
	plt.xlabel('Time (us)')
	plt.ylabel('Normalized Amplitude')

	fft_data = fft_signal(data)

	# Frequency Domain Plot
	plt.subplot(212)
	f_ghz = (tuning_frequency + (np.arange(0, sample_rate, sample_rate/N) - (sample_rate/2) + (sample_rate/N))) / 1e9
	plt.plot(f_ghz, fft_data)
	plt.xlim(f_ghz[0], f_ghz[-1])
	plt.ylim(-140, 0)
	plt.xlabel('Frequency (GHz)')
	plt.ylabel('Amplitude (dBFS)')
	plt.show()
