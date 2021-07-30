# DeepSig Modulation Machine Learning Classification

- [DeepSig Modulation Machine Learning Classification](#deepsig-modulation-machine-learning-classification)
	- [The Paper](#the-paper)
	- [The Dataset](#the-dataset)
		- [I/Q Signal Recordings](#iq-signal-recordings)
		- [Label](#label)
		- [SNR](#snr)
	- [Recreating the model](#recreating-the-model)
		- [Data Input](#data-input)
		- [Project Goal](#project-goal)
		- [Results](#results)

In 2017, DeepSig published the paper [Over-the-Air Deep Learning Based Radio Signal Classification](https://ieeexplore.ieee.org/document/8267032). 

In 2018, they released the dataset used to train the models described in the paper. This dataset can be downloaded [here](https://www.deepsig.ai/datasets). Note that a verified email is required to download the dataset. 

## The Paper

The paper [Over-the-Air Deep Learning Based Radio Signal Classification](https://ieeexplore.ieee.org/document/8267032) written by Tim O'Shea, Tamoghna Roy, and T. Charles Clancy, does an excellent job exploring the impact of deep learning model hyper parameters on classifying signal modulation types. 

## The Dataset

The dataset provided is stored in a hdf5 format, which contains three arrays of data. These arrays are given the labels "X", "Y", and "Z".

There is a total of 2,555,904 samples and 24 classes, with 106,496 samples per class. 

### I/Q Signal Recordings

The "X" labeled dataset contains the signal recordings. Each element has shape (1024, 2), where axis 0 is the time domain, and axis 1 has the real and imaginary value. 

### Label

The "Y" labeled dataset contains a one-hot encoding of the label for the corresponding indexed recording in the "X" dataset. There are 24 possible labels, which are given in classes.txt. 

```
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
```

### SNR

The "Z" labeled dataset contains the Signal to Noise Ratio (SNR) for the corresponding indexed recording in the "X" dataset. These values range from -20 to 30 in two step intervals. 

## Recreating the model

The goal of this project is to recreate and further build on the deep neural network models that the paper establishes. These models can then be used as a foundation to build more complex radio frequency classification models by either adapting and fine tuning current models, or using the general structure as a starting point. 

### Data Input

There are two different ways to interpret Radio Frequency (RF) signal data.

The first is in the time domain, where the amplitude of the received signal is recorded over time. This format is often recorded as an I/Q modulated signal, where the values are complex numbers containing a real and imaginary value. 

The second way is using the frequency domain. Time domain data can be converted to frequency domain by taking a [Fourier Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) of the dataset. In the frequency domain, it is often easier to see the channels that devices communicate on, and visualize their shape. 

Because it is typical to analyze signals in the frequency domain, my main goal in furthering the papers exploration is to attempt to build a classification model using frequency domain data as input. 

### Project Goal

The goal of this project is to recreate a Convolutional Neural Network (CNN) from scratch that can match the accuracy of the models created by the researchers in the paper. 

It was shown in the paper that the SNR of the signal drastically impacts the models ability to correctly classify the signal type. Due to this, I have selected a slice of the dataset that only uses SNR >= 0 and SNR >= 10. Any input lower than this becomes very hard to distinguish from noise. 

### Results

This project focuses on modifying the model structure and its training hyper-parameters, including:
- Number of layers in the model
- Kernel size of the Conv layer
- Number of Conv layers per stack
- Including Batch Normalization or Dropout
- Structure of the Dense layers
- Optimizer used (Adam, SGD, RMSprop, etc.)
- Optimizer momentum
- Learning rate

Despite multiple attempts to build a good frequency domain classification model, the best model I was able to create stalled out at about 45% test accuracy. This is not anywhere near the 90%+ results of the paper. 

It turns out that for this data and model type, it is much easier for the CNN model to use the IQ time domain signal as input. Whether it be because this format preserves the phase, or if it has more distinctive features, it performed better. 

After trying different model hyper-parameters, my best model was able to achieve 95.65% classification accuracy, meeting the goal of this project. 

The best accuracy achieved by a model can likely be raised by a percent or two by fine tuning the training, normalization, and adding data regularization. 

From this project, I was able to experiment and learn enough to have a solid foundation of building large models from scratch for large datasets. 