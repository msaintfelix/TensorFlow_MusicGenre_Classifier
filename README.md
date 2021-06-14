# TensorFlow_MusicGenre_Classifier
Classifying wav files using their MFCC.

# Objective

The objective of this project is to classify 30 sec audio files by genre using a TensorFLow MLP model. To classify these audio samples in .wav format, we will preprocess them by calculating their MFCC, which is a temporal representation of the energy variations for each perceived frequency band. In this case, we are choosing 13 bands.

# Environment and tools

The GTZAN dataset can be found here:
https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

This code is a standalone Jupyter Notebook in a Conda-Python3 environment using Numpy, Scikit-Learn, Matplotlib and TensorFlow Keras.

# Project steps

	1 - Inspect and preprocess data
	
The dataset consists of 30 second songs in .wav format (999 total), divided into 10 folders, each corresponding to a musical genre. Firstly, we will label each genre using its folder index and split each 30 second sample in the desired amount of slices. Then we will calculate the MFCC (Mel-Frequency Cepstrum Coefficients) for each slice and store it with the associated label in a json file.

In sound processing, the mel-frequency cepstrum is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.

All audio files are clean (no noise), no extra preprocessing is required. 

	2 - Train and predict
	
We are using a MLP model with a Flatten() layer whose input shape matches the number of MFCC bands and time frames. The other layers are Dense() fully connected with a 10 output softmax Dense() as the final layer (for all 10 classes/genres).

	3 - Model performance
	
We are using Matplotlib with 'accuracy' metric to plot the training and validation curves after 50 epochs and an RMSprop optimizer with a learning rate lr=0.001.

Tweaking various hyperparameters led to a final 70% on the training set and 60% on the validation set. Not bad, but more features are probably needed to reach better accuracy given the limited number of training examples. More slices result in shorter samples and affects performance. 

A more complex MLP architecture did not improve the final results and slowed down the training process.
