# TensorFlow_MusicGenre_Classifier
Classifying wav files using their MFCC.

# Objective

The objective of this project is to classify 30 sec audio files by genre using TensorFlow models. To classify these audio samples in .wav format, we will preprocess them by calculating their MFCC, which is a temporal representation of the energy variations for each perceived frequency band. In this case, we are choosing 13 bands.

For a detailed presentation of this project, check out my article published by Towards Data Science:
https://towardsdatascience.com/music-genre-detection-with-deep-learning-cf89e4cb2ecc

# Environment and tools

The GTZAN dataset can be found here:
https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

The first notebook is a standalone Jupyter Notebook in a Conda-Python3 environment using Numpy, Scikit-Learn, Matplotlib and TensorFlow Keras. 
The second notebook is a more modular app structure using a convolutional neural network.
The third notebook features an LSTM model (RNN).

# Project steps

	1 - Inspect and preprocess data
	
The dataset consists of 30 second songs in .wav format (999 total), divided into 10 folders, each corresponding to a musical genre. Firstly, we will label each genre using its folder index and split each 30 second sample in the desired amount of slices. Then we will calculate the MFCC (Mel-Frequency Cepstrum Coefficients) for each slice and store it with the associated label in a json file.

In sound processing, the mel-frequency cepstrum is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.

All audio files are clean (no noise), no extra preprocessing is required. 

	2 - Train and predict
	
In the first notebook, we are using a MLP model with a Flatten() layer whose input shape matches the number of MFCC bands and time frames. The other layers are Dense() fully connected and the final layer is a 10 output softmax Dense() for all 10 classes/genres.

In the second and third notebooks, we are using a CNN and an LSTM with fully connected MLP as final stage.

	3 - Models performance
	
We are using Matplotlib with 'accuracy' metric to plot the training and validation curves after 30 epochs and an RMSprop optimizer with a learning rate lr=0.001.

For the MLP version, tweaking various hyperparameters led to a final 70% on the training set and 60% on the validation set. Not bad, but more features are probably needed to reach better accuracy given the limited number of training examples. More slices result in shorter samples and affects performance. A more complex MLP architecture did not improve the final results and slowed down the training process.

For the CNN version, we get close to 80% on validation data. A make_prediction() method was added for inference purposes.
The LSTM version reaches the same range of performance.

Check out the execution on Kaggle: https://www.kaggle.com/marchenrysaintfelix/music-genre-cnn-classifier-with-75-val-acc
