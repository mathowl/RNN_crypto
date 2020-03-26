# RNN_crypto
Keras recurrent neural network for crypto prices based on an example from sentdex 

## Data
The crypto-prices come from a python tutorial from sentdex.

## Data Preprocessing
Unpack the rar and run the crypto_preprocessing.ipynb. This gives a shelve with the necessary data for the neural net.

## Training the neural net
Add filepaths for the models and tensorboard logs in RNN-Arch_opt.py. The architecture is Sequential of the form LSTM -> Dropout -> BatchNorm -> Dense. 
The RNN-Arch_opt.py varies over LSTM layers, dropout values and dense layers. Optimization results can be viewed using tensorboard. Epoch accuracy reaches 0.59 for LSTM-3-dense-1-dropout-0.3
