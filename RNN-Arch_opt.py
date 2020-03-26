import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import shelve

#include a filepath for the models and logs to run the script.


# load data
s_out = shelve.open("RNN_data")
train_X = s_out["train_X"]
validation_X= s_out["validation_X"]
train_y=s_out["train_y"]
validation_y=s_out["validation_y"]

SEQ_LEN=s_out["SEQ_LEN"]
FUTURE_PERIOD_PREDICT=s_out["FUTURE_PERIOD_PREDICT"]
RATIO_TO_PREDICT=s_out["RATIO_TO_PREDICT"]

print(RATIO_TO_PREDICT)

# neural net config
EPOCHS = 15
BATCH_SIZE = 64

LSTM_layers = [2,3,4]
dense_layers = [1,2]
dropout_values = [0.2,0.3]

for LSTM_layer in LSTM_layers:
    for dense_layer in dense_layers:
        for dropout_value in dropout_values:
            NAME=f"LSTM-{LSTM_layer}-dense-{dense_layer}-dropout-{dropout_value}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
            print(NAME)


            MODEL_PATH = r'filepath\{}.model'.format(NAME) #insert a filepath for models
            PROB = r"Crypt-RNN\{}".format(NAME)
            LOGDIR = r'filepath\ML-logs\{}'.format(PROB) #insert a filepath for logs


            
            model = Sequential()
            
            for l in range(0,LSTM_layer-2):
                model.add(LSTM(128,input_shape=(train_X.shape[1:]),return_sequences=True))
                model.add(Dropout(dropout_value))
                model.add(Activation('relu'))
                model.add(BatchNormalization())

            model.add(LSTM(32,input_shape=(train_X.shape[1:])))
            model.add(Dropout(dropout_value))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            for l in range(0,dense_layer-1):
                model.add(Dense(32))
                model.add(Activation('relu'))
                model.add(Dropout(dropout_value))

            model.add(Dense(2))
            model.add(Activation('softmax'))

            #opt=tf.keras.optimizers.Adam(lr=0.001,decay=1e-6)
            model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam', metrics =['accuracy']) #bin_cross loss remains constant



            checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')            
            es=EarlyStopping(monitor='val_loss', mode='min', baseline=0.4, verbose= 1, patience=10)
            tensorboard = TensorBoard(log_dir = LOGDIR)
            callbacks_list=[tensorboard,checkpoint,es]

            model.fit(train_X,train_y, batch_size=BATCH_SIZE, validation_data = (validation_X,validation_y),epochs=EPOCHS,callbacks=callbacks_list)
