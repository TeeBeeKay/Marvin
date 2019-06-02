import numpy as np
import random
import os
import wave
import keras.layers
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.utils import to_categorical


command_list = os.listdir('commands/')

vocab = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 
    'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ]


# convnet definition
def create_01_model(shape = [32,32,256], num_classes = 38):
    input_shape = (None, 50, 1) # time, frequency bins, channels
    net = Sequential()
    net.add(keras.layers.BatchNormalization(input_shape = input_shape))
    net.add(keras.layers.GaussianNoise(0.1))
    net.add(Conv2D(filters=shape[0], kernel_size=(20,8), strides=1, activation='relu'))
    net.add(keras.layers.Dropout(0.1))
    net.add(MaxPool2D())
    net.add(Conv2D(filters=shape[1], kernel_size=(10,4), strides=1, activation='relu'))
    net.add(keras.layers.Dropout(0.1))
    net.add(MaxPool2D())
    net.add(Conv2D(filters=shape[2], kernel_size=(10,4), strides=1, activation='relu'))
    net.add(keras.layers.Dropout(0.1))
    net.add(MaxPool2D())
    net.add(keras.layers.GlobalMaxPooling2D())
    net.add(keras.layers.Dropout(0.4))
    net.add(Dense(num_classes, activation='softmax'))
    opt = keras.optimizers.Adam(lr=0.001)
    net.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])
    return net



def load_audio_data(w):
    # takes an open wav file, w, and returns spectrogram
    byte_data = w.readframes(w.getnframes())
    chunk_data = np.frombuffer(byte_data, dtype='<i2')
    # truncate to nearest 10ms and then separate into 10ms chunks
    chunk_data = chunk_data[0:chunk_data.shape[0]//160*160].reshape((-1, 160))
    power = np.abs(np.fft.rfft(chunk_data)) ** 2
    return power[::,0:50]


def import_dataset(split = 0.8, dataset_path = 'dataset\\', length = 120):
    words = vocab
    x = np.zeros((100000,length,50))
    y = []
    extra_x, extra_y = import_commands()
    extra_y = extra_y.argmax(1) + len(words) - 1
    index = 0
    indices = []
    for i in range(0, len(words)):
        print("Importing: ", i, words[i])
        data = np.load(dataset_path + words[i] + '.npy')
        new_index = index + data.shape[0]
        if data.shape[1] <= length:
            x[index:new_index, 0:data.shape[1]] = data
        else:
            x[index:new_index] = data[:,0:length]
        index = new_index
        y.extend([float(i)] * data.shape[0])
        indices.append(index)
    x[index:index+extra_x.shape[0]] = extra_x[:,0:length,:,0]
    y.extend(extra_y)
    y = to_categorical(np.array(y), num_classes=extra_y.max()+1)
    x = x[0:y.shape[0]]
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1)) # add extra channels dimension
    ## Shuffle data
    randstate = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(randstate)
    np.random.shuffle(y)
    split = int(x.shape[0] * split)
    train_x = x[0:split]
    train_y = y[0:split]
    valid_x = x[split:-1]
    valid_y = y[split:-1]
    return train_x, train_y, valid_x, valid_y


def import_commands(dataset_path = 'commands/'):
    length = 300 # pad to 3 seconds
    command_list = os.listdir(dataset_path)
    command_index = 0
    i = 0
    num_samples = 2000
    x = np.zeros((num_samples, length, 50, 1))
    y = np.zeros((num_samples, len(command_list)))
    for command in command_list:
        if command not in vocab:
            files = [dataset_path + command + '/' + filename for filename in os.listdir(dataset_path + command)]
            for filepath in files:
                with wave.open(filepath) as w:
                    y[i, command_index] = 1
                    data = load_audio_data(w)
                    data = data.reshape(data.shape[0], data.shape[1], 1)
                    if data.shape[0] == length:
                        x[i] = data
                    elif data.shape[0] > length:
                        x[i] = data[0:length] # truncate to fit in array
                    elif data.shape[0] < length:
                        x[i, 0:data.shape[0]] = data # zero pad the end
                    i += 1
            command_index += 1
    x = x[0:i]
    y = y[0:i]
    return x,y




train_x, train_y, valid_x, valid_y = import_dataset(length=120)
#net = create_01_model(num_classes=train_y.shape[1])
net = create_01_model(shape=[32,32,128], num_classes=train_y.shape[1])

#normalise
for i in range(len(train_x)):
    train_x[i] = train_x[i]/train_x[i].max()

for i in range(len(valid_x)):
    valid_x[i] = valid_x[i]/valid_x[i].max()


batch_size = 600
epochs = 100
cw = 1/train_y.sum(0)

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
             keras.callbacks.ModelCheckpoint(filepath=str(time.time()).split('.')[0]+'.h5', monitor='val_loss', save_best_only=True)]

net.fit(train_x, train_y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_x, valid_y),
    class_weight=cw,
    callbacks=callbacks)