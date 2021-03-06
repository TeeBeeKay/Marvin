import numpy as np
import random
import import_data
import pyaudio
import os
import wave
import json
import csv

from phue import Bridge
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D
import keras.layers
from keras.utils import to_categorical
from keras import backend as K

from collections import deque



################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################
command_list = os.listdir('commands/')

vocab = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 
    'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ]


# convnet definition
def create_01_model(shape = [64,64,256]):
    input_shape = (None, 50, 1) # time, frequency bins, channels
    num_classes = len(vocab)
    net = Sequential()
    net.add(keras.layers.BatchNormalization(input_shape = input_shape))
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


################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

def load_audio_data(w):
    # takes an open wav file, w, and returns spectrogram
    byte_data = w.readframes(w.getnframes())
    chunk_data = np.frombuffer(byte_data, dtype='<i2')
    # truncate to nearest 10ms and then separate into 10ms chunks
    chunk_data = chunk_data[0:chunk_data.shape[0]//160*160].reshape((-1, 160))
    power = np.abs(np.fft.rfft(chunk_data)) ** 2
    return power[::,0:50]

def save_dataset(num_samples=100):
    print("importing dataset")
    dataset_path = "data_speech_commands_v0.02\\"
    vocab = [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
        'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 
        'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 
        'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
        ]
    for i in range(len(vocab)):
        print("importing %d of %d", i, len(vocab))
        path = dataset_path + vocab[i] + '\\' # folder path for each word
        files = [path + filename for filename in os.listdir(path)] # list of all filenames for each word
        print("Importing %s: %d files", vocab[i], len(files))
        data = []
        for filename in files:
            with wave.open(filename, 'rb') as w:
                if w.getnframes() == 16000: # only import files of 1 second in length
                    byte_data = w.readframes(w.getnframes())
                    chunk_data = np.frombuffer(byte_data, dtype='<i2').reshape((-1, 160))
                    power = np.abs(np.fft.rfft(chunk_data)) ** 2
                    data.append(power[::,0:50])
        savepath = "dataset\\" + vocab[i]
        np.save(savepath, np.array(data), allow_pickle=False)

def import_dataset(split = 0.8, dataset_path = 'dataset\\', shuffle = True, length = 100, extra_words = []):
    words = vocab + extra_words
    x = np.zeros((100000,length,50))
    y = []
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
    y = to_categorical(np.array(y), num_classes=len(words))
    x = x[0:y.shape[0]]
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1)) # add extra channels dimension
    ## Shuffle data
    print(x.shape[0], y.shape[0])
    if shuffle:
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
    else:
        return x, y, indices

def import_commands(dataset_path = 'commands/'):
    length = 300 # pad to 3 seconds
    command_list = os.listdir(dataset_path)
    command_index = 0
    i = 0
    num_samples = 2000
    x = np.zeros((num_samples, length, 50, 1))
    y = np.zeros((num_samples, len(command_list)))
    for command in command_list:
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


"""
# import data and process it
train_x, train_y, valid_x, valid_y = import_dataset(length=150, extra_words = ['lights', 'weather'])
net = create_01_model()
class_weights = 1/train_y.sum(0)
net.pop()
net.add(Dense(38, activation='softmax'))
net.load_weights('commands2.h5')
#globalpool = keras.layers.GlobalMaxPooling2D()(net.layers[4].output)
#encoder = keras.models.Model(inputs = net.inputs, outputs = globalpool)
encoder = keras.models.Model(inputs = net.inputs, outputs = net.layers[-3].output)
num_comparisons = 1000
encoded_data = encoder.predict(train_x[0:num_comparisons]).T # transpose the axes
encoded_data = encoded_data / np.linalg.norm(encoded_data + 1e-9, axis=0)
labels = train_y[0:num_comparisons].argmax(1)


net = create_01_model()
net.pop()
net.add(Dense(38, activation='softmax'))
net.load_weights('commands2.h5')
x,y = import_commands()
encoder = keras.models.Model(inputs = net.inputs, outputs = net.layers[-3].output)
encoded_data = encoder.predict(x).T # transpose the axes
encoded_data = encoded_data / np.linalg.norm(encoded_data + 1e-9, axis=0)
labels = y.argmax(1)

# cosine similarity network
nearest_neighbour = Dense(units = labels.shape[0], activation='linear', use_bias=False)(encoder.output)
classifier = keras.models.Model(inputs=encoder.input, outputs=nearest_neighbour)
weights = classifier.get_weights()
weights[-1] = encoded_data
classifier.set_weights(weights)

b = Bridge('192.168.0.12')

from matplotlib.mlab import PCA
pca = PCA(encoded_data + 1e-10)
plt.scatter(pca.Y[:,0], pca.Y[:,1], c = labels)


net.fit(train_x, train_y,batch_size=75,epochs=50,verbose=1,validation_data=(valid_x, valid_y),class_weight=cw)
"""

def train(epochs = 100, batch_size = 100, cw = 0):
    if cw == 0:
        net.fit(train_x, train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(valid_x, valid_y))
    else:
        print('using class weights')
        net.fit(train_x, train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(valid_x, valid_y),
            class_weight=cw)

def trainbatch(epochs = 100, batch_size = 100):
    # pretty terrible way to do this
    for i in range(epochs):
        print('Epoch', i)
        net.fit(train_x, train_y,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            validation_data=(valid_x, valid_y))
        net.fit(x, y,
            batch_size=batch_size//3,
            epochs=1,
            verbose=1,
            validation_data=(valid_x, valid_y))

def preprocess(data, chunk_size):
    samples = [data[x:x+2] for x in range(0, chunk_size * 2, 2)]
    intsamples = [int.from_bytes(sample, 'little', signed=True) for sample in samples]
    chunk = np.array(intsamples).reshape((-1, 160))
    chunk = np.abs(np.fft.rfft(chunk)) ** 2
    chunk = chunk[::,0:50].reshape(1, -1, 50, 1)
    return chunk

def record(length = 2):
    psession = pyaudio.PyAudio()
    stream = psession.open(format = pyaudio.paInt16,
                            channels = 1,
                            rate = 16000,
                            input = True,
                            frames_per_buffer = 10000
    )
    chunk_size = length * 16000
    
    data = stream.read(chunk_size)
    chunk = preprocess(data, chunk_size)
    return chunk

def interactive_test():
    chunk = record(2)
    result = classifier.predict(chunk)[0] # predict the distance from reference utterances
    top_samples = result.argsort()[-5:] # sort the distances and select the five best i.e. last five indices in sorted list
    top_labels = [labels[id] for id in top_samples[::-1]] # convert references to vocab labels
    words = [command_list[label] for label in top_labels] # convert labels to words
    distance = np.zeros(len(command_list))
    for i in range(len(command_list)):
        distance[i] = np.mean(result[labels == i])
    #return distance
    return words
    
    result = net.predict(chunk)
    return result
    print(vocab[result.argmax()], result[0, result.argmax()])


def stream():
    psession = pyaudio.PyAudio()
    stream = psession.open(format = pyaudio.paInt16,
                            channels = 1,
                            rate = 16000,
                            input = True,
                            frames_per_buffer = 16000
    )
    window = 100
    stride = 50
    chunk_size = stride * 160                       # 160 represents 10ms
    audio = deque(maxlen=5 * 16000)                 # store 5 seconds of audio
    data = np.zeros((1, window, 50, 1))
    while True:                                     # loop forever, grabbing chunks, analyzing, printing output
        data = np.roll(data, -stride, axis=1)               # push the old data out of the queue
        audio_data = stream.read(chunk_size)
        audio.extend(audio_data)                    # store the audio for saving later
        chunk = preprocess(audio_data, chunk_size)
        data[:,-stride:] = chunk                    # fill the final stride with data
        result = classifier.predict(data)[0]
        distance = np.zeros(len(command_list))
        for j in range(len(command_list)):
            distance[j] = np.mean(result[labels == j])
        if distance.max() > 1:
            print(command_list[distance.argmax()])
            if distance.argmax() == 0:
                b.lights[0].on = False
            if distance.argmax() == 1:
                b.lights[0].on = True
