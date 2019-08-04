import numpy as np
import pyaudio
import os
import wave
from phue import Bridge
from collections import deque
import sys
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import time

import keras
from keras.layers import Dense


command_path = 'commands/'
if sys.platform == 'linux':
    command_path = '/media/usb/commands/'

def load_audio_data(w):
    # takes an open wav file, w, and returns spectrogram
    byte_data = w.readframes(w.getnframes())
    data = np.frombuffer(byte_data, dtype='<i2')
    return mfcc(data)


def preprocess(data, chunk_size):
    samples = [data[x:x+2] for x in range(0, chunk_size * 2, 2)]
    intsamples = [int.from_bytes(sample, 'little', signed=True) for sample in samples]
    chunk = np.array(intsamples).reshape((-1, 160))
    chunk = np.abs(np.fft.rfft(chunk)) ** 2
    chunk = chunk[::,0:50].reshape(1, -1, 50, 1)
    return chunk

audio = deque(maxlen=16000)
def stream(window=1, stride = 50):
    psession = pyaudio.PyAudio()
    stream = psession.open(format = pyaudio.paInt16,
                            channels = 1,
                            rate = 16000,
                            input = True,
                            frames_per_buffer = 16000
    )
    chunk_size = stride * 160                       # 160 represents 10ms
    audio_data = stream.read(16000)            # fill audio buffer
    audio_array = np.fromstring(audio_data, np.int16)
    audio.extend(audio_array)
    while True:                                     # loop forever, grabbing chunks, analyzing, printing output
        audio_data = stream.read(chunk_size)
        audio_array = np.fromstring(audio_data, np.int16)
        audio.extend(audio_array)
        data = mfcc(np.array(audio))
        data = np.reshape(data, (1, data.shape[0], data.shape[1], 1))
        result = encoder.predict(data)[0]
        distance = euclidean(result, centroid)
        origin = np.zeros((1, result.shape[0]))
        dto = euclidean(result, origin)
        normalised_distance = distance/cto
        print(np.around(normalised_distance, decimals=2))
        score = normalised_distance - dtc
        if(score.min() < 0):
            print(command_list[score.argmin()])
        """
        distance = euclidean(result, encoded_data)
        top_results = distance.argsort()
        print(labels[top_results[0:5]])
        class_distance = np.zeros(len(command_list))
        for j in range(len(command_list)):
            class_distance[j] = np.mean(distance[labels == j])
        if class_distance.min() < 30:
            print(class_distance)
            print(command_list[class_distance.argmin()])
            if class_distance.argmin() == 0:
                b.lights[0].on = False
            if class_distance.argmin() == 1:
                b.lights[0].on = True
        else:
            print(class_distance)
        """


def import_commands(dataset_path = 'commands/'):
    length = 100 # pad to 3 seconds
    command_list = os.listdir(dataset_path)
    command_index = 0
    i = 0
    num_samples = 2000
    x = np.zeros((num_samples, length, 13, 1))
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
    return x,y, command_list


x, y, command_list = import_commands(command_path)

#net = keras.models.load_model('1550788585.h5')
net = keras.models.load_model('1559757989.h5')
encoder = keras.models.Model(inputs = net.inputs, outputs = net.layers[-3].output)
encoded_data = encoder.predict(x)
labels = y.argmax(1)

# euclidean distance
def euclidean(query, dataset):
    distance = np.sqrt((dataset - query)**2)
    return distance.sum(1)

# find the centroid of each class
centroid = np.zeros((len(command_list), encoded_data.shape[1]))
for i in range(centroid.shape[0]):
    centroid[i] = np.mean(encoded_data[labels == i], axis=0)

# find the distance to origin for each centroid
cto = euclidean(np.zeros(centroid.shape[1]), centroid)

# find the average distance to centroid for each class
dtc = np.zeros(len(command_list))
for i in range(dtc.shape[0]):
    dtc[i] = np.mean(euclidean(centroid[i], encoded_data[labels == i]), axis=0)
dtc /= cto

b = Bridge('192.168.0.12')

print('Beginning listening...')
stream()