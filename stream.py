import numpy as np
import pyaudio
import os
import wave
from phue import Bridge
from collections import deque
import sys

import keras
from keras.layers import Dense


command_path = 'commands/'
if sys.platform == 'linux':
    command_path = '/media/usb/commands'

def load_audio_data(w):
    # takes an open wav file, w, and returns spectrogram
    byte_data = w.readframes(w.getnframes())
    chunk_data = np.frombuffer(byte_data, dtype='<i2')
    # truncate to nearest 10ms and then separate into 10ms chunks
    chunk_data = chunk_data[0:chunk_data.shape[0]//160*160].reshape((-1, 160))
    power = np.abs(np.fft.rfft(chunk_data)) ** 2
    return power[::,0:50]


def preprocess(data, chunk_size):
    samples = [data[x:x+2] for x in range(0, chunk_size * 2, 2)]
    intsamples = [int.from_bytes(sample, 'little', signed=True) for sample in samples]
    chunk = np.array(intsamples).reshape((-1, 160))
    chunk = np.abs(np.fft.rfft(chunk)) ** 2
    chunk = chunk[::,0:50].reshape(1, -1, 50, 1)
    return chunk


def stream(window=100, stride = 50):
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
    return x,y, command_list


x, y, command_list = import_commands(command_path)

net = keras.models.load_model('1550788585.h5')
encoder = keras.models.Model(inputs = net.inputs, outputs = net.layers[-3].output)
encoded_data = encoder.predict(x).T # transpose the axes
encoded_data = encoded_data / np.linalg.norm(encoded_data + 1e-9, axis=0) # normalise
labels = y.argmax(1)

# cosine similarity network
nearest_neighbour = Dense(units = labels.shape[0], activation='linear', use_bias=False)(encoder.output)
classifier = keras.models.Model(inputs=encoder.input, outputs=nearest_neighbour)
weights = classifier.get_weights()
weights[-1] = encoded_data
classifier.set_weights(weights)

b = Bridge('192.168.0.12')

print('Beginning listening...')
stream()