import os
import json
import wave
import numpy as np

def preprocess_audio(audio_data):
    # eventually this should compute MFCC
    # but initially fft should do

    # Breaking compatibility with model v2 and v3 by changing 500 to 100
    # chunk_data = audio_data.reshape((500,-1))
    chunk_data = audio_data.reshape((-1, 441))
    power = np.abs(np.fft.rfft(chunk_data)) ** 2
    return power


def generate_data():

    # produce a list of all files
    positive_files = ['data/positive/' + file for file in os.listdir('data/positive')]
    negative_files = ['data/negative/' + file for file in os.listdir('data/negative')]
    all_files = positive_files + negative_files
    X = [] # input data
    Y = [] # output data

    for file in all_files:
        
        with wave.open(file, 'rb') as w:
            byte_data = w.readframes(w.getnframes())
            arr_data = np.frombuffer(byte_data, dtype='<i2')
            processed_data = preprocess_audio(arr_data)
            X.append(processed_data[::,0:50])
            Y.append(np.zeros(1))
            
    for i in range(len(positive_files)):
        Y[i][0] = 1

    # Reorganise to shape: time, batch, values. Currently has the shape: batch, time, values
    X = np.array(X)
    X = np.swapaxes(X, 0, 1)
    Y = np.array(Y)
    Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
    Y = np.swapaxes(Y, 0, 1)

    return X, Y

def generate_from_file_list(files):
    X = []
    for file in files:
        with wave.open(file, 'rb') as w:
            byte_data = w.readframes(w.getnframes())
            arr_data = np.frombuffer(byte_data, dtype='<i2')
            processed_data = preprocess_audio(arr_data)
            X.append(processed_data[::,0:50])
    X = np.array(X)
    X = np.swapaxes(X, 0, 1)
    return X

def load_from_json(json_file_name):
    with open(json_file_name) as json_file:
        json_data = json.load(json_file)
        # Find directory by looking for last occurence of "/"
        prefix = json_file_name[0:json_file_name.rfind('/') + 1]
        # Look through all entries and find those which contain the keyword
        for entry in json_data:
            if "marvin" in entry["Labels"]:
                entry["Keyword"] = True
            else:
                entry["Keyword"] = False
        
        # Load wav data -- could just optimise by combining above but keeping separate for now
        x_bins = {}
        y_bins = {}
        X = []
        Y = []
        for entry in json_data:
            with wave.open(prefix + entry["Filename"], 'rb') as w:
                byte_data = w.readframes(w.getnframes())
                arr_data = np.frombuffer(byte_data, dtype='<i2')
                processed_data = preprocess_audio(arr_data)
                X.append(processed_data[::,0:50])

                # check if there's a bin for data of that length
                data_length = len(processed_data)
                if data_length in x_bins:
                    x_bins[data_length].append(processed_data[::,0:50])
                else:
                    x_bins[data_length] = []
                    y_bins[data_length] = []
                    x_bins[data_length].append(processed_data[::,0:50])

            if entry["Keyword"]:
                Y.append(np.array(1.0))
                y_bins[data_length].append(np.array(1.0))
            else:
                Y.append(np.array(0.0))
                y_bins[data_length].append(np.array(0.0))
        
        # find the maximum length and pad all arrays to this length
        maximum_length = len(max(X, key=len))
        for i in range(len(X)):
            if len(X[i]) < maximum_length:
                difference = maximum_length - len(X[i])
                # zero pad the start of the sequence
                X[i] = np.pad(X[i], ((difference, 0), (0,0)) , 'constant')
        X = np.array(X)
        X = np.swapaxes(X, 0, 1)
        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0], 1))

        for key in x_bins:
            x_bins[key] = np.array(x_bins[key])
            x_bins[key] = np.swapaxes(x_bins[key], 0, 1)
            y_bins[key] = np.array(y_bins[key])
            y_bins[key] = np.reshape(y_bins[key], (y_bins[key].shape[0], 1))
        return x_bins, y_bins

def to_one_hot(input_string):
    # actually returns dense vector. Convert to one hot with tf.one_hot(dense, 26)
    input_string = input_string.lower()
    dense = []
    for character in input_string:
        if ord(character) > 96 and ord(character) < 123:
            dense.append(ord(character) - 97)
        else:
            dense.append(26)
    #pad to 100 characters in length
    if len(dense) < 100:
        dense += [26] * (100 - len(dense))
    return dense
