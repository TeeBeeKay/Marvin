import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import collections
import uuid


CHUNK = 100
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 20000
RECORD_SECONDS = 2
LENGTH = RATE * RECORD_SECONDS // CHUNK
THRESHOLD = 7.5e+17


sound = collections.deque(maxlen=LENGTH)
volume = collections.deque(maxlen=LENGTH)
state = 0 # 0: initialising, 1: waiting, 2: recording, 3: flushing



def getSamples(stream, sound, volume, CHUNK):
    data = stream.read(CHUNK)
    sound.append(data)
    # create a list of 4 byte strings representing a signed int32 sample
    samples = [data[x:x+4] for x in range(0, CHUNK * 4, 4)]
    # convert to list of ints
    intsamples = [int.from_bytes(sample, 'little', signed=True) for sample in samples]
    # convert to numpy array
    samplearray = np.array(intsamples)
    power = np.abs(np.fft.rfft(samplearray)) ** 2
    volume.append(power.mean())

def writeSamples(filename, sound, CHANNELS, FORMAT, RATE, p):
    wf = wave.open(filename + '.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(sound))
    wf.close()

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=2048)
    
    # fill buffer
    while len(sound) < LENGTH:
        data = stream.read(CHUNK)
        sound.append(data)
        # create a list of 4 byte strings representing a signed int32 sample
        samples = [data[x:x+4] for x in range(0, CHUNK * 4, 4)]
        # convert to list of ints
        intsamples = [int.from_bytes(sample, 'little', signed=True) for sample in samples]
        # convert to numpy array
        samplearray = np.array(intsamples)
        power = np.abs(np.fft.rfft(samplearray)) ** 2
        volume.append(power.mean())
    state = 1
    bufferstate = 0
    print('Buffer full... listening')
    # main listen loop
    try:
        while True:
            #listen for sound above threshold
            if state == 1:
                getSamples(stream, sound, volume, CHUNK)
                vol = np.array(volume)
                if vol[-50:-1].mean() > THRESHOLD:
                    state = 2
                    print('Recording')
            #wait until sound is centred in recording
            if state == 2:
                getSamples(stream, sound, volume, CHUNK)
                vol = np.array(volume)
                if vol[0:LENGTH//2].mean() > vol[LENGTH//2:LENGTH].mean():
                    filename = str(uuid.uuid4())
                    writeSamples(filename, sound, CHANNELS, FORMAT, RATE, p)
                    state = 3
                    print("Saved as " + filename)
                    print("Flushing")
            #flush the buffer with silence
            if state == 3:
                getSamples(stream, sound, volume, CHUNK)
                if volume[-1] > THRESHOLD:
                    bufferstate = 0
                else:
                    bufferstate += 1
                if bufferstate > LENGTH:
                    bufferstate = 0
                    state = 1
                    print('Listening')
    except KeyboardInterrupt:
        pass
    stream.stop_stream()
    stream.close()
    p.terminate()


record()
