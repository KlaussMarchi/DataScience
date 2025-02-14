import pyaudio
import numpy as np
import tensorflow as tf
from time import sleep

commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
model    = tf.keras.models.load_model('model.keras')

FORMAT   = pyaudio.paInt16
CHANNELS = 1

FRAMES_PER_BUFFER = 3000 # janela de captura (1000 a 16000)
RATE = 16000 # Taxa de amostragem, 1 segundo de áudio
WAVE_LENGTH = 16000

p = pyaudio.PyAudio()
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def getSpectrogram(waveform):
    input_len = WAVE_LENGTH # TAMANHO DO AUDIO
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    # aumente frame_length e frame_step para precisão
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def processAudio(bufferAudio):
    waveform = tf.convert_to_tensor(bufferAudio / 32768, dtype=tf.float32)
    spectrogram  = getSpectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, 0)
    return spectrogram


def predictAudio(audio_buffer):
    audio = processAudio(audio_buffer)
    prediction = model(audio)
    probabilities = tf.nn.softmax(prediction[0]).numpy()
    label_pred = np.argmax(probabilities)
    command = commands[label_pred]
    probability = probabilities[label_pred]
    return command, probability


def handleMicrofone():
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    bufferData = np.zeros(RATE, dtype=np.int16)
    
    while True:
        newData = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                            
        # STREAM TO BUFFER
        newAudio = np.frombuffer(newData, dtype=np.int16)
        newSize  = newAudio.shape[0]
        
        # SHIFT BUFFER CONTINUESLY
        bufferData = np.roll(bufferData, -newSize)
        bufferData[-newSize:] = newAudio
        
        command, prob = predictAudio(bufferData)
        
        if prob > 0.7: 
            print(f'{command}: {prob:.2f}')
            bufferData = np.zeros(RATE, dtype=np.int16)
            sleep(2)


handleMicrofone()
