import numpy as np
import pandas as pd
import torch
import torchaudio
import sounddevice

from scipy.io.wavfile import write
from sklearn.metrics import confusion_matrix
import seaborn as sn

from cnn_roi_orya import ConvNet_roi_orya

SAMPLE_RATE = 16000

device = torch.device("cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
wav_model = bundle.get_model().to(device)

classes = {0: "male", 1: "female"}


def Norm(X):
    embedding = X.detach().cpu().numpy()
    for i in range(len(embedding)):
        mlist = embedding[0][i]
        embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(embedding).to(device)


def recording(name):
    filename = name
    duration = 3
    print("Recording ..")
    recording = sounddevice.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sounddevice.wait()
    print("Done.")
    write(filename + ".wav", SAMPLE_RATE, recording)
    return filename + ".wav"


def inference(file_path):
    waveform, sr = torchaudio.load(file_path, num_frames=SAMPLE_RATE * 3)

    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    waveform = waveform.to(device)

    return waveform


def print_results(y):
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    print(f'Predicted: {classes[max].capitalize()}')
    print(f'male: {round(predict[0][0] * 100, 4)}%')
    print(f'female:  {round(predict[0][1] * 100, 4)}%')
    return classes[max].capitalize()


if __name__ == '__main__':
    gender_model = torch.load("36gender_Model-epoch_36_Weights.pth", map_location=torch.device("cpu"))
    gender_model.eval()

    with torch.inference_mode():
        recording(name="name")
        tor = inference(file_path="../name.wav")
        embedding, _ = wav_model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = gender_model(embedding)
        ans = print_results(y)
        print(ans)