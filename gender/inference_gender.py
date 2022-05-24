import numpy as np
import pandas as pd
import torch
import torchaudio
import sounddevice

from scipy.io.wavfile import write
from sklearn.metrics import confusion_matrix
import seaborn as sn

from model_definition_gender import ConvNet_roi_orya

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
    # print(f'Predicted: {classes[max].capitalize()}')
    # print(f'male: {round(predict[0][0] * 100, 4)}%')
    # print(f'female:  {round(predict[0][1] * 100, 4)}%')
    return classes[max].capitalize()


def testing(name):
    with torch.inference_mode():
        # recording(name="name")
        tor = inference(file_path=f"../{name}.wav")
        embedding, _ = wav_model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = gender_model(embedding)
        ans = print_results(y)
        print(f"{name} is {ans}")


if __name__ == '__main__':
    gender_model = torch.load("Final_gender_model.pth", map_location=torch.device("cpu"))
    gender_model.eval()

    testing("negative_01")
    testing("negative_02")
    testing("negative_03")
    testing("negative_04")
    testing("negative_05")
    testing("negative_06")
    testing("negative_07")
    print(" ")
    testing("neutral_01")
    testing("neutral_02")
    testing("neutral_03")
    testing("neutral_04")
    testing("neutral_05")
    testing("neutral_06")
    testing("neutral_07")
    print(" ")
    testing("positive_01")
    testing("positive_02")
    testing("positive_03")
    testing("positive_04")
    testing("positive_05")
    testing("positive_06")
    testing("positive_07")
    print(" ")
    testing("Chinese_twenties_male_1")
    testing("Chinese_twenties_male_2")
    testing("Chinese_twenties_male_3")
    testing("Chinese_twenties_male_4")
    testing("Chinese_twenties_male_5")
    print(" ")
    testing("eng_female")
    testing("eng_male (1)")
    testing("ENG_MALE")
    testing("English_fifties_female_1")
    testing("English_thirties_male_2")
    testing("English_nineties_male_3")
    testing("English_thirties_male_4")
    testing("English_twenties_male_5")
    print(" ")
    testing("french_female")
    testing("french_male")
    testing("ENG_MALE")
    print(" ")
    testing("Persian_thirties_male_2")
    testing("Persian_thirties_male_4")
    print(" ")
    testing("Portuguese_thirties_male_1")
    testing("Portuguese_thirties_male_2")
    testing("Portuguese_thirties_male_4")
    print(" ")
    testing("Romansh_teens_male_1")
    print(" ")
    testing("Rwanda_twenties_male_2")
    testing("Rwanda_twenties_male_3")
    testing("Rwanda_twenties_male_5")
    print(" ")
    testing("Adult_Male_1")
    testing("Adult_Male_3")
    testing("Adult_Male_4")
    testing("Adult_Male_5")
    testing("Adult_Male_12")
    testing("Adult_Female_6")
    testing("Adult_Female_7")
    testing("Adult_Female_8")
    testing("Adult_Female_9")
    testing("Adult_Female_10")
    testing("Adult_Female_11")
    print(" ")
    testing("teen1")
    testing("teen2")
    testing("teen3")
    testing("teen4")
    testing("teen5")
    testing("teen6")
    testing("teen7")
    testing("teen8")
    testing("teen9")
    testing("teen10")
    testing("teen11")
    testing("teen12")
    testing("teen13")
    testing("teen14")
