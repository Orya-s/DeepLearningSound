import numpy as np
import torch
import torchaudio
import sounddevice
from scipy.io.wavfile import write


def record(filename):
    sr = 16000
    sec = 10
    print("Recording")
    rec = sounddevice.rec(int(sec * sr), samplerate=sr, channels=1)
    sounddevice.wait()
    write(filename + ".wav", sr, rec)


# loading the model
age_model = torch.load("models\\5.4\\age_CNN_Model-epoch_10_Weights.pth")
gender_model = torch.load("models\\5.4\\gender_CNN_Model-epoch_10_Weights.pth")
roi_model = gender_model
roi_model.eval()

# recording and saving the audio
record("test_roi")

# loading the audio
speech_array, sampling_rate = torchaudio.load("test_roi.wav", normalize=True)
transform = torchaudio.transforms.Resample(sampling_rate, 16_000)
speech_array = transform(speech_array)
array = speech_array[0]
print(array)
print("------")

# removing empty frames from the beginning of the recording
print("arr before cut ", array, "\n len-", len(array))
array = array[100:]
print("arr after cut ", array, "\n len-", len(array))
print("-----")

# model definition - num of labels
sum_op = 2  # change between models
n_class_samples = [0 for i in range(sum_op)]
sum = [0, 0]  # , 0, 0, 0]  # change between models

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
WAV2VEC2_model = bundle.get_model()

# defining 3 seconds per recording
window = 48000
step = 32000
pointer = 0
n_samples = 0

while pointer + window <= len(array):
    curr = array[pointer:pointer + window]
    tensor_data = torch.tensor(np.array([curr]))

    with torch.inference_mode():
        features, _ = WAV2VEC2_model(tensor_data)
        features = features.unsqueeze(0)

        with torch.no_grad():
            outputs = roi_model(features)

            # euler fix
            p = torch.exp(outputs)
            print(p)
            _, predicted = torch.max(p, 1)
            pred = predicted[0].item()
            print(pred)

            n_samples += 1
            sum[pred] += 1

    pointer += step

print("\n", sum)