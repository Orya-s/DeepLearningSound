import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000

classes = {0: "teen", 1: "adult"}


def Norm(X):
    em = X.detach().cpu().numpy()
    for i in range(len(em)):
        mlist = em[0][i]
        em[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(em)


def inf(em, age_model):
    em = em.unsqueeze(0)
    em = Norm(em)
    # print(em.shape)
    ans = age_model(em).detach().numpy()
    predict = [np.exp(c) for c in ans]
    return classes[np.argmax(predict)].capitalize()


if __name__ == '__main__':
    model = torch.load("binary_models_age\\1.5\\CoVoVox_lr0.0001_NM\\age_Binary_Model-e_11_Weights.pth")
    model.eval()
    wav_path = 'audio\\teen1.wav'
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform[0]
    waveform = torch.tensor(np.array([waveform]))

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    device = torch.device("cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav_model = bundle.get_model().to(device)
    embedding, _ = wav_model(waveform)

    with torch.inference_mode():
        print(inf(embedding, model))
        # embedding = wav file (1 channel) after Wav2Vec
        # model = the age model, already loaded
