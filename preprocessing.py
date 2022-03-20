import datasets
import torch
import requests
import IPython
import torchaudio
import numpy as np
import glob
import pickle
import os

if __name__ == '__main__':
    X = []
    Y = []
    k = 1
    for file in glob.glob("C:\\Users\\תמה וגלעד\\Documents\\Gilad\\לימודים\\שנה ג'\\א'\\למידת מכונה\\LanguageRecognition\\data\\*.pkl"):
        fname = os.path.basename(file)
        tag = fname.split('_')[0]
        path = "data\\" + fname
        pack = []
        with open(path, 'rb') as f:
            pack = pickle.load(f)
        for tens in pack:
            X.append(tens)
            Y.append(tag)
        print("finished pkl file number " + str(k))
        k += 1

    d = [X,Y]

    path = "C:\\Users\\תמה וגלעד\\Documents\\Gilad\\לימודים\\שנה ג'\\א'\\למידת מכונה\\LanguageRecognition\\data.pkl"
    with open(path, 'wb') as f:
        pickle.dump(d, f)