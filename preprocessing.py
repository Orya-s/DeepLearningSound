import datasets
import torch
import requests
import IPython
# import torchaudio

import numpy as np
import glob
import pickle
import os

if __name__ == '__main__':
    X = []
    #Y = []
    k = 1
    for file in glob.glob("data\\*.pkl"):
        fname = os.path.basename(file)
        #print(fname)
        tag = fname.split('_')[0]
        #print(tag)
        path = "data\\" + fname
        #print(path)
        pack = []
        with open(path, 'rb') as f:
            pack = pickle.load(f)
        for tens in pack:
            X.append(tens)
            #Y.append(tag)
        print("finished pkl file number " + str(k))
        k += 1

    d = X
    #d = [X, Y]

    path = "data.pkl"
    with open(path, 'wb') as f:
        pickle.dump(d, f)
