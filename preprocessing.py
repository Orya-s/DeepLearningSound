import datasets
import torch
import requests
import IPython
# import torchaudio

import numpy as np
import glob
import pickle
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class prepareData:
    def __init__(self):
        X = []
        k = 1
        for file in glob.glob("data\\*.pkl"):
            fname = os.path.basename(file)
            path = "data\\" + fname
            pack = []
            with open(path, 'rb') as f:
                pack = pickle.load(f)
            for tens in pack:
                X.append(tens)
            print("finished pkl file number " + str(k))
            k += 1
        d = X

        X_data = []
        y1_data = []  # gender
        Y_data = []  # age
        ages = {"teens": 1, "twenties": 2, "thirties": 3, "fourties": 4, "fifties": 5, "sixties": 6,
                "seventies": 7, "eighties": 8, "nineties": 9}
        genders = {"male": 0, "female": 1}

        for i in X:  # iterating over the lines of the data
            for j in i:  # iterating over the set in each line
                if j in genders:
                    y1_data.append(genders[j])
                elif j in ages:
                    Y_data.append(ages[j])
                else:  # tensor
                    j = torch.nn.functional.normalize(j, p=10.0, dim=1)
                    X_data.append(j)

        print("\n")
        X_train, X_test, y_train, y_test = train_test_split(np.array(X_data), np.array(Y_data), test_size=0.20)
        # X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size=0.125)

        print("X_train size -", len(X_train))
        print("X_test size -", len(X_test))
        # print("X_val size -", len(X_val))
        print("y_train size -", len(y_train))
        print("y_test size -", len(y_test))
        # print("y_val size -", len(y_val))

        train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        # val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        self.train_loader = DataLoader(train, batch_size=50, shuffle=True)  # Combines a dataset and a sampler,
            # and provides an iterable over the given dataset.
        self.test_loader = DataLoader(test, batch_size=50, shuffle=True)  # , drop_last=True)
        # self.val_loader = DataLoader(val, batch_size=50)  # , drop_last=True)




        # path = "data.pkl"
        # with open(path, 'wb') as f:
        #     pickle.dump(d, f)
