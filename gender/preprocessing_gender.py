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
        for file in glob.glob("..\\data\\all-voice-common.pkl"):
            fname = os.path.basename(file)
            path = "..\\data\\" + fname
            with open(path, 'rb') as f:
                pack = pickle.load(f)
            for tens in pack:
                X.append(tens)
            print("finished pkl file number " + str(k) + " - " + fname)
            k += 1
        d = X

        # path = f'data\\all-data-vox1.pkl'
        # with open("data/all-data-dev-vox1-40-5-Orya.pkl", 'wb') as f:
        #     pickle.dump(X, f)

        X_data = []
        y_gender = []  # gender
        self.ages = {"teens": 0, "twenties": 1, "thirties": 2, "fourties": 3}
        self.ignore_age = {"fifties": 4, "sixties": 5, "seventies": 6, "eighties": 7, "nineties": 8}
        self.genders = {"male": 0, "female": 1}

        gender_sum = [0, 0]
        age_sum = [0, 0, 0, 0]
        g, a, t = 0, 0, 0
        female_sum, male_sum = 0, 0
        for i in X:  # iterating over the lines of the data
            for j in i:  # iterating over the set in each line
                if j in self.genders:
                    g = j
                elif j in self.ages or j in self.ignore_age:
                    a = j
                else:  # tensor
                    j = torch.nn.functional.normalize(j, p=10.0, dim=1)
                    t = j

            l = 1

            if g == "male":
                male_sum += 1

            if g == "female":
                female_sum += 1

            # if male_sum > 60400 and g == "male":  # 14700
            #     l = 0

            # if female_sum > 1500 and g == "female":
            #     l = 0

            # if a in self.ignore_age:
            #     l = 0

            for i in range(l):  # if l not 0:
                y_gender.append(self.genders[g])
                gender_sum[self.genders[g]] += 1
                X_data.append(t)

        print(male_sum)
        print(female_sum)
        print("gender_sum is ", gender_sum)
        print(sum(gender_sum))
        print("\n")

        self.y = np.array(y_gender)  # change this line and below for other label
        X_train, X_test, y_train, y_test = train_test_split(np.array(X_data), np.array(y_gender), test_size=0.20)
        X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size=0.15)

        print("X_train size -", len(X_train))
        print("X_test size -", len(X_test))
        print("X_val size -", len(X_val))
        print("y_train size -", len(y_train))
        print("y_test size -", len(y_test))
        print("y_val size -", len(y_val))

        train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        self.train_x = torch.from_numpy(X_train)
        self.train_y = torch.from_numpy(y_train)

        self.test_x = torch.from_numpy(X_test)
        self.test_y = torch.from_numpy(y_test)

        self.train_loader = DataLoader(train, batch_size=50, shuffle=True)
        # and provides an iterable over the given dataset.
        self.test_loader = DataLoader(test, batch_size=50, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=50, shuffle=True)
