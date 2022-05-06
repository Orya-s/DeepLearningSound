import torch
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
        for file in glob.glob("..\\data\\*.pkl"):
            fname = os.path.basename(file)
            path = "..\\data\\" + fname
            with open(path, 'rb') as f:
                pack = pickle.load(f)
            for tens in pack:
                X.append(tens)
            print("finished pkl file number " + str(k) + " - " + fname)
            k += 1

        X_data = []
        y_gender = []  # gender
        y_age = []  # age
        self.ages = {"teens": 0, "twenties": 1, "thirties": 1, "fourties": 1, "fifties": 1, "sixties": 1,
                     "seventies": 1, "eighties": 1, "nineties": 1}
        self.genders = {"male": 0, "female": 1}

        gender_sum = [0, 0]
        age_sum = [0, 0]
        g, a, t = 0, 0, 0
        female_sum, male_sum = 0, 0
        for i in X:  # iterating over the lines of the data
            for j in i:  # iterating over the set in each line
                if j in self.genders:
                    g = j
                elif j in self.ages:
                    a = j
                else:  # tensor
                    t = j  # torch.Size([1, 149, 32])

            if g == "male":
                male_sum += 1

            if g == "female":
                female_sum += 1

            # Binary model - age_sum is  [3118 teen , 44434 adult]

            y_gender.append(self.genders[g])
            gender_sum[self.genders[g]] += 1
            y_age.append(self.ages[a])
            age_sum[self.ages[a]] += 1
            X_data.append(t)

        print("gender_sum is ", gender_sum)
        print("age_sum is ", age_sum)
        print(sum(gender_sum))
        print("\n")

        self.y = np.array(y_age)
        X_train, X_test, y_train, y_test = train_test_split(np.array(X_data), np.array(y_age), test_size=0.20)
        X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size=0.15)

        print("train size -", len(X_train))
        print("test size -", len(X_test))
        print("val size -", len(X_val))

        train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        self.train_loader = DataLoader(train, batch_size=50, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=50, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=50, shuffle=True)
