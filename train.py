import time

import torch
from sklearn.utils import class_weight

from cnn_model_definition import Convolutional_Neural_Network
from preprocessing import prepareData

import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

start = time.time()
print('Start')
print('Start training:')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolutional_Neural_Network().to(device)
data = prepareData()

sum_op = len(data.genders)

# setting model's parameters
learning_rate = model.get_learning_rate()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(data.y), y=data.y)
class_weights = torch.tensor(class_weights, dtype=torch.float)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# criterion = torch.nn.CrossEntropyLoss()
epoch, batch_size = model.get_epochs(), model.get_batch_size()

for e in range(epoch):
    print("\nepoch - ", e)

    model.train()
    train_loss, val_loss, test_loss = 0, 0, 0
    count_train = 0
    train_size = 0
    test_size = 0

    for tensor, label in data.train_loader:

        epoch_size = len(tensor)
        train_epoch_idx = np.random.choice(len(label), epoch_size, replace=False)
        np.random.shuffle(train_epoch_idx)
        batch_num = int(np.ceil(epoch_size / batch_size))

        for b in tqdm(range(batch_num)):
            optimizer.zero_grad()
            batch_loc = train_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
            x_batch, y_batch = tensor[batch_loc], label[batch_loc]

            y_pred = model(x_batch.to(device))
            loss = criterion(y_pred, y_batch.long().to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            count_train += 1
            train_size += len(label)

    train_loss = np.round(train_loss / count_train, 4)
    print("\ntrain_size - ", train_size)
    print("\ntrain loss - ", train_loss)

    # checking the model's performances per epoch

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(sum_op)]
        n_class_samples = [0 for i in range(sum_op)]
        for embedding, labels in data.test_loader:
            embedding = embedding.to(device)

            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = model(embedding)

            # euler FIX
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):

                _label = labels[i]
                pred = predicted[i]
                if _label == pred:
                    n_class_correct[_label] += 1
                n_class_samples[_label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(sum_op):
            if n_class_correct[i] == 0:
                acc = 0
            else:
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {i + 1}: {acc} %')

        # val = data.val_loader
        # count_val = 0
        # for tensor, label in data.val_loader:
        #     val_epoch_idx = np.random.choice(len(label), len(label), replace=False)
        #     for h in range(int(np.ceil(len(label) / batch_size))):
        #         val_batch_loc = val_epoch_idx[(h * batch_size): ((h + 1) * batch_size)]
        #         mini_x_val, mini_y_val = tensor[val_batch_loc], label[val_batch_loc]
        #         y_pred_val = model(mini_x_val.to(device))
        #         val_loss += criterion(y_pred_val, mini_y_val.long().to(device)).item()
        #         count_val += 1
        # val_loss = np.round(val_loss / count_val, 4)
        # print("\nval loss - ", val_loss)
        if e % 10 == 0:
            torch.save(model, 'models\\' + "/" + "gender_" + model.to_string() + str(e + 1) + "_Weights.pth")

            y_pred = []
            y_true = []

            # iterate over test data
            for inputs, labels in data.test_loader:
                output = model(inputs)  # Feed Network

                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth

            # constant for classes
            classes = data.genders.keys()

            # Build confusion matrix
            cf_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            plt.savefig('confusion_matrix' + "_gender_" + model.to_string() + str(e + 1) + "_Weights.png")

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in data.test_loader:
    output = model(inputs)  # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)  # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

# constant for classes
classes = data.genders.keys()

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix' + "_gender_" + model.to_string() + str(e + 1) + "_Weights.png")
print("End")
end = time.time()
print("Total time = ", end - start)

torch.save(model, 'models\\' + "/" + "gender_" + model.to_string() + str(e + 1) + "_Weights.pth")
