import torch
from cnn_model_definition import Convolutional_Neural_Network
from preprocessing import prepareData
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib


print('Start training:')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolutional_Neural_Network().to(device)
data = prepareData()

# setting model's parameters
learning_rate = model.get_learning_rate()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()
epoch, batch_size = model.get_epochs(), model.get_batch_size()

# # preparing txt report file
# training_results_path = 'results\\'
# results_df = pd.DataFrame([], columns=['train_loss', 'val_loss', 'top_1_test_acc', 'top_5_test_acc', 'top_10_test_acc'])
# now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
# dir_path = training_results_path + now_time
# if not os.path.isdir(dir_path):
#     os.mkdir(dir_path)
#
# file = open(dir_path + '/reuslts.txt', 'w')
# file_txt = ['Date and time :  ' + now_time, 'Learning Rate : ' + str(learning_rate), 'Batch Size : ' + str(batch_size),
#             'Epoch Number : ' + str(epoch)]
# for s in file_txt:
#     file.write('\r----------------------------\r\r')
#     file.write(s)


def top_k_accuracy(k, proba_pred_y, mini_y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(mini_y_test)
    for j in range(len(mini_y_test)):
        final_pred[j] = True if sum(top_k_pred[j] == mini_y_test[j]) > 0 else False
    return np.mean(final_pred)


for e in range(epoch):

    model.train()
    train_loss = 0
    count_train = 0

    i = 0
    for tensor, age in data.train_loader:

        epoch_size = len(tensor)
        train_epoch_idx = np.random.choice(len(age), epoch_size, replace=False)
        np.random.shuffle(train_epoch_idx)
        batch_num = int(np.ceil(epoch_size / batch_size))

        for b in tqdm(range(batch_num)):
            optimizer.zero_grad()
            batch_loc = train_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
            x_batch, y_batch = tensor[batch_loc], age[batch_loc]

            y_pred = model(x_batch.to(device))
            loss = criterion(y_pred, y_batch.long().to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            count_train += 1

            # i += 1
            # if i % 5 == 0:
            #     print("\n", loss)

    train_loss = np.round(train_loss / count_train, 4)
    print("\nloss - ", train_loss)

    # checking the model's performances per epoch

    with torch.no_grad():
        model.eval()

        # calculating predictions on the test set
        final_accuracy = np.array([0, 0, 0], dtype=float)
        count_test = 0
        for tensor, age in data.test_loader:
            test_epoch_idx = np.random.choice(len(age), len(age), replace=False)

            for l in range(int(np.ceil(len(age) / batch_size))):
                test_batch_loc = test_epoch_idx[(l * batch_size):((l + 1) * batch_size)]
                mini_x_test, mini_y_test = tensor[test_batch_loc], age[test_batch_loc]
                proba_pred_y = model(mini_x_test.to(device))
                count_test += 1
                accuracy_list = []

                for k in [1, 5, 10]:
                    accuracy_list += [top_k_accuracy(k, proba_pred_y, mini_y_test)]
                final_accuracy += np.array(accuracy_list)

        final_accuracy /= count_test

    #torch.save(model, dir_path + "/" + model.to_string() + str(e + 1) + ".pth")
    new_final_accuracy = [round(x * 100, 3) for x in list(final_accuracy)]
    print("new_final_accuracy - ", new_final_accuracy)


