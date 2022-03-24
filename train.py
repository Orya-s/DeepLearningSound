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


for e in range(epoch):
    print("\nepoch - ", e)

    model.train()
    train_loss, val_loss, test_loss = 0, 0, 0
    train_accuracy = 0
    count_train = 0
    train_size, test_size = 0, 0

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

            # train acc
            train_output = torch.exp(y_pred)
            train_output = torch.argmax(train_output, dim=1) + 1
            # print("train_output - ", train_output)
            train_accuracy += (train_output == label).float().sum()

    print("\ntrain_size - ", train_size)

    train_loss = np.round(train_loss / count_train, 4)
    train_accuracy /= train_size
    print("train loss - ", train_loss, " , train accuracy - ", train_accuracy)

    # checking the model's performances per epoch

    with torch.no_grad():  # only after validation ?
        model.eval()

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

        # calculating predictions on the test set

        final_accuracy = 0
        count_test = 0
        accuracy_pred = 0
        for tensor, label in data.test_loader:
            #test_epoch_idx = np.random.choice(len(label), len(label), replace=False)
           # for l in range(int(np.ceil(len(label) / batch_size))):
                #test_batch_loc = test_epoch_idx[(l * batch_size):((l + 1) * batch_size)]
                #mini_x_test, mini_y_test = tensor[test_batch_loc], label[test_batch_loc]

            proba_pred_y = model(tensor.to(device))

            t_loss = criterion(proba_pred_y, label.long().to(device))
            test_loss += t_loss.item()
            count_test += 1

            #print("label - ", label)
            output = torch.exp(proba_pred_y)
            output = torch.argmax(output, dim=1) + 1
            #print("output - ", output)
            accuracy_pred += (output == label).float().sum()
            #print("accuracy_pred - ", accuracy_pred)
            test_size += len(label)

        print("\ntest size - ", test_size, " , num of correct predictions - ", accuracy_pred)

        test_loss = np.round(test_loss / count_test, 4)
        accuracy_pred /= test_size
        print("test loss - ", test_loss, " , test accuracy - ", accuracy_pred)
        
        #print("\ncount train - ", count_train, " , count test - ", count_test)

    # torch.save(model, dir_path + "/" + model.to_string() + str(e + 1) + ".pth")

