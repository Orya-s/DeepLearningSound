import torch
import torch.nn as nn

DROP_OUT = 0.5
NUM_OF_CLASSES = 3


class ConvNet_roi_orya(nn.Module):

    def __init__(self, num_of_classes):
        super().__init__()

        self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(5, 5), padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv_2d_6 = nn.Conv2d(256, 64, kernel_size=(2, 2), padding=1)

        self.conv_2d_7 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.max_pool_2d_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv_2d_8 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(32, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, num_of_classes)

    def forward(self, X):
        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.bn_5(x)
        x = self.max_pool_2d_3(x)

        x = nn.ReLU()(self.conv_2d_6(x))

        x = nn.ReLU()(self.conv_2d_7(x))
        x = self.max_pool_2d_4(x)

        x = nn.ReLU()(self.conv_2d_8(x))

        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))
        x = self.drop_2(x)

        x = self.dense_2(x)
        y = nn.LogSoftmax(dim=1)(x)

        return y

    def get_epochs(self):
        return 30

    def get_learning_rate(self):
        return 0.0001

    def get_batch_size(self):
        return 50

    def to_string(self):
        return "CNN_Model-epoch_"

