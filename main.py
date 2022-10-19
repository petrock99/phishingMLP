# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
import itertools

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def loadData():
    # Extract the .csv if it hasn't been already
    csvPath = "./datasets/DS4Tan.csv"
    if not os.path.exists(csvPath):
        with zipfile.ZipFile("./datasets/DS4Tan.csv.zip", "r") as zip_ref:
            zip_ref.extractall("./datasets")

    # Read in the .csv dataset
    df_data = pd.read_csv(csvPath)
    # print(df_data.shape)
    # print(df_data.head(5))

    # Convert -1's in the 'Label' column to 0's. pytorch binary NNs require this.
    assert df_data['Label'].isin([-1, 1]).all(), "Expected only -1 & 1 in the 'Label' column"
    df_data.loc[df_data['Label'] < 0, 'Label'] = 0
    # print(df_data.head(5))

    # Remove any rows containing at least one null value
    df_data.dropna(inplace=True)
    # print(df_data.shape)

    # Ignore Categorical columns: columns with 3 or less unique values
    # e.g. columns with only -1, 0 and/or 1 values will be ignored.
    # is_not_categorical = lambda col : (1. * df_data[col].nunique() / df_data[col].count()) >= 0.002
    # is_not_categorical = lambda col : (1. * df_data[col].nunique() / df_data[col].count()) >= 0.0004
    # is_not_categorical = lambda col : df_data[col].nunique() >= 20
    # is_not_categorical = lambda col : col != 'Label'        # everything but 'Label' column
    is_not_categorical = lambda col : df_data[col].nunique() > 3
    useful_col_names = [col for col in df_data.keys() if is_not_categorical(col)]
    # print(df_data[column_names].shape)
    return (df_data, useful_col_names)


def correlationPlot(df_data):
    # Correlation Plot for the Numerical(continuous) features
    corr = df_data.corr()
    fig = plt.figure(figsize=(12,12), dpi=80)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5)
    plt.title('Correlation of Numerical(Continuous) Features', fontsize=15, font="Serif")
    plt.show()


def distributionOfMean(df_all_data, useful_col_names):
    df_distr = df_all_data.groupby('Label')[useful_col_names].mean().reset_index().T
    df_distr.rename(columns={0: 'Phishing', 1: "Legitimate"}, inplace=True)

    # plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'w'
    ax = df_distr[1:-3][['Phishing', 'Legitimate']].plot(kind='bar', title="Distribution of Average values across 'Label'",
                                                         figsize=(12, 8), legend=True, fontsize=12)
    ax.set_xlabel("Numerical Features", fontsize=14)
    ax.set_ylabel("Average Values", fontsize=14)
    # ax.set_ylim(0,500000)
    plt.show()


def buildTrainAndTestSets(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)  # 75/25 split
    print("\n--Training data samples--")
    print(x_train.shape)

    # Use a MinMaxscaler to scale all the features of Train & Test dataframes
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train.values)
    x_test = scaler.fit_transform(x_test.values)
    print("Scaled values of Train set \n")
    print(x_train)
    print("\nScaled values of Test set \n")
    print(x_test)

    # Convert the Train and Test sets into Tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train.values.ravel()).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test.values.ravel()).float()

    print("\nTrain set Tensors \n")
    print(x_train_tensor)
    print(y_train_tensor)
    print("\nTest set Tensors \n")
    print(x_test_tensor)
    print(y_test_tensor)

    return (x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor)


def buildDataLoader(x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor):
    # Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
    y_train_tensor = y_train_tensor.unsqueeze(1)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    # Pytorch’s DataLoader is responsible for managing batches.
    # You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    # For the validation/test dataset
    y_test_tensor = y_test_tensor.unsqueeze(1)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    return (train_dataloader, test_dataloader)


class BinaryMLPModel(nn.Module):
    def __init__(self, n_input_dim):
        super(BinaryMLPModel, self).__init__()

        n_hidden1 = 300  # Number of hidden nodes
        n_hidden2 = 100
        n_output = 1  # Number of output nodes = for binary classifier

        self.layer_1 = nn.Linear(n_input_dim, n_hidden1)
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)

        # Use the GPU if it supports CUDA, otherwise use the CPU
        use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if use_gpu else 'cpu'
        self.to(self.device)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        return x

    def __str__(self):
        return f'{super().__str__()}\nUsing: {self.device}'


def train(model, train_dataloader):
    learning_rate = 0.001
    epochs = 150

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train_loss = []
    for epoch in range(epochs):
        # Within each epoch run the subsets of data = batch sizes.
        for xb, yb in train_dataloader:
            y_pred = model(xb)            # Forward Propagation
            loss = loss_func(y_pred, yb)  # Loss Computation
            optimizer.zero_grad()         # Clearing all previous gradients, setting to zero
            loss.backward()               # Back Propagation
            optimizer.step()              # Updating the parameters
        # print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
        train_loss.append(loss.item())
    print('Last iteration loss value: '+str(loss.item()))

    plt.plot(train_loss)
    plt.show()


def test(model, test_dataloader, y_test_tensor):
    y_pred_list = []
    y_test = []
    model.eval()
    # Since we don't need model to back propagate the gradients in test use torch.no_grad()
    # to reduce memory usage and speed up computation
    with torch.no_grad():
        for xb_test, yb_test in test_dataloader:
            y_test_pred = model(xb_test)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.detach().numpy())
            y_test.append(yb_test.detach().numpy())

    # Takes arrays and makes them list of list for each batch
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [a.squeeze().tolist() for a in y_test]
    # flattens the lists in sequence
    y_test_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_test = list(itertools.chain.from_iterable(y_test))

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix of the Test Set")
    print("-----------")
    print(conf_matrix)
    print("Accuracy of the MLP :\t"+str(accuracy_score(y_test, y_test_pred)))
    print("Precision of the MLP :\t"+str(precision_score(y_test, y_test_pred)))
    print("Recall of the MLP    :\t"+str(recall_score(y_test, y_test_pred)))
    print("F1 Score of the Model :\t"+str(f1_score(y_test, y_test_pred)))

def main():
    # Set RNG seeds for more reproducible results
    torch.manual_seed(12345)
    np.random.seed(12345)

    (df_data, useful_col_names) = loadData()
    df_filtered_data = df_data[useful_col_names]
    correlationPlot(df_filtered_data)
    distributionOfMean(df_data, useful_col_names)
    (x_train_tensor,
     x_test_tensor,
     y_train_tensor,
     y_test_tensor) = buildTrainAndTestSets(df_filtered_data,
                                            df_data['Label'])
    (train_dataloader,
     test_dataloader) = buildDataLoader(x_train_tensor,
                                        x_test_tensor,
                                        y_train_tensor,
                                        y_test_tensor)
    model = BinaryMLPModel(x_train_tensor.shape[1])
    print(model)
    train(model, train_dataloader)
    test(model, test_dataloader, y_test_tensor)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
