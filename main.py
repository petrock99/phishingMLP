# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import itertools
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

kLabelColumn = 'Label'

class BinaryMLPModel(nn.Module):
    def __init__(self, n_inputs, n_hiddens_list):
        assert isinstance(n_hiddens_list, list), "n_hiddens_list must be a list or tuple"

        super(BinaryMLPModel, self).__init__()

        # Build a list of hidden linear layers and corresponding batchnorms
        self.layers = []
        self.batchnorms = []
        in_features = n_inputs
        out_features = 0
        for out_features in n_hiddens_list:
            self.layers.append(nn.Linear(in_features, out_features))
            self.batchnorms.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        # Set up the output layer, which only has one node to make this model
        # a binary classifier
        n_output=1
        self.layer_out = nn.Linear(out_features, n_output)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        # Use the GPU if it supports CUDA, otherwise use the CPU
        use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if use_gpu else 'cpu'
        self.to(self.device)

    def forward(self, inputs):
        x = inputs
        for (layer, batchnorm) in zip(self.layers, self.batchnorms):
            x = self.relu(layer(x))
            x = batchnorm(x)
        x = self.dropout(x)
        return self.sigmoid(self.layer_out(x))

    def __str__(self):
        return f'{super().__str__()}\nUsing: {self.device}'


class PhishingDetector:
    def __init__(self, csv_name, n_hiddens_list, plotData = False):
        # Set RNG seeds for more reproducible results
        torch.manual_seed(12345)
        np.random.seed(12345)

        self.df_data = self.loadData(csv_name)
        if plotData == True:
            self.plotData()

        self.buildTrainAndTestSets()
        self.buildDataLoaders()

        n_inputs = self.x_train_tensor.shape[1]
        self.model = BinaryMLPModel(n_inputs, n_hiddens_list)
        # print(self.model)

    def loadData(self, csv_name):
        # Extract the .csv if it hasn't been already
        datasets_path = "./datasets"
        csv_path = os.path.join(datasets_path, csv_name)
        if not os.path.exists(csv_path):
            zip_path = csv_path + ".zip"
            assert os.path.exists(zip_path), "Expected '" + zip_path + "' to exist"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(datasets_path)

        # Read in the .csv dataset
        df_data = pd.read_csv(csv_path)
        # print(df_data.shape)
        # print(df_data.head(5))

        # Convert -1's in the 'Label' column to 0's to make pytorch happy
        assert df_data[kLabelColumn].isin([-1, 1]).all(), "Expected only -1 & 1 in the 'Label' column"
        df_data.loc[df_data[kLabelColumn] < 0, kLabelColumn] = 0
        # print(df_data.head(5))

        # Remove any rows containing at least one null value
        df_data.dropna(inplace=True)
        # print(df_data.shape)
        return df_data

    def buildTrainAndTestSets(self):
        self.x = self.df_data.loc[:, self.df_data.columns != kLabelColumn]
        self.y = self.df_data[kLabelColumn]

        x_train, x_test, y_train, self.y_test = train_test_split(self.x, self.y, random_state=42)  # 75/25 split
        # print(f"\n--Training data samples--\n{x_train.shape}")

        # Use a MinMaxscaler to scale all the features of Train & Test dataframes
        scaler = preprocessing.MinMaxScaler()
        x_train = scaler.fit_transform(x_train.values)
        x_test = scaler.fit_transform(x_test.values)
        # print(f"Scaled values of Train set\n{x_train}")
        # print(f"\nScaled values of Test set\n{x_test}")

        # Convert the Train and Test sets into Tensors
        self.x_train_tensor = torch.from_numpy(x_train).float()
        self.y_train_tensor = torch.from_numpy(y_train.values.ravel()).float()
        self.x_test_tensor = torch.from_numpy(x_test).float()
        self.y_test_tensor = torch.from_numpy(self.y_test.values.ravel()).float()

        # print(f"\nTrain set Tensors\n{self.x_train_tensor}\n{self.y_train_tensor}")
        # print(f"\nTest set Tensors\n{self.x_test_tensor}\n{self.y_test_tensor}")

    def buildDataLoaders(self):
        # Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
        self.y_train_tensor = self.y_train_tensor.unsqueeze(1)
        train_dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        # Pytorch’s DataLoader is responsible for managing batches.
        # You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
        self.train_dataloader = DataLoader(train_dataset, batch_size=64)

        # For the validation/test dataset
        self.y_test_tensor = self.y_test_tensor.unsqueeze(1)
        test_dataset = TensorDataset(self.x_test_tensor, self.y_test_tensor)
        self.test_dataloader = DataLoader(test_dataset, batch_size=32)

    def plotData(self):
        # Ignore Categorical columns: columns with 3 or less unique values
        # e.g. columns with only -1, 0 and/or 1 values will be ignored.
        # is_not_categorical = lambda col : self.df_data[col].nunique() >= 20
        # is_not_categorical = lambda col : col != kLabelColumn        # everything but 'Label' column
        is_not_categorical = lambda col : self.df_data[col].nunique() > 3
        numerical_col_names = [col for col in self.df_data.keys() if is_not_categorical(col)]

        # Correlation Plot for the Numerical features
        corr = self.df_data[numerical_col_names].corr()
        fig = plt.figure(figsize=(12,12), dpi=80)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5)
        plt.title('Correlation of Numerical (Continuous) Features', fontsize=15, font="Serif")
        plt.show()

        # Display a distribution of the numerical column means
        df_distr = self.df_data.groupby(kLabelColumn)[numerical_col_names].mean().reset_index().T
        df_distr.rename(columns={0: 'Phishing', 1: "Legitimate"}, inplace=True)

        plt.rcParams['axes.facecolor'] = 'w'
        ax = df_distr[1:-3][['Phishing', 'Legitimate']].plot(kind='bar', title="Distribution of Average values across " + kLabelColumn,
                                                             figsize=(12, 8), legend=True, fontsize=12)
        ax.set_xlabel("Numerical Features", fontsize=14)
        ax.set_ylabel("Average Values", fontsize=14)
        plt.show()

    def train(self, learning_rate, n_epochs, plotLoss = False):
        loss_func = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        train_loss = []
        for epoch in range(n_epochs):
            # Within each epoch run the subsets of data = batch sizes.
            loss = None
            for xb, yb in self.train_dataloader:
                y_pred = self.model(xb)  # Forward Propagation
                loss = loss_func(y_pred, yb)  # Loss Computation
                optimizer.zero_grad()  # Clearing all previous gradients, setting to zero
                loss.backward()  # Back Propagation
                optimizer.step()  # Updating the parameters
            # if epoch % math.ceil(n_epochs/10) == 0:
            #     print(f"Loss in iteration {epoch} is: {loss.item()}")
            train_loss.append(loss.item())

        last_loss = train_loss[-1]
        if plotLoss == True:
            print(f"Last iteration loss value: {last_loss}")
            plt.plot(train_loss)
            plt.show()

        return last_loss

    def test(self):
        self.model.eval()
        # Since we don't need model to back propagate the gradients in test use torch.no_grad()
        # to reduce memory usage and speed up computation
        y_pred_list = []
        with torch.no_grad():
            for xb_test, yb_test in self.test_dataloader:
                y_pred_tag = torch.round(self.model(xb_test))
                y_pred_list.append(y_pred_tag.detach().numpy())

        # Takes arrays and makes them list of list for each batch
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        # flattens the lists in sequence
        y_test_pred = list(itertools.chain.from_iterable(y_pred_list))

        conf_matrix = confusion_matrix(self.y_test, y_test_pred)
        accuracy = accuracy_score(self.y_test, y_test_pred)
        if accuracy > .9:
            print("Confusion Matrix of the Test Set")
            print("-----------")
            print(conf_matrix)
            print(f"Accuracy of the MLP   :\t{accuracy}")
            print(f"Precision of the MLP  :\t{precision_score(self.y_test, y_test_pred)}")
            print(f"Recall of the MLP     :\t{recall_score(self.y_test, y_test_pred)}")
            print(f"F1 Score of the Model :\t{f1_score(self.y_test, y_test_pred)}")

        return accuracy


def main():

    csv_name_list = ["DS4Tan.csv"]
    n_epoch_list = [20, 40, 80, 160, 360]
    learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
    n_hidden_lists = [[100, 150], [100, 300], [200, 300], [100, 200, 50], [100, 200, 300], [300, 200, 100], [300, 200], [300, 100]]
    high_scores = []
    for csv_name in csv_name_list:
        for n_hidden_list in n_hidden_lists:
            for learning_rate in learning_rate_list:
                for n_epochs in n_epoch_list:
                    print("****************************************")
                    print(f"Hidden layers: {n_hidden_list}, learning rate: {learning_rate}, epoch: {n_epochs}")
                    ds4_tran = PhishingDetector(csv_name, n_hidden_list, plotData=False)
                    loss = ds4_tran.train(learning_rate=learning_rate, n_epochs=n_epochs, plotLoss=False)
                    if loss < 0.2:
                        accuracy = ds4_tran.test()
                        if accuracy > 0.9:
                            high_scores.append([csv_name, accuracy, n_hidden_list, n_epochs, learning_rate])
                        else:
                            print(f"Accuracy below threshold: {accuracy}")
                    else:
                        print(f"Loss above threshold: {loss}. Skipping test phase.")

    high_scores_str = "\n".join(high_scores)
    print("------------------------------------------")
    print(f"high scores:\n{high_scores_str}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
