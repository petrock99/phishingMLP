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
import time
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
        layers = []
        batchnorms = []
        in_features = n_inputs
        out_features = 0
        for out_features in n_hiddens_list:
            layers.append(nn.Linear(in_features, out_features))
            batchnorms.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        # Set up the output layer, which only has one node to make this model
        # a binary classifier
        n_output=1
        layers.append(nn.Linear(out_features, n_output))

        self.n_hiddens_list = n_hiddens_list
        self.layers = nn.ModuleList(layers)
        self.batchnorms = nn.ModuleList(batchnorms)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        # Use the GPU if it supports CUDA, otherwise use the CPU
        use_gpu = torch.cuda.is_available()
        self.device = 'cuda' if use_gpu else 'cpu'
        self.to(self.device)

    def forward(self, inputs):
        x = inputs
        assert len(self.layers) == len(self.batchnorms) + 1, "Expected last layer to be the output later"
        for (layer, batchnorm) in zip(self.layers[:-1], self.batchnorms):
            x = self.relu(layer(x))
            x = batchnorm(x)
        x = self.dropout(x)
        return self.sigmoid(self.layers[-1](x))

    def __str__(self):
        return f'{super().__str__()}\nUsing: {self.device}'


class PhishingDetector:
    def __init__(self, csv_name, plot_data = False):
        # Set RNG seeds for more reproducible results
        torch.manual_seed(12345)
        np.random.seed(12345)

        # Load the .csv from disk
        self.df_data = self.load_data(csv_name)
        if plot_data == True:
            self.plot_data()
        # Build the training & test data sets & loaders
        self.build_train_and_test_sets()
        self.build_data_loaders()

        # Build a path to save the loss plots into. Include the date/time
        # to make differentiating runs easier.
        self.loss_plots_path = os.path.join("./loss_plots", time.strftime(f"{csv_name} %y-%m-%d at %H.%M.%S"))

    def build_model(self, n_hiddens_list):
        # Set RNG seeds for more reproducible results
        torch.manual_seed(12345)
        np.random.seed(12345)

        n_inputs = self.x_train_tensor.shape[1]
        self.n_hiddens_list = n_hiddens_list
        self.model = BinaryMLPModel(n_inputs, n_hiddens_list)
        # print(self.model)

    def load_data(self, csv_name):
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

    def build_train_and_test_sets(self):
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

    def build_data_loaders(self):
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

    def plot_data(self):
        # Ignore Categorical columns: columns with 3 or less unique values
        # e.g. columns with only -1, 0 and/or 1 values will be ignored.
        # is_not_categorical = lambda col : self.df_data[col].nunique() >= 20
        # is_not_categorical = lambda col : col != kLabelColumn        # everything but 'Label' column
        is_not_categorical = lambda col : self.df_data[col].nunique() > 3
        numerical_col_names = [col for col in self.df_data.keys() if is_not_categorical(col)]

        # Correlation Plot for the Numerical features
        corr = self.df_data[numerical_col_names].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5)
        plt.title('Correlation of Numerical (Continuous) Features', fontsize=15, font="Serif")
        plt.show()
        plt.clf()

        # Display a distribution of the numerical column means
        df_distr = self.df_data.groupby(kLabelColumn)[numerical_col_names].mean().reset_index().T
        df_distr.rename(columns={0: 'Phishing', 1: "Legitimate"}, inplace=True)

        plt.rcParams['axes.facecolor'] = 'w'
        ax = df_distr[1:-3][['Phishing', 'Legitimate']].plot(kind='bar', title="Distribution of Average values across " + kLabelColumn,
                                                             figsize=(12, 8), legend=True, fontsize=12)
        ax.set_xlabel("Numerical Features", fontsize=14)
        ax.set_ylabel("Average Values", fontsize=14)
        plt.show()
        plt.clf()

    def plot_loss(self, train_loss, learning_rate, n_epochs, to_disk=False):
        plt.plot(train_loss)
        if to_disk == True:
            # Create a folder to save the loss plots into if necessary
            if not os.path.exists(self.loss_plots_path):
                os.makedirs(self.loss_plots_path)
            plot_file_path = os.path.join(self.loss_plots_path,
                                          f"{self.model.n_hiddens_list}-{learning_rate}-{n_epochs}.png")
            plt.savefig(plot_file_path)
        else:
            plt.show()
        plt.clf()

    def train(self, learning_rate, n_epochs, plot_loss = False, to_disk = False):
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
        if plot_loss == True:
            self.plot_loss(train_loss, learning_rate, n_epochs, to_disk)
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
        precision = precision_score(self.y_test, y_test_pred)
        recall = recall_score(self.y_test, y_test_pred)
        f1 = f1_score(self.y_test, y_test_pred)
        avg_score = (accuracy + precision + recall + f1) / 4
        if avg_score > .95:
            print(f"Confusion Matrix of the Test Set\n{conf_matrix}")
            print(f"Accuracy:   {accuracy}")
            print(f"Precision:  {precision}")
            print(f"Recall:     {recall}")
            print(f"F1:         {f1}")
            print(f"Avg Score:  {avg_score}")

        return avg_score


def main():
    print(f"\nUsing {'GPU' if torch.cuda.is_available() else 'CPU'}")

    csv_name_list = ["DS4Tan.csv"]
    # n_epoch_list = [40, 80, 160, 360]
    # learning_rate_list = [0.01, 0.001, 0.0001]
    # n_hidden_lists = [[100, 150], [100, 300], [200, 300], [100, 200, 50], [100, 200, 300], [300, 200, 100], [300, 200], [300, 100]]
    n_epoch_list = [300]
    # learning_rate_list = [0.01, 0.001, 0.0001]
    learning_rate_list = [0.0001]
    n_hidden_lists = [[200, 300], [300, 200], [100, 300], [300, 100], [100, 200, 50], [50, 200, 100], [100, 200, 300], [300, 200, 100], [50, 50, 50], [100, 100, 100], [400, 500], [500, 400]]
    for csv_name in csv_name_list:
        high_scores = []
        ds4_tran = PhishingDetector(csv_name, plot_data=False)
        for n_hidden_list in n_hidden_lists:
            for learning_rate in learning_rate_list:
                for n_epochs in n_epoch_list:
                    print("\n****************************************")
                    print(f"Hidden layers: {n_hidden_list}, learning rate: {learning_rate}, epoch: {n_epochs}")
                    # Start the model off fresh each run
                    ds4_tran.build_model(n_hidden_list)
                    # Kick off the training run
                    loss = ds4_tran.train(learning_rate=learning_rate, n_epochs=n_epochs, plot_loss=True, to_disk=True)
                    avg_score = ds4_tran.test()
                    if avg_score > 0.95:
                        # Keep track of high performing configurations
                        high_scores.append([csv_name, avg_score, loss, n_hidden_list, n_epochs, learning_rate])
                    else:
                        print(f"Avg score below threshold: {avg_score}, loss: {loss}")

        # Sort by avg_score in descending order
        def sort_func(element):
            return element[1]
        high_scores.sort(reverse=True, key=sort_func)
        print("------------------------------------------")
        print("high scores:")
        [print(i) for i in high_scores]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
