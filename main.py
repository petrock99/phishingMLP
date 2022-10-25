# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import copy
from datetime import timedelta
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
kHighScoreThreshold = 0.965
kCategoricalColumnThreshold = 0.95
kBatchSize = 64
kTestDataRatio = 0.15

# Tensor.shape returns a Tensor.Size, which  prints a list of values. e.g. [x, y].
# numpy.shape & pd.DataFrame.shape return a tuple. e.g. (x, y).
# Mimic the tuple printing with a Tensor.Size.
def tensor_size_pretty_str(size):
    return f"({size[0]}, {size[1]})"

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
        n_output = 1
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
        for layer, batchnorm in zip(self.layers[:-1], self.batchnorms):
            x = self.relu(layer(x))
            x = batchnorm(x)
        x = self.dropout(x)
        return self.sigmoid(self.layers[-1](x))

    def __str__(self):
        return f'{super().__str__()}\nUsing: {self.device}'


class PhishingDetector:
    def __init__(self, csv_name):
        # Set RNG seeds for more reproducible results
        torch.manual_seed(12345)
        np.random.seed(12345)

        # These are set up at runtime
        self.n_hiddens_list = []
        self.learning_rate = 0.0
        self.model = None
        self.optimizer = None
        self.loss_func = nn.BCELoss()   # Binary Cross Entropy
        self.best_state_dict = {}

        # Build a path to save the results into. Include the date/time
        # to make differentiating run results easier, and to semi-guarantee
        # a unique name within the 'results' directory.
        self.results_path = os.path.join("./results", time.strftime(f"{csv_name} %y-%m-%d at %H.%M.%S"))
        assert not os.path.exists(self.results_path), f"Expected '{self.results_path}' to not exist (yet)."
        # Create the results folder
        os.makedirs(self.results_path)
        print(f"Saving results to '{os.path.abspath(self.results_path)}'")

        # Load the .csv from disk
        self.df_data = self.load_data(self.results_path, csv_name)

        # Build the train, validate & test data sets and corresponding loaders
        (self.x_train_tensor, self.y_train_tensor,
         self.x_validate_tensor, self.y_validate_tensor,
         self.x_test_tensor, self.y_test_tensor,
         self.train_dataloader, self.validate_dataloader,
         self.test_dataloader, self.y_test) = self.split_data(self.df_data)
        print(self.data_split_str())

        # plot stats about the data
        self.plot_data()

    def build_model(self, n_hiddens_list, learning_rate):
        # Set/Reset RNG seeds for more reproducible results
        torch.manual_seed(12345)
        np.random.seed(12345)

        # (Re)Create the model.
        # Deletes the old model, if it exists, so a new one can be created
        # from scratch with new parameters & weights etc.
        n_inputs = self.x_train_tensor.shape[1]
        self.n_hiddens_list = n_hiddens_list
        self.learning_rate = learning_rate
        self.model = BinaryMLPModel(n_inputs, n_hiddens_list)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_state_dict = {}
        # print(self.model)

    @staticmethod
    def load_data(results_path, csv_name):
        # Extract the .csv if it hasn't been already
        datasets_path = "./datasets"
        csv_path = os.path.join(datasets_path, csv_name)
        if not os.path.exists(csv_path):
            zip_path = f"{csv_path}.zip"
            assert os.path.exists(zip_path), f"Expected '{zip_path}' to exist"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(datasets_path)

        # Read in the .csv dataset
        df_data = pd.read_csv(csv_path)
        # print(df_data.shape)
        # print(df_data.head(5))

        # Convert -1's in the 'Label' column to 0's to make pytorch happy.
        # 0 == Phishing, 1 == Legitimate
        assert df_data[kLabelColumn].isin([-1, 1]).all(), f"Expected only -1 & 1 in the '{kLabelColumn}' column"
        df_data.loc[df_data[kLabelColumn] < 0, kLabelColumn] = 0
        # print(df_data.head(5))

        # Remove any rows containing at least one null value
        df_data.dropna(inplace=True)

        # Get the list of dupes that are about to be removed
        df_dupes = df_data[df_data.duplicated()].copy()

        # Remove any duplicate rows
        df_data.drop_duplicates(inplace=True)

        cols_to_drop = []
        if kCategoricalColumnThreshold < 100.0:
            # Remove categorical columns containing the same value in 95% or more of its rows
            # [TODO] There has got to be a simpler, cleaner and/or built in way to do this.
            def is_col_to_drop(col):
                n_rows = df_data.shape[0]
                value_counts = df_data[col].value_counts()
                for value in value_counts.keys():
                    if value_counts[value] / n_rows >= kCategoricalColumnThreshold:
                        return True
                return False
            cols_to_drop = [col for col in df_data.keys() if col != kLabelColumn and is_col_to_drop(col)]
        else:
            # Remove columns with the same value in all of its rows
            nunique = df_data.nunique()
            cols_to_drop = nunique[nunique == 1].index

        # Remove the offending column from df_data & df_dupes
        df_data.drop(columns=cols_to_drop, inplace=True)
        df_dupes.drop(columns=cols_to_drop, inplace=True)

        # Removing columns could produce duplicate rows.
        # Append the remaining list of dupes, if any, that are about to be removed
        pd.concat([df_dupes, df_data[df_data.duplicated()]])
        # Write the full set of duplicate rows to disk
        df_dupes.to_csv(os.path.join(results_path, f"{os.path.splitext(csv_name)[0]}_all_dupes.csv"))

        # Remove any duplicates (again)
        df_data.drop_duplicates(inplace=True)

        # Make sure that the number of legitimate & phishing rows are balanced.
        # Removing dupes make have created an imbalance, or it could have been
        # imbalanced from the .csv.
        label_counts = df_data[kLabelColumn].value_counts()
        # Calculate the number of rows to remove, if any
        n_rows_to_remove = label_counts[1] - label_counts[0]
        if n_rows_to_remove != 0:
            # Get a list of rows matching the label with more rows than the other.
            # 'n_rows_to_remove > 0' remove legitimate rows, otherwise remove phishing rows
            matched_row_indexes = df_data[df_data[kLabelColumn] == (1 if n_rows_to_remove > 0 else 0)].index
            assert len(matched_row_indexes) >= n_rows_to_remove, f"Trying to remove more rows than available.\n" \
                                                                 f"# to remove: {n_rows_to_remove}, # available: {len(matched_row_indexes)}"
            # Remove the last 'n_rows_to_remove' indexes in matched_row_indexes from df_data to balance out the
            # number of Phishing & Legitimate rows
            df_data.drop(matched_row_indexes[-abs(n_rows_to_remove):], inplace=True)
            # Verify that the above code balanced out the data set
            label_counts = df_data[kLabelColumn].value_counts()
            assert label_counts[1] == label_counts[0], f"Phishing({label_counts[0]}) & Legitimate({label_counts[1]}) rows are not balanced."

        # remove all the dupes from df_dupes so it only contains unique rows for easier parsing
        # when evaluating the data set.
        df_dupes.drop_duplicates(inplace=True)
        # Write the unique dupes to disk
        df_dupes.to_csv(os.path.join(results_path, f"{os.path.splitext(csv_name)[0]}_dupes.csv"))

        # print(df_data.shape)
        return df_data

    @staticmethod
    def split_data(df_data):
        # Extract the raw data and label from df_data
        x = df_data.loc[:, df_data.columns != kLabelColumn]     # Don't include 'Label'
        y = df_data[kLabelColumn]

        # Set up a 70/15/15 split for training/validating/testing
        # Will randomize the entries.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=kTestDataRatio, random_state=42)
        validation_size_ratio = len(x_test) / len(x_train)
        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=validation_size_ratio, random_state=42)
        assert len(x_test) == len(x_validate), "Expected x_validate & x_test to be the same size"

        # Use a MinMaxScaler to scale all the features of Train, Validate & Test dataframes
        # to normalized values
        scaler = preprocessing.MinMaxScaler()
        x_train = scaler.fit_transform(x_train.values)
        x_validate = scaler.fit_transform(x_validate.values)
        x_test = scaler.fit_transform(x_test.values)
        # print(f"Scaled values of Training set\n{x_train}\n" \
        #       f"Scaled values of Validation set\n{x_validate}\n" \
        #       f"Scaled values of Testing set\n{x_test}\n")

        # Convert the Train, Validate and Test sets into Tensors
        x_train_tensor = torch.from_numpy(x_train).float()
        y_train_tensor = torch.from_numpy(y_train.values.ravel()).float()
        x_validate_tensor = torch.from_numpy(x_validate).float()
        y_validate_tensor = torch.from_numpy(y_validate.values.ravel()).float()
        x_test_tensor = torch.from_numpy(x_test).float()
        y_test_tensor = torch.from_numpy(y_test.values.ravel()).float()
        # print(f"Training set Tensors\n{x_train_tensor}\n{y_train_tensor}\n" \
        #       f"Validation set Tensors\n{x_validate_tensor}\n{y_validate_tensor}\n" \
        #       f"\nTesting set Tensors\n{x_test_tensor}\n{y_test_tensor}")

        # Stuff x_train_tensor, y_train_tensor into a TensorDataset and
        # create a DataLoader from it.
        y_train_tensor = y_train_tensor.unsqueeze(1)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=kBatchSize)

        # Stuff x_validate_tensor, y_validate_tensor into a TensorDataset and
        # create a DataLoader from it.
        y_validate_tensor = y_validate_tensor.unsqueeze(1)
        validate_dataset = TensorDataset(x_validate_tensor, y_validate_tensor)
        validate_dataloader = DataLoader(validate_dataset, batch_size=kBatchSize)

        # Stuff x_test_tensor, y_test_tensor into a TensorDataset and
        # create a DataLoader from it.
        y_test_tensor = y_test_tensor.unsqueeze(1)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=kBatchSize)

        return (x_train_tensor, y_train_tensor,
                x_validate_tensor, y_validate_tensor,
                x_test_tensor, y_test_tensor,
                train_dataloader, validate_dataloader, test_dataloader,
                y_test)

    def data_split_str(self):
        training_percent = (1.0 - kTestDataRatio * 2.0) * 100
        test_percent = kTestDataRatio * 100.0
        return f"\n-- Data Split ({training_percent} / {test_percent} / {test_percent}) --\n" \
               f"\tAll:        {self.df_data.shape}\n" \
               f"\tTraining:   {tensor_size_pretty_str(self.x_train_tensor.shape)} / {tensor_size_pretty_str(self.y_train_tensor.shape)}\n" \
               f"\tValidate:   {tensor_size_pretty_str(self.x_validate_tensor.shape)} / {tensor_size_pretty_str(self.y_validate_tensor.shape)}\n" \
               f"\tTest:       {tensor_size_pretty_str(self.x_test_tensor.shape)} / {tensor_size_pretty_str(self.y_test_tensor.shape)}"


    def plot_data(self):

        def correlation_plot(df_data, title, filename):
            fig, ax = plt.subplots(figsize=(8, 8))
            corr = df_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5, ax=ax)
            plt.title(title, fontsize=15, font="Serif")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_path, filename))
            # Clear the figure & axis for the next plot
            plt.clf()
            plt.cla()

        # Correlation Plot of all features
        correlation_plot(self.df_data.loc[:, self.df_data.columns != kLabelColumn],     # Don't include 'Label'
                         "Correlation of All Features",
                         f"correlation-all-features.png")

        # Find the names of all the Numerical columns (columns that represent data),
        # ignoring Categorical columns (columns that represent a class or type)
        # e.g. columns with only -1, 0 and/or 1 values are Categorical and will be ignored.
        # def is_numerical(col): return self.df_data[col].nunique() >= 20
        # def is_numerical(col): return col != kLabelColumn       # everything but 'Label' column
        def is_numerical(col): return self.df_data[col].nunique() > 3
        numerical_col_names = [col for col in self.df_data.keys() if is_numerical(col)]
        assert kLabelColumn not in numerical_col_names, f"'{kLabelColumn}' column shouldn't be Numerical"

        # Correlation Plot of the Numerical features
        correlation_plot(self.df_data[numerical_col_names],
                         "Correlation of Numerical (Continuous) Features",
                         f"correlation-numerical-features.png")

        # Plot a distribution of the avg numerical values across the 'Label' column
        df_distr = self.df_data.groupby(kLabelColumn)[numerical_col_names].mean().reset_index().T
        df_distr.rename(columns={0: 'Phishing', 1: "Legitimate"}, inplace=True)
        df_distr = df_distr[1:-3][['Phishing', 'Legitimate']]
        plt.rcParams['axes.facecolor'] = 'w'
        ax = df_distr.plot(kind='bar', title=f"Distribution of Average Numerical Values Across {kLabelColumn}",
                           figsize=(8, 8), legend=True, fontsize=12)
        ax.set_xlabel("Numerical Features", fontsize=14)
        ax.set_ylabel("Average Values", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f"avg-distribution-numerical-features.png"))
        # Clear the figure & axis for the next plot
        plt.clf()
        plt.cla()

        # Plot the number of unique values in each numerical column, grouped by label
        df_nuniques = self.df_data.groupby(kLabelColumn)[numerical_col_names].nunique().reset_index().T
        df_nuniques.rename(columns={0: 'Phishing', 1: "Legitimate"}, inplace=True)
        df_nuniques = df_nuniques[1:-3][['Phishing', 'Legitimate']]
        ax = df_nuniques.plot(kind='bar', title=f"Distribution of Unique Numerical Values Across {kLabelColumn}",
                              legend=True, figsize=(8, 8), fontsize=12)
        ax.set_xlabel("Numerical Features")
        ax.set_ylabel("# Unique Values")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f"distribution-unique-numerical-features.png"))
        # Clear the figure & axis for the next plot
        plt.clf()
        plt.cla()

        # Plot the number of each unique value in each categorical column
        categorical_col_names = [col for col in self.df_data.keys() if not is_numerical(col) and col != kLabelColumn]
        df_categorical = self.df_data[categorical_col_names].apply(pd.value_counts).T
        ax = df_categorical.plot(kind='bar', title=f"Distribution of Categorical Values",
                                 legend=True, figsize=(10, 8), fontsize=12)
        ax.set_xlabel("Categorical Features")
        ax.set_ylabel("# Values")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f"distribution-categorical-features.png"))
        # Clear the figure & axis for the next plot
        plt.clf()
        plt.cla()


    def plot_results(self, test_accuracy, avg_test_score,
                     min_validate_loss, min_validate_loss_epoch,
                     max_validate_accuracy, max_validate_accuracy_epoch,
                     train_accuracy_list, validate_accuracy_list,
                     train_loss_list, validate_loss_list, n_epochs):
        assert os.path.exists(self.results_path), f"Expected '{self.results_path}' to exist"

        # Set up a 1x2 grid of subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2)
        fig_size = fig.get_size_inches()
        fig.set_figwidth(fig_size[0] * 2)

        # Plot the training & validation accuracy data on the left
        ax_left.plot(train_accuracy_list, label="Training")
        ax_left.plot(validate_accuracy_list, label="Validation")
        # Plot the max accuracy location in the graph
        ax_left.plot(max_validate_accuracy_epoch, max_validate_accuracy, label="Max Val Acc", marker=".", markersize=10)
        ax_left.set_title(f"Test Acc: {test_accuracy:4f}, Max Val Acc: {max_validate_accuracy:4f}, Avg Score: {avg_test_score:4f}")
        ax_left.set(xlabel='Epoch', ylabel=f'Accuracy %')
        ax_left.legend()

        # Plot the training & validation loss data on the right
        ax_right.plot(train_loss_list, label="Training")
        ax_right.plot(validate_loss_list, label="Validation")
        # Plot the minimum loss location in the graph
        ax_right.plot(min_validate_loss_epoch, min_validate_loss, label="Min Val Loss", marker=".", markersize=10)
        ax_right.set_title(f"Min Val Loss: {min_validate_loss}")
        ax_right.set(xlabel='Epoch', ylabel='Loss')
        ax_right.legend()

        # Create a path to the .png file with the stats for this model in the name.
        # e.g. "[5, 5]-0.001-300-result.png"
        plot_file_path = os.path.join(self.results_path,
                                      f"{self.model.n_hiddens_list}-{self.learning_rate}-{n_epochs}-results.png")
        plt.tight_layout()
        # Save the plot image to disk
        plt.savefig(plot_file_path)
        # Clear the figure & axis for the next plot since pyplot holds state
        # instead of reusing figures etc.
        plt.clf()
        plt.cla()
        plt.close(fig)

    def save_model(self, n_epochs):
        assert os.path.exists(self.results_path), f"Expected '{self.results_path}' to exist"

        # Create a path to the .pt file with the stats for this model in the name.
        # e.g. "[5, 5]-0.001-300-state_dict.pt"
        state_dict_file_path = os.path.join(self.results_path,
                                            f"{self.model.n_hiddens_list}-{self.learning_rate}-{n_epochs}-state_dict.pt")
        # Save the state dictionary to disk
        torch.save(self.model.state_dict(), state_dict_file_path)

    def load_best_model(self):
        # Load the state dictionary that produced the lowest loss during
        # training back into the model so it will use those weights and
        # hopefully produce optimal results.
        assert self.best_state_dict != None, "best_state_dict should be populated by now."
        self.model.load_state_dict(self.best_state_dict)

    def save_best_model(self):
        self.best_state_dict = copy.deepcopy(self.model.state_dict())

    # Called once per epoch
    def train(self):
        n_batches = 0
        running_loss = 0.0
        running_accuracy = 0.0

        # Within each epoch process the training data in batches
        self.model.train()
        for x_train_batch, y_train_batch in self.train_dataloader:
            n_batches += 1
            y_pred = self.model(x_train_batch)  # Forward Propagation
            loss = self.loss_func(y_pred, y_train_batch)  # Loss Computation

            # Keep a running tally of the loss & accuracy in each batch from the data loader
            loss_value = loss.item()
            running_loss += loss_value
            running_accuracy += accuracy_score(y_train_batch, torch.round(y_pred).detach().numpy())

            self.optimizer.zero_grad()  # Clearing all previous gradients, setting to zero
            loss.backward()             # Back Propagation
            self.optimizer.step()       # Updating the parameters

        # Calculate the loss & accuracy for this epoch by taking the average of all
        # the losses & accuracies respectively from all the batches
        #
        # [TODO] Set up a weighted average. The last batch may not be a full batch.
        # Batches are small relative to the size of the data set so the difference
        # may be negligible.
        loss = running_loss / n_batches
        accuracy = running_accuracy / n_batches
        return (loss, accuracy)

    # Called once per epoch
    def validate(self):
        running_accuracy = 0.0
        running_loss = 0.0
        n_batches = 0

        # Since we don't need the model to back propagate the gradients during
        # validation use torch.no_grad() to reduce memory usage and speed up computation
        self.model.eval()
        with torch.no_grad():
            for x_validate_batch, y_validate_batch in self.validate_dataloader:
                n_batches += 1
                # Run x_validate_batch through the model
                y_pred = torch.round(self.model(x_validate_batch))
                y_pred_numpy = y_pred.detach().numpy()
                # Keep a running tally of the loss & accuracy during validation
                running_loss += self.loss_func(y_pred, y_validate_batch).item()
                running_accuracy += accuracy_score(y_validate_batch, y_pred_numpy)

        # Calculate the loss & accuracy for this epoch by taking the average of all
        # the losses & accuracies respectively from all the batches
        #
        # [TODO] Set up a weighted average. The last batch may not be a full batch.
        # Batches are small relative to the size of the data set so the difference
        # may be negligible.
        loss = running_loss / n_batches
        accuracy = running_accuracy / n_batches
        return (loss, accuracy)

    # Called after all the training & validation is complete
    def test(self):
        y_pred_list = []

        # Since we don't need the model to back propagate the gradients in test
        # use torch.no_grad() to reduce memory usage and speed up computation
        self.model.eval()
        with torch.no_grad():
            for x_test_batch, y_test_batch in self.test_dataloader:
                y_pred = torch.round(self.model(x_test_batch))
                y_pred_list.append(y_pred.detach().numpy())

        # Takes arrays and makes them list of list for each batch
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        # flattens the lists in sequence
        y_test_pred = list(itertools.chain.from_iterable(y_pred_list))
        # return the results
        return (confusion_matrix(self.y_test, y_test_pred),
                accuracy_score(self.y_test, y_test_pred),
                precision_score(self.y_test, y_test_pred),
                recall_score(self.y_test, y_test_pred),
                f1_score(self.y_test, y_test_pred))


def main():
    start_time = time.time()
    print(f"\nUsing {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Set up lists of parameters for the model factory to run through.
    csv_name_list = ["DS4Tan.csv"]
    n_epoch_list = [150, 300]
    learning_rate_list = [0.01, 0.001, 0.0001]
    n_hidden_lists = [[50, 50], [100, 100],
                      [50, 50, 50], [100, 100, 100],
                      [200, 200], [300, 300],
                      [400, 400], [500, 500],
                      [100, 150], [150, 100],
                      [100, 300], [300, 100],
                      [200, 300], [300, 200],
                      [400, 500], [500, 400],
                      [800, 600], [600, 800],
                      [100, 200, 50], [50, 200, 100],
                      [100, 200, 300], [300, 200, 100],
                      [50, 100, 200], [200, 100, 50],
                      [100, 50, 100, 50], [50, 100, 50, 100],
                      [300, 100, 300, 100], [100, 300, 100, 300],
                      [600, 100, 600, 100], [100, 600, 100, 600],
                      [103, 307], [307, 103],
                      [173, 421, 223], [223, 421, 173],
                      [173, 421, 223, 829, 103], [103, 829, 223, 421, 173]]

    # Run through the list of parameters set up above to generate a series of models with
    # different configs to find the 'best' model for the data set.
    for csv_name in csv_name_list:
        high_scores = []
        high_scores_to_disk = []
        ds4_tran = PhishingDetector(csv_name)
        for n_hidden_list in n_hidden_lists:
            for learning_rate in learning_rate_list:
                for n_epochs in n_epoch_list:
                    # Close any pyplot figures that may still be open from previous epochs
                    plt.close('all')

                    header_str = "\n****************************************\n" \
                                 + f"csv: '{csv_name}', hidden layers: {n_hidden_list}, epoch: {n_epochs}, learning rate: {learning_rate}"
                    print(header_str)

                    train_loss_list, train_accuracy_list = [], []
                    validate_loss_list, validate_accuracy_list = [], []
                    min_validate_loss = float("inf")
                    min_validate_loss_epoch = -1
                    max_validate_accuracy = 0
                    max_validate_accuracy_epoch = -1

                    # Start the model off fresh each run
                    ds4_tran.build_model(n_hidden_list, learning_rate)
                    for epoch in range(n_epochs):
                        # Kick off the training run for this epoch
                        (train_loss, train_accuracy) = ds4_tran.train()
                        # Validate the training so far
                        (validate_loss, validate_accuracy) = ds4_tran.validate()

                        # Keep track of the losses & accuracies from each training & validation phase
                        train_loss_list.append(train_loss)
                        train_accuracy_list.append(train_accuracy)
                        validate_loss_list.append(validate_loss)
                        validate_accuracy_list.append(validate_accuracy)

                        # Keep track of the max accuracy during validation, and its corresponding
                        # model state dictionary. It will be loaded back into the model for testing.
                        # This helps prevent over fitting.
                        if max_validate_accuracy < validate_accuracy:
                            max_validate_accuracy = validate_accuracy
                            max_validate_accuracy_epoch = epoch
                            ds4_tran.save_best_model()

                        # Keep track of the minimum loss during validation for plotting purposes
                        if validate_loss < min_validate_loss:
                            min_validate_loss = validate_loss
                            min_validate_loss_epoch = epoch

                    # Load the best model that was generated during training in order
                    # to (hopefully) produce the best testing results.
                    ds4_tran.load_best_model()

                    # Run the test phase with the newly trained model.
                    (test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1) = ds4_tran.test()
                    # Calculate the average of all the scores returned from test
                    avg_test_score = (test_accuracy + test_precision + test_recall + test_f1) / 4

                    # Plot the accuracies & losses
                    ds4_tran.plot_results(test_accuracy, avg_test_score,
                                          min_validate_loss, min_validate_loss_epoch,
                                          max_validate_accuracy, max_validate_accuracy_epoch,
                                          train_accuracy_list, validate_accuracy_list,
                                          train_loss_list, validate_loss_list, n_epochs)

                    # If the avg score is above a threshold then consider it a good run
                    if test_accuracy > kHighScoreThreshold:
                        # Build & print the metrics string
                        metrics_str = f"Accuracy:       {test_accuracy}\n" \
                                      f"Precision:      {test_precision}\n" \
                                      f"Recall:         {test_recall}\n" \
                                      f"F1:             {test_f1}\n" \
                                      f"Avg Score:      {avg_test_score}\n" \
                                      f"Max Val Acc:    {max_validate_accuracy}\n" \
                                      f"Confusion Matrix:\n{test_conf_matrix}"
                        print(metrics_str)
                        # Add the header to the metrics string to be written to disk later
                        metrics_str = f"{header_str}\n{metrics_str}"
                        # Keep track of high performing configurations
                        high_scores.append([test_accuracy, avg_test_score, n_hidden_list, n_epochs, learning_rate])
                        high_scores_to_disk.append((test_accuracy, metrics_str))
                        # Save the model to disk
                        ds4_tran.save_model(n_epochs)
                    else:
                        print(f"Test accuracy below threshold: {test_accuracy}, max validation accuracy: {max_validate_accuracy}")

        # Sort high_scores & high_scores_to_disk by test_accuracy in descending order
        def sort_func(high_score): return high_score[0]
        high_scores.sort(reverse=True, key=sort_func)
        high_scores_to_disk.sort(reverse=True, key=sort_func)

        # print the high scores
        print("\n\n------------------------------------------")
        print(f"High Scores from '{csv_name}' in Descending Order")
        [print(f"\t{i}") for i in high_scores]
        elapsed_time_str = f"Elapsed Time: {timedelta(seconds=time.time() - start_time)}"
        print(f"\n{elapsed_time_str}")

        # Save high_scores to disk for reference later
        high_scores_path = os.path.join(ds4_tran.results_path, "high_scores.txt")
        with open(high_scores_path, 'w') as fp:
            fp.write(f"{ds4_tran.data_split_str()}\n")
            fp.write(f"High Scores from '{csv_name}' in Descending Order\n")
            [fp.write(f"{i}\n") for _, i in high_scores_to_disk]
            fp.write(f"\n{elapsed_time_str}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
