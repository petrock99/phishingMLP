import copy
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import seaborn as sns
from stuff.model import BinaryMLPModel
from stuff import utils
import time
import zipfile

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import ( accuracy_score,
                              roc_auc_score,
                              confusion_matrix,
                              precision_score,
                              recall_score,
                              f1_score )
from sklearn.utils import shuffle

import torch
import torch.nn
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

kLabelColumn = 'Label'
kBatchSize = None
kDatasetsPath = None
kSameValueInColumnThreshold = None
kValidateDataRatio = None
kNumKFolds = None
kEarlyStopPatience = None
kUseGPU = None


class PhishingDetector:
    def __init__(self, csv_name):
        # Set RNG seeds for more reproducible results
        self.reset_RNG()

        # These are set up at runtime
        self.n_hiddens_list = []
        self.learning_rate = 0.0
        self.model = None
        self.optimizer = None
        self.loss_func = nn.BCELoss()   # Binary Cross Entropy
        self.best_state_dict = {}
        global kUseGPU
        self.device = 'cuda' if kUseGPU else 'cpu'

        # Build a path to save the results into. Include the date/time
        # to make differentiating run results easier, and to semi-guarantee
        # a unique name within the 'results' directory.
        self.results_path = os.path.join("./results", time.strftime(f"{utils.strip_extension(csv_name)} %y-%m-%d at %H.%M.%S"))
        assert not os.path.exists(self.results_path), f"Expected '{self.results_path}' to not exist (yet)."
        # Create the results folder
        os.makedirs(self.results_path)
        print(f"Saving results to '{os.path.abspath(self.results_path)}'")

        # Load the .csv from disk
        self.df_data = self.load_data(self.results_path, csv_name)

        # Build the train & validate tensors & datasets and k-fold object
        (self.x_train_tensor, self.y_train_tensor, self.train_dataset,
         self.x_validate_tensor, self.y_validate_tensor, self.validate_dataloader,
         self.k_fold) = self.split_data(self.df_data, self.device)

        # plot stats about the data
        self.plot_data()

    def build_model(self, n_hiddens_list, learning_rate):
        # Set/Reset RNG seeds for more reproducible results
        self.reset_RNG()

        # (Re)Create the model.
        # Deletes the old model, if it exists, so a new one can be created
        # from scratch with new parameters & weights etc.
        global kUseGPU
        n_inputs = self.x_train_tensor.shape[1]
        self.n_hiddens_list = n_hiddens_list
        self.learning_rate = learning_rate
        self.model = BinaryMLPModel(n_inputs, n_hiddens_list, kUseGPU)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_state_dict = {}
        # print(self.model)

    @staticmethod
    def reset_RNG():
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)
        np.random.seed(12345)
        random.seed(12345)

    @staticmethod
    def set_constants(constant_dict):
        global kBatchSize, kDatasetsPath, kSameValueInColumnThreshold, kValidateDataRatio, kNumKFolds, kEarlyStopPatience, kUseGPU
        kBatchSize = constant_dict["kBatchSize"]
        kDatasetsPath = constant_dict["kDatasetsPath"]
        kSameValueInColumnThreshold = constant_dict["kSameValueInColumnThreshold"]
        kValidateDataRatio = constant_dict["kValidateDataRatio"]
        kNumKFolds = constant_dict["kNumKFolds"]
        kEarlyStopPatience = constant_dict["kEarlyStopPatience"]
        kUseGPU = constant_dict["kUseGPU"]

    @staticmethod
    def load_data(results_path, csv_name):
        # Extract the .csv if it hasn't been already
        global kDatasetsPath
        csv_path = os.path.join(kDatasetsPath, csv_name)
        if not os.path.exists(csv_path):
            zip_path = f"{csv_path}.zip"
            assert os.path.exists(zip_path), f"Expected '{zip_path}' to exist"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(kDatasetsPath)

        # Read in the .csv dataset
        df_data = pd.read_csv(csv_path)
        # print(df_data.shape)
        # print(df_data.head(5))

        # Convert -1's in the 'Label' column to 0's to make pytorch happy.
        # 1 == Phishing, 0 == Legitimate
        global kLabelColumn
        assert df_data[kLabelColumn].isin([-1, 1]).all(), f"Expected only -1 & 1 in the '{kLabelColumn}' column"
        df_data.loc[df_data[kLabelColumn] < 0, kLabelColumn] = 0
        # print(df_data.head(5))

        # Remove any rows containing at least one null value
        df_data.dropna(inplace=True)

        # Get the list of dupes that are about to be removed
        df_dupes = df_data[df_data.duplicated()].copy()

        # Remove any duplicate rows
        df_data.drop_duplicates(inplace=True)

        global kSameValueInColumnThreshold
        if kSameValueInColumnThreshold < 100.0:
            # Remove categorical columns containing the same value in 95% or more of its rows
            # [TODO] There has got to be a simpler, cleaner and/or built in way to do this.
            def is_col_to_drop(col):
                n_rows = df_data.shape[0]
                value_counts = df_data[col].value_counts()
                for value in value_counts.keys():
                    if value_counts[value] / n_rows >= kSameValueInColumnThreshold:
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
        df_dupes.to_csv(os.path.join(results_path, f"{utils.strip_extension(csv_name)}_all_dupes.csv"))

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
            # 'n_rows_to_remove > 0' remove phishing rows, otherwise remove legitimate rows
            matched_row_indexes = df_data[df_data[kLabelColumn] == (1 if n_rows_to_remove > 0 else 0)].index
            assert len(matched_row_indexes) >= n_rows_to_remove, f"Trying to remove more rows than available.\n" \
                                                                 f"# to remove: {n_rows_to_remove}, # available: {len(matched_row_indexes)}"
            # Remove the last 'n_rows_to_remove' indexes in matched_row_indexes from df_data to balance out the
            # number of Phishing & Legitimate rows
            df_data.drop(matched_row_indexes[-abs(n_rows_to_remove):], inplace=True)
            # Verify that the above code balanced out the data set
            label_counts = df_data[kLabelColumn].value_counts()
            assert label_counts[1] == label_counts[0], f"Legitimate({label_counts[0]}) & Phishing({label_counts[1]}) rows are not balanced."

        # remove all the dupes from df_dupes, so it only contains unique rows for easier parsing
        # when evaluating the data set.
        df_dupes.drop_duplicates(inplace=True)
        # Write the unique dupes to disk
        df_dupes.to_csv(os.path.join(results_path, f"{utils.strip_extension(csv_name)}_dupes.csv"))

        # Write the dataset we will run against to disk
        df_data.to_csv(os.path.join(results_path, f"{utils.strip_extension(csv_name)}_filtered.csv"))

        # print(df_data.shape)
        return df_data

    @staticmethod
    def split_data(df_data, device):
        # Extract the raw data and label from df_data
        global kLabelColumn
        x_series = df_data.loc[:, df_data.columns != kLabelColumn]     # Don't include 'Label'
        y_series = df_data[kLabelColumn]

        # Use a MinMaxScaler to scale all the features to normalized values
        # Also convert the Pandas Series objects into Numpy Arrays
        x_np = preprocessing.MinMaxScaler().fit_transform(x_series.values)
        y_np = y_series.values
        # print(f"Scaled values\n{x_scaled}\n"

        global kValidateDataRatio
        if kValidateDataRatio > 0:
            (x_train,
             x_validate,
             y_train,
             y_validate) = train_test_split(x_np,
                                            y_np,
                                            test_size=kValidateDataRatio,
                                            shuffle=True,
                                            random_state=42,
                                            stratify=y_np)
        else:
            x_train, y_train = shuffle(x_np, y_np, random_state=42)
            x_validate = None
            y_validate = None

        # Convert the Train & Validate sets into Tensors
        x_train_tensor = torch.from_numpy(x_train).float().to(device)
        y_train_tensor = torch.from_numpy(y_train.ravel()).float().to(device)
        # print(f"Training set Tensors\n{x_train_tensor}\n{y_train_tensor}\n")

        # Stuff x_train_tensor, y_train_tensor into a TensorDataset.
        # The DataLoader will be created later based on the k-fold split
        y_train_tensor = y_train_tensor.unsqueeze(1)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

        x_validate_tensor = None
        y_validate_tensor = None
        validate_dataloader = None
        if x_validate is not None:
            x_validate_tensor = torch.from_numpy(x_validate).float().to(device)
            y_validate_tensor = torch.from_numpy(y_validate.ravel()).float().to(device)
            # print(f"Validation set Tensors\n{x_validate_tensor}\n{y_validate_tensor}\n")

            # Stuff x_validate_tensor, y_validate_tensor into a TensorDataset
            y_validate_tensor = y_validate_tensor.unsqueeze(1)
            validate_dataset = TensorDataset(x_validate_tensor, y_validate_tensor)
            validate_batch_size = kBatchSize if kBatchSize != 0 else len(validate_dataset)
            validate_dataloader = DataLoader(validate_dataset, batch_size=validate_batch_size)

        # Set up the k-fold object that will handle the shuffling of the data during .split(...)
        global kNumKFolds
        k_fold = StratifiedKFold(n_splits=kNumKFolds)

        return (x_train_tensor, y_train_tensor, train_dataset,
                x_validate_tensor, y_validate_tensor, validate_dataloader,
                k_fold)

    def data_split_str(self, csv_name):
        global kValidateDataRatio
        training_percent = (1.0 - kValidateDataRatio) * 100
        validation_percent = kValidateDataRatio * 100.0
        validate_str = f"\tValidate:   {utils.tensor_size_pretty_str(self.x_validate_tensor.shape)} / {utils.tensor_size_pretty_str(self.y_validate_tensor.shape)}\n" \
                       if kValidateDataRatio > 0 else ''
        return f"-- Dataset '{csv_name}' --\n" \
               f"\tSplit:      {training_percent} / {validation_percent}\n" \
               f"\tAll:        {self.df_data.shape}\n" \
               f"\tTraining:   {utils.tensor_size_pretty_str(self.x_train_tensor.shape)} / {utils.tensor_size_pretty_str(self.y_train_tensor.shape)}\n" \
               f"{validate_str}"

    def plot_data(self):

        def correlation_plot(df_data, title, filename):
            _, corr_ax = plt.subplots(figsize=(8, 8))
            corr = df_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='BuPu', robust=True, center=0, square=True, linewidths=.5, ax=corr_ax)
            plt.title(title.replace("X", str(df_data.shape[1]), 1), fontsize=15, font="Serif")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_path, filename))
            # Clear the figure & axis for the next plot
            plt.clf()
            plt.cla()

        # Correlation Plot of all features
        global kLabelColumn
        correlation_plot(self.df_data.loc[:, self.df_data.columns != kLabelColumn],     # Don't include 'Label'
                         "Correlation of All X Features",
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
                         "Correlation of X Numerical Features",
                         f"correlation-numerical-features.png")

        # Plot a distribution of the avg numerical values across the 'Label' column
        df_distr = self.df_data.groupby(kLabelColumn)[numerical_col_names].mean().reset_index().T
        df_distr.rename(columns={0: "Legitimate", 1: "Phishing"}, inplace=True)
        df_distr = df_distr[1:-3][["Legitimate", "Phishing"]]
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
        df_nuniques.rename(columns={0: "Legitimate", 1: "Phishing"}, inplace=True)
        df_nuniques = df_nuniques[1:-3][["Legitimate", "Phishing"]]
        ax = df_nuniques.plot(kind='bar', title=f"Distribution of Unique Numerical Values Across {kLabelColumn}",
                              legend=True, figsize=(8, 8), fontsize=12)
        ax.set_xlabel(f"{len(numerical_col_names)} Numerical Features")
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
        ax.set_xlabel(f"{len(categorical_col_names)} Categorical Features")
        ax.set_ylabel("# Values")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f"distribution-categorical-features.png"))
        # Clear the figure & axis for the next plot
        plt.clf()
        plt.cla()

        self.plot_k_fold()

    # Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
    def plot_k_fold(self):
        global kNumKFolds
        cv = self.k_fold
        X = self.x_train_tensor.cpu()
        y = self.y_train_tensor.cpu()
        n_splits = kNumKFolds
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm
        lw = 10

        # Generate the training/testing visualizations for each CV split
        fig, ax = plt.subplots(figsize=(6, 3))
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=lw,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )

        # Plot the data classes at the end
        ax.scatter(
            range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
        )

        # Formatting
        yticklabels = list(range(n_splits)) + ["class"]
        ax.set(
            yticks=np.arange(n_splits + 1) + 0.5,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylabel="Fold",
            ylim=[n_splits + 1.2, -0.2]
        )
        ax.set_title(f"{type(cv).__name__} of Shuffled Data", fontsize=15)

        ax.legend(["Testing set", "Training set"], loc=(1.02, 0.8))
        # Make the legend fit
        plt.tight_layout()
        fig.subplots_adjust(right=0.7)
        # Save the plot to disk
        plt.savefig(os.path.join(self.results_path, f"stratified-kfold-train-distribution.png"))
        # Clear the figure & axis for the next plot
        plt.clf()
        plt.cla()

    # Create a path to the .png file with the stats for this model in the name.
    # e.g. "[5, 5]-0.001-2-result.png" or "[5, 5]-0.001-result.png"
    def plot_results_path(self, fold=None):
        fold_str = f"-{fold}" if fold is not None else ""
        return os.path.join(self.results_path,
                            f"{self.model.n_hiddens_list}-{self.learning_rate}{fold_str}-results.png")

    def plot_results(self, test_accuracy, test_auc,
                     min_validate_loss, min_validate_loss_epoch,
                     max_validate_accuracy, max_validate_accuracy_epoch,
                     train_accuracy_list, validate_accuracy_list,
                     train_auc_list, validate_auc_list,
                     train_loss_list, validate_loss_list,
                     fold):
        assert os.path.exists(self.results_path), f"Expected '{self.results_path}' to exist"

        # Set up a 1x2 grid of subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2)
        fig_size = fig.get_size_inches()
        fig.set_figwidth(fig_size[0] * 2)

        # Plot the training & validation accuracy & AUCdata on the left
        ax_left.plot(train_accuracy_list, label="Training Acc")
        ax_left.plot(train_auc_list, label="Training AUC")
        ax_left.plot(validate_accuracy_list, label="Validation Acc")
        ax_left.plot(validate_auc_list, label="Validation AUC")
        # Plot the max accuracy location in the graph
        ax_left.plot(max_validate_accuracy_epoch, max_validate_accuracy, label="Max Val Acc", marker=".", markersize=10)
        ax_left.set_title(f"Test Acc: {test_accuracy:0.4f}, Test AUC: {test_auc:0.4f}, Max Val Acc: {max_validate_accuracy:0.4f}")
        ax_left.set(xlabel='Epoch', ylabel=f'Accuracy %')
        ax_left.legend()

        # Plot the training & validation loss data on the right
        ax_right.plot(train_loss_list, label="Training")
        ax_right.plot(validate_loss_list, label="Validation")
        # Plot the minimum loss location in the graph
        ax_right.plot(min_validate_loss_epoch, min_validate_loss, label="Min Val Loss", marker=".", markersize=10)
        ax_right.set_title(f"Min Val Loss: {min_validate_loss:0.4f}")
        ax_right.set(xlabel='Epoch', ylabel='Loss')
        ax_right.legend()

        # Create a path to the .png file with the stats for this model in the name.
        # e.g. "[5, 5]-0.001-2-result.png"
        plot_file_path = self.plot_results_path(fold)
        plt.tight_layout()
        # Save the plot image to disk
        plt.savefig(plot_file_path)
        # Clear the figure & axis for the next plot since pyplot holds state
        # instead of reusing figures etc.
        plt.clf()
        plt.cla()
        plt.close(fig)

        return plot_file_path

    def save_model(self):
        assert os.path.exists(self.results_path), f"Expected '{self.results_path}' to exist"

        # Create a path to the .pt file with the stats for this model in the name.
        # e.g. "[5, 5]-0.001-300-state_dict.pt"
        state_dict_file_path = os.path.join(self.results_path,
                                            f"{self.model.n_hiddens_list}-{self.learning_rate}-state_dict.pt")
        # Save the state dictionary to disk
        torch.save(self.model.state_dict(), state_dict_file_path)

    def load_best_model(self):
        # Load the state dictionary that produced the lowest loss during
        # training back into the model, so it will use those weights and
        # hopefully produce optimal results.
        assert self.best_state_dict is not None, "best_state_dict should be populated by now."
        self.model.load_state_dict(self.best_state_dict)

    def save_best_model(self):
        self.best_state_dict = copy.deepcopy(self.model.state_dict())

    def cross_validation(self, num_epochs, n_hidden_list, learning_rate):

        test_accuracy_sum = 0
        test_auc_sum = 0
        test_precision_sum = 0
        test_recall_sum = 0
        test_f1_sum = 0
        test_conf_matrix_sum = np.array([[0, 0], [0, 0]], dtype=float)
        fold_stats_str_list = []
        results = {}
        plot_img_paths = []

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(self.k_fold.split(self.x_train_tensor.cpu(), self.y_train_tensor.cpu())):

            # Build subsets of self.train_dataset for the specified indexes
            train_subdataset = torch.utils.data.Subset(self.train_dataset, train_ids)
            test_subdataset = torch.utils.data.Subset(self.train_dataset, test_ids)

            # Define data loaders for training and testing data in this fold
            train_batch_size = kBatchSize if kBatchSize != 0 else len(train_ids)
            train_dataloader = DataLoader(train_subdataset, batch_size=train_batch_size)
            test_batch_size = kBatchSize if kBatchSize != 0 else len(test_ids)
            test_dataloader = DataLoader(test_subdataset, batch_size=test_batch_size)

            (test_accuracy,
             test_auc,
             test_precision,
             test_recall,
             test_f1,
             test_conf_matrix,
             fold_stats_str,
             plot_img_path) = self.run(train_dataloader,
                                       test_dataloader,
                                       num_epochs,
                                       n_hidden_list,
                                       learning_rate,
                                       fold)

            test_accuracy_sum += test_accuracy
            test_auc_sum += test_auc
            test_precision_sum += test_precision
            test_recall_sum += test_recall
            test_f1_sum += test_f1
            test_conf_matrix_sum += test_conf_matrix
            fold_stats_str_list.append(fold_stats_str)
            plot_img_paths.append(plot_img_path)

            results.update({ fold : (test_accuracy,
                                     test_auc,
                                     test_precision,
                                     test_recall,
                                     test_f1,
                                     test_conf_matrix,
                                     fold_stats_str) })

        # combine the fold images vertically into a single image
        images = [Image.open(x) for x in plot_img_paths]
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        # Save the combo image into a new file
        new_im.save(self.plot_results_path())
        # delete the old image files
        [os.remove(x) for x in plot_img_paths]

        # Return the average of each measure
        n_folds = len(fold_stats_str_list)
        return { "accuracy" : test_accuracy_sum / n_folds,
                 "area_under_curve" : test_auc_sum / n_folds,
                 "precision" : test_precision_sum / n_folds,
                 "recall" : test_recall_sum / n_folds,
                 "f1" : test_f1_sum / n_folds,
                 "conf_matrix" : test_conf_matrix_sum / n_folds,
                 "fold_stats_str_list" : fold_stats_str_list,
                 "results" : results }

    def run(self, train_dataloader, test_dataloader,
            num_epochs, n_hidden_list, learning_rate,
            fold=None):

        start_time = time.time()

        # Start the model off fresh each run
        self.build_model(n_hidden_list, learning_rate)

        train_loss_list, train_accuracy_list, train_auc_list = [], [], []
        validate_loss_list, validate_accuracy_list, validate_auc_list = [], [], []
        min_validate_loss = float("inf")
        min_validate_loss_epoch = -1
        max_validate_accuracy = 0
        max_validate_accuracy_epoch = -1

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            # Kick off the training run for this epoch
            (train_loss, train_accuracy, train_auc) = self.train_epoch(train_dataloader)
            # Validate the training so far
            (validate_loss, validate_accuracy, validate_auc) = self.validate_epoch()

            # Keep track of the losses, accuracies & areas under the curve from
            # each training & validation phase
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            train_auc_list.append(train_auc)
            validate_loss_list.append(validate_loss)
            validate_accuracy_list.append(validate_accuracy)
            validate_auc_list.append(validate_auc)

            # Keep track of the max accuracy during validation for plotting purposes
            if max_validate_accuracy < validate_accuracy:
                max_validate_accuracy = validate_accuracy
                max_validate_accuracy_epoch = epoch

            # Keep track of the minimum loss during validation, and its corresponding
            # model state dictionary. It will be loaded back into the model for testing.
            # This helps prevent over fitting.
            if validate_loss < min_validate_loss:
                min_validate_loss = validate_loss
                min_validate_loss_epoch = epoch
                self.save_best_model()

            # If the max validation accuracy hasn't improved in a while then bail out.
            # The model has started to overfit and likely will not improve if we continue.
            global kEarlyStopPatience
            if 50 < epoch and min_validate_loss_epoch + kEarlyStopPatience < epoch:
                break

        # Load the best model that was generated during training in order
        # to (hopefully) produce the best testing results.
        self.load_best_model()

        # Run the test phase with the newly trained model.
        (test_conf_matrix,
         test_accuracy,
         test_auc,
         test_precision,
         test_recall,
         test_f1) = self.test_epoch(test_dataloader)
        elapsed_time = timedelta(seconds=time.time() - start_time)

        # Plot the accuracies & losses
        plot_img_path = self.plot_results(test_accuracy, test_auc,
                                          min_validate_loss, min_validate_loss_epoch,
                                          max_validate_accuracy, max_validate_accuracy_epoch,
                                          train_accuracy_list, validate_accuracy_list,
                                          train_auc_list, validate_auc_list,
                                          train_loss_list, validate_loss_list,
                                          fold)
        fold_stats_str = f"fold: {fold} -- Accuracy: {test_accuracy:0.4f}, AUC: {test_auc:0.4f}, Min Val Loss: {min_validate_loss:0.4f}, Min Val Loss Epoch: {min_validate_loss_epoch}, Elapsed Time: {elapsed_time}"
        print(fold_stats_str)

        return (test_accuracy,
                test_auc,
                test_precision,
                test_recall,
                test_f1,
                test_conf_matrix,
                fold_stats_str,
                plot_img_path)

    # Called once per epoch
    def train_epoch(self, train_dataloader):
        # [TODO] PERF: Pre-allocate these tensors at len(train_dataloader.dataset) and copy the
        # data into it. Its more efficient than concatenating tensors over and over again.
        y_train_tensor = None
        y_pred_prob_tensor = None   # probability the prediction is correct
        loss_tensor = None

        # Within each epoch process the training data in batches
        self.model.train()
        for x_train_batch, y_train_batch in train_dataloader:
            # Clear any gradiants from previous iterations
            self.optimizer.zero_grad()  # Clearing all previous gradients, setting to zero
            # Run x_validate_batch through the model
            y_pred_prob = self.model(x_train_batch)  # Forward Propagation - probability between 0 & 1
            # Calculate the loss for this batch
            loss = self.loss_func(y_pred_prob, y_train_batch)  # Loss Computation

            # Keep a list of all the y values with corresponding predictions & losses
            if y_train_tensor is None:
                y_train_tensor = y_train_batch
                y_pred_prob_tensor = y_pred_prob
                loss_tensor = loss.reshape(1)
            else:
                y_train_tensor = torch.concat((y_train_tensor, y_train_batch))
                y_pred_prob_tensor = torch.concat((y_pred_prob_tensor, y_pred_prob))
                loss_tensor = torch.concat((loss_tensor, loss.reshape(1)))

            loss.backward()             # Back Propagation
            self.optimizer.step()       # Updating the parameters

        # Calculate the loss, accuracy & area under the curve for this epoch
        y_train = y_train_tensor.cpu().detach().numpy()
        loss_value = torch.mean(loss_tensor).item()
        accuracy = accuracy_score(y_train, torch.round(y_pred_prob_tensor).cpu().detach().numpy())
        auc = roc_auc_score(y_train, y_pred_prob_tensor.cpu().detach().numpy())

        return loss_value, accuracy, auc

    # Called once per epoch
    def validate_epoch(self):
        # [TODO] PERF: Pre-allocate these tensors at len(self.validate_dataloader.dataset) and copy the
        # data into it. Its more efficient than concatenating tensors over and over again.
        y_pred_prob_tensor = None   # probability the prediction is correct
        loss_tensor = None

        # Since we don't need the model to back propagate the gradients during
        # validation use torch.no_grad() to reduce memory usage and speed up computation
        self.model.eval()
        with torch.no_grad():
            for x_validate_batch, y_validate_batch in self.validate_dataloader:
                # Run x_validate_batch through the model
                y_pred_prob = self.model(x_validate_batch)      # probability between 0 & 1
                # Calculate the loss for this batch
                loss = self.loss_func(y_pred_prob, y_validate_batch)

                # Keep a list of all the y values, corresponding predictions & losses
                if y_pred_prob_tensor is None:
                    y_pred_prob_tensor = y_pred_prob
                    loss_tensor = loss.reshape(1)
                else:
                    y_pred_prob_tensor = torch.concat((y_pred_prob_tensor, y_pred_prob))
                    loss_tensor = torch.concat((loss_tensor, loss.reshape(1)))

        # Calculate the loss, accuracy & area under the curve for this epoch
        y_validate = self.y_validate_tensor.cpu().detach().numpy()
        loss_value = torch.mean(loss_tensor).item()
        accuracy = accuracy_score(y_validate, torch.round(y_pred_prob_tensor).cpu().detach().numpy())
        auc = roc_auc_score(y_validate, y_pred_prob_tensor.cpu().detach().numpy())

        return loss_value, accuracy, auc

    # Called after all the training & validation is complete
    def test_epoch(self, test_dataloader):
        # [TODO] PERF: Pre-allocate these tensors at len(test_dataloader.dataset) and copy the
        # data into it. Its more efficient than concatenating tensors over and over again.
        y_test_tensor = None
        y_pred_prob_tensor = None  # probability the prediction is correct

        # Since we don't need the model to back propagate the gradients in test
        # use torch.no_grad() to reduce memory usage and speed up computation
        self.model.eval()
        with torch.no_grad():
            for x_test_batch, y_test_batch in test_dataloader:
                # Run x_test_batch through the model
                y_pred_prob = self.model(x_test_batch)      # probability between 0 & 1

                # Keep a list of all the y values and corresponding predictions
                if y_test_tensor is None:
                    y_test_tensor = y_test_batch
                    y_pred_prob_tensor = y_pred_prob
                else:
                    y_test_tensor = torch.concat((y_test_tensor, y_test_batch))
                    y_pred_prob_tensor = torch.concat((y_pred_prob_tensor, y_pred_prob))

        # Move the tensors from gpu to cpu memory, and convert to numpy arrays
        y_test = y_test_tensor.cpu().detach().numpy()
        y_test_pred = torch.round(y_pred_prob_tensor).cpu().detach().numpy()
        y_test_pred_prob = y_pred_prob_tensor.cpu().detach().numpy()

        # Return the various cores
        return (confusion_matrix(y_test, y_test_pred),
                accuracy_score(y_test, y_test_pred),
                roc_auc_score(y_test, y_test_pred_prob),
                precision_score(y_test, y_test_pred),
                recall_score(y_test, y_test_pred),
                f1_score(y_test, y_test_pred))
