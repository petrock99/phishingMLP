# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
from datetime import timedelta
import matplotlib.pyplot as plt
import os
import sys
import time

from stuff.detector import PhishingDetector
from stuff.sorted_logger import SortedLogger
from stuff import utils

import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

kBatchSize = 0                          # 0 == no batching.
kNumKFolds = 5
kEarlyStopPatience = 150
kNumEpochs = 5000                       # High epochs because Early Stopping is supported
kHighAccuracyThreshold = 0.965          # 0 <-> 1.0
kSameValueInColumnThreshold = 0.95      # 0 <-> 1.0
kValidateDataRatio = 0.2                # 0 <-> 1.0
kUseGPU = torch.cuda.is_available()
kDatasetsPath = "./datasets"

# Define some default values
kDefaultCSVNameList = ["DS4Tan.csv"]
kDefaultLearningRateList = [0.01, 0.001, 0.0001]
kDefaultNumHiddenLists = [[50, 50], [100, 100],
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


def stats_str():
    epoch_str = f"\tNumber of Epochs:                 {kNumEpochs}\n"
    full_dataset_str = "Full Dataset"
    return f"-- Stats --\n" \
           f"\tBatch Size:                       {kBatchSize if kBatchSize != 0 else full_dataset_str}\n" \
           f"\tEarly Stop Patience:              {kEarlyStopPatience}\n" \
           f"\tCommon Column Value Threshold:    {kSameValueInColumnThreshold}\n" \
           f"\tNumber of Folds:                  {kNumKFolds}\n"\
           f"{'' if kValidateDataRatio > 0 else epoch_str}"


def write_metrics_to_disk(metrics_path, header, metrics):
    # Write the results to metrics_path. If metrics_path already exists this will overwrite it.
    with open(metrics_path, 'w') as fp:
        fp.write(f"{header}\n{stats_str()}\n")
        fp.write(f"-- Metrics in Accuracy Descending Order --\n")
        [fp.write(f"{i}\n") for _, i in metrics]


def main():
    # Build lists of parameters for the model factory to run through.
    args = parse_args()
    n_hidden_lists = [args.hidden_layers] if args.hidden_layers else kDefaultNumHiddenLists
    learning_rate_list = args.learning_rates if args.learning_rates else kDefaultLearningRateList
    csv_name_list = args.csv_names if args.csv_names else kDefaultCSVNameList

    start_time = time.time()
    print(f"\nUsing {'GPU' if kUseGPU else 'CPU'}")
    if len(csv_name_list) > 1:
        print(f"Datasets: {csv_name_list}")
    if len(learning_rate_list) > 1:
        print(f"Learning Rates: {learning_rate_list}")

    PhishingDetector.set_constants({ "kBatchSize" : kBatchSize,
                                     "kDatasetsPath" : kDatasetsPath,
                                     "kSameValueInColumnThreshold" : kSameValueInColumnThreshold,
                                     "kValidateDataRatio" : kValidateDataRatio,
                                     "kNumKFolds" : kNumKFolds,
                                     "kEarlyStopPatience" : kEarlyStopPatience,
                                     "kUseGPU" : kUseGPU })

    # Run through the list of parameters set up above to generate a series of models with
    # different configs to find the 'best' model for the data set.
    for csv_name in csv_name_list:
        high_scores = []
        metrics = []
        ds4_tran = PhishingDetector(csv_name)
        header_str = f"{ds4_tran.data_split_str(csv_name)}\n{stats_str()}"
        print(f"\n{header_str}")

        # Set up the metrics file and its initial text
        metrics_path = os.path.join(ds4_tran.results_path, f"{utils.strip_extension(csv_name)}-metrics.txt")
        logger = SortedLogger(metrics_path,
                              f"{header_str}\n"
                              f"-- Metrics in Accuracy Descending Order --\n")

        # Print where/how to view the metrics during processing
        # 'head -n 26' will print the first 26 lines of the file, which are the
        # metrics for the best run available.
        print(f"View metrics live via either:"
              f"\n\twatch head -n 26 \"{os.path.abspath(metrics_path)}\""
              f"\n\t\tor"
              f"\n\twhile :; do clear; head -n 26 \"{os.path.abspath(metrics_path)}\"; sleep 2; done")

        for n_hidden_list in n_hidden_lists:
            for learning_rate in learning_rate_list:
                cv_start_time = time.time()

                # Close any pyplot figures that may still be open from previous epochs
                plt.close('all')

                header_str = "\n****************************************\n"
                header_body_str = f"hidden layers: {n_hidden_list}, learning rate: {learning_rate}"
                print(f"{header_str}csv: '{csv_name}', {header_body_str}")

                results_dict = ds4_tran.cross_validation(kNumEpochs, n_hidden_list, learning_rate)

                test_accuracy = results_dict["accuracy"]
                test_auc = results_dict["area_under_curve"]
                test_precision = results_dict["precision"]
                test_recall = results_dict["recall"]
                test_f1 = results_dict["f1"]
                test_conf_matrix = results_dict["conf_matrix"]
                fold_stats_str_list = results_dict["fold_stats_str_list"]

                # Build & print the metrics string
                false_positive_rate, false_negative_rate = utils.calc_failure_rate(test_conf_matrix)
                cv_elapsed_time = timedelta(seconds=time.time() - cv_start_time)
                fold_stats_str = '\n'.join(fold_stats_str_list)
                metrics_str = f"Accuracy %:          {(test_accuracy * 100.0):0.4f}\n" \
                              f"Area Under Curve %:  {(test_auc * 100.0):0.4f}\n" \
                              f"Precision %:         {(test_precision * 100.0):0.4f}\n" \
                              f"Recall %:            {(test_recall * 100.0):0.4f}\n" \
                              f"F1 %:                {(test_f1 * 100.0):0.4f}\n" \
                              f"False Pos %:         {(false_positive_rate * 100.0):0.4f}\n" \
                              f"False Neg %:         {(false_negative_rate * 100.0):0.4f}\n" \
                              f"Elapsed Time:        {cv_elapsed_time}\n" \
                              f"Confusion Matrix:\n" \
                              f"{np.round(test_conf_matrix)}\n"

                # If the accuracy is above a threshold then consider it a good run
                if test_accuracy > kHighAccuracyThreshold:
                    # Only dump the metrics for the good runs to not clutter the console
                    print(metrics_str)
                    # Keep track of high performing configurations
                    high_scores.append([test_accuracy, test_auc, n_hidden_list, learning_rate])
                else:
                    print(f"Test accuracy below threshold. Accuracy: {test_accuracy}, Area Under Curve: {test_auc}, Elapsed Time: {cv_elapsed_time}")

                # Add the header to the metrics string
                metrics_str = f"{header_str}{header_body_str}\n" \
                              f"{fold_stats_str}\n\n" \
                              f"{metrics_str}"
                # Add metrics_str to the logger. It will sort all the metrics by test_accuracy
                # and write them to metrics_path so we can view the changhes live in a Terminal
                # window. See above comments about the 'watch' command.
                logger.add_str(test_accuracy, metrics_str)

        # Append the elapsed time to the logger file
        elapsed_time_str = f"Elapsed Time: {timedelta(seconds=time.time() - start_time)}\n\n"
        footer_str = f"\n\n------------------------------------------\n" \
                     f"{elapsed_time_str}"
        logger.set_footer(footer_str)

        # Sort high_scores by test_accuracy in descending order
        def sort_func(tuple): return tuple[0]
        high_scores.sort(reverse=True, key=sort_func)

        # print the high scores
        print("\n\n------------------------------------------")
        print(f"High Scores from '{csv_name}' in Accuracy Descending Order")
        [print(f"\t{i}") for i in high_scores]
        print(f"\n{elapsed_time_str}")

def parse_args():
    def float_range(min, max):
        def float_range_checker(arg):
            try:
                f = float(arg)
            except ValueError:
                raise argparse.ArgumentTypeError("Expected a floating point number")
            if f < min or f > max:
                raise argparse.ArgumentTypeError(f"Expected range of [{min} .. {max}]")
            return f

        # Return function handle to checking function
        return float_range_checker

    def unsigned_int(arg):
        try:
            i = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Expected an integral value")
        if i < 0:
            raise argparse.ArgumentTypeError("Expected a minimum value of 0")
        return i

    def csv_filename(arg):
        try:
            name = str(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Expected a string value")

        # Make sure its a .csv
        if os.path.splitext(name)[1] != ".csv":
            raise argparse.ArgumentTypeError(f"Expected .csv file name: {name}")

        # make sure all the .csv or .csv.zip file exist
        csv_path = os.path.join(kDatasetsPath, name)
        if not os.path.exists(csv_path):
            zip_path = f"{csv_path}.zip"
            if not os.path.exists(zip_path):
               raise argparse.ArgumentTypeError(f"No such file or directory: {csv_path} or {zip_path}")
        return name

    global kUseGPU, kBatchSize, kNumEpochs, \
           kEarlyStopPatience, kSameValueInColumnThreshold, kNumKFolds, \
           kValidateDataRatio, kHighAccuracyThreshold
    arg_parser = argparse.ArgumentParser(fromfile_prefix_chars='@')  # Supports putting arguments in a config file
    arg_parser.add_argument('--csv_names',
                            metavar='CSV_NAME',
                            type=csv_filename,
                            action='store',
                            help=f'One or more .csv names. Default: {kDefaultCSVNameList}',
                            nargs='+',
                            required=False)
    arg_parser.add_argument('--hidden_layers',
                            metavar='N_NODES',
                            type=unsigned_int,
                            action='store',
                            help='List of the number of nodes in each hidden layer',
                            nargs='+',
                            required=False)
    arg_parser.add_argument('--learning_rates',
                            metavar='LEARNING_RATE',
                            type=float_range(0,1),
                            action='store',
                            help=f"List of learning rates to train with, between 0 & 1. Default: {kDefaultLearningRateList}",
                            nargs='+',
                            required=False)
    arg_parser.add_argument('--force_cpu',
                            action='store_true',
                            help='Force running on the CPU instead of GPU',
                            default=False,
                            required=False)
    arg_parser.add_argument('--batch_size',
                            type=unsigned_int,
                            action='store',
                            help=f"Batch size used during processing. Zero will load the entire dataset in one batch. Default: {kBatchSize}",
                            required=False)
    arg_parser.add_argument('--epochs',
                            metavar='N_EPOCHS',
                            type=unsigned_int,
                            action='store',
                            help=f'Max number of epochs for training. Default: {kNumEpochs}',
                            required=False)
    arg_parser.add_argument('--validation_split',
                            metavar='SPLIT',
                            type=float_range(0, 1),
                            action='store',
                            help='The ratio of the dataset used for validation, and Early Stopping, during training. Remainder of the dataset will be used for training. Default: {kValidateDataRatio}',                            required=False)
    arg_parser.add_argument('--accuracy_threshold',
                            metavar='THRESHOLD',
                            type=float_range(0, 1),
                            action='store',
                            help='Minimum accuracy needed to print results to the console. Default: {kHighAccuracyThreshold}',
                            required=False)
    arg_parser.add_argument('--early_stop_patience',
                            metavar='PATIENCE',
                            type=unsigned_int,
                            action='store',
                            help=f"Number of epochs to wait for change before stopping training. Default: {kEarlyStopPatience}",
                            required=False)
    arg_parser.add_argument('--common_value_threshold',
                            metavar='THRESHOLD',
                            type=float_range(0, 1),
                            action='store',
                            help='Used to drop columns with the same value in the specified percentage of entries. Value between 0 & 1. Default: {kSameValueInColumnThreshold}',
                            required=False)
    arg_parser.add_argument('--k_folds',
                            metavar='FOLDS',
                            type=unsigned_int,
                            action='store',
                            help=f"Number of folders to use during k-fold validation. Default: {kNumKFolds}",
                            required=False)

    # Load any arguments passed to the script
    args = arg_parser.parse_args(sys.argv[1:])

    # Update global variable values from any arguments passed to the script
    if args.force_cpu:
        kUseGPU = False
    if args.batch_size is not None:
        kBatchSize = args.batch_size
    if args.epochs is not None:
        kNumEpochs = args.epochs
    if args.validation_split is not None:
        kValidateDataRatio = args.validation_split
    if args.accuracy_threshold is not None:
        kHighAccuracyThreshold = args.accuracy_threshold
    if args.early_stop_patience is not None:
        kEarlyStopPatience = args.early_stop_patience
    if args.common_value_threshold is not None:
        kSameValueInColumnThreshold = args.common_value_threshold
    if args.k_folds is not None:
        kNumKFolds = args.k_folds

    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
