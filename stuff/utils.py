
import os


# Tensor.shape returns a Tensor.Size, which  prints a list of values. e.g. [x, y].
# numpy.shape & pd.DataFrame.shape return a tuple. e.g. (x, y).
# Mimic the tuple printing with a Tensor.Size.
def tensor_size_pretty_str(size):
    return f"({size[0]}, {size[1]})"


# Strips the extension off a file name or file path, if one exists
def strip_extension(file_name):
    return os.path.splitext(f"{file_name}")[0]


def calc_failure_rate(confusion_matrix):
    true_negative = confusion_matrix[0, 0]
    false_negative = confusion_matrix[1, 0]
    false_positive = confusion_matrix[0, 1]
    true_positive = confusion_matrix[1, 1]
    false_positive_rate = false_positive / (false_positive + true_negative)
    false_negative_rate = false_negative / (false_negative + true_positive)
    return false_positive_rate, false_negative_rate
