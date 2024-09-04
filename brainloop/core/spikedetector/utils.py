import numpy as np
import random
import torch
import datetime
import shutil
from pathlib import Path


def confusion_matrix(preds, labels):
    if len(preds) != len(labels):
        raise ValueError("len(pred) != len(labels)")

    confusion_array = np.array([0, 0, 0, 0])  # FN, TN, FP, TP
    for p, l in zip(preds, labels):
        confusion_i = int(2*p + (p == l))
        confusion_array[confusion_i] += 1
    return confusion_array


def confusion_stats(confusion_matrix):
    fn, tn, fp, tp = confusion_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = (tp + tn) / (fn + tn + fp + tp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    return [accuracy*100, recall*100, precision*100]  # *100 for stats in percent


def random_seed(seed, silent=False):
    if not silent:
        print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_time():
    """
    Gets the time in the string format of yymmdd_HHMMSS_ffffff (https://www.programiz.com/python-programming/datetime/strftime)
    Ex: If run at 9/26/22 at 3:53pm and 30 seconds and 1 microsecond, yymmdd_HHMMSS_ffffff = 220926_155330_000001

    :return: str
        Formatted time
    """

    return datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")


def copy_file(src_path, dest_folder):
    """
    Copies file at src_path and stores in dest_folder

    :param src_path: Path or str
        Path of source file
    :param dest_folder: Path or str
        Path to folder in which source file is saved
        The copied file has the same name as the source file
        Ex) test.py saved on 9/26/22 at 1:30pm will be saved as 220926_1330_test.py
    """

    shutil.copyfile(src_path, Path(dest_folder) / Path(src_path).name)
    # print(f"Saved a copy of script to {copied_path}")


def round(n):
    """
    Rounds a float (n) to the nearest integer
    Uses standard math rounding, i.e. 0.5 rounds up to 1
    """

    if isinstance(n, np.ndarray):
        n_int = n.astype(int)
        return n_int + ((n - n_int).astype(int) * 2)
    else:
        n_int = int(n)
        return n_int + int((n - n_int) * 2)


def torch_to_np(tensor):
    # Convert a possible torch tensor to np
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor
