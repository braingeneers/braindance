PATH_REC_DL = "/data/MEAprojects/DLSpikeSorter/{}/data.raw.h5"
PATH_REC_SI = "/data/MEAprojects/dandi/000034/sub-mouse412804/sub-mouse412804_ecephys.nwb"

PATH_SM4_DL = "/data/MEAprojects/DLSpikeSorter/{}/spikesort_matlab4"
PATH_SM4_SI = "/data/MEAprojects/dandi/000034/sub-mouse412804/spikesort_matlab4"


import numpy as np
import pickle


def find_closest(arr, v):
    # arr must be sorted
    idx = np.searchsorted(arr, v)
    if idx == 0:
        closest = arr[idx]
    elif idx >= len(arr) - 1:
        closest = arr[-1]
    else:
        left = arr[idx-1]
        right = arr[idx]
        if right - v < v - left:
            closest = right
        else:
            closest = left

    return idx, closest


def calc_f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def rec_si():
    from src.recording import Recording
    return Recording(PATH_REC_SI)


def kilosort_first_si():
    # first = first curation
    from src.sorters.kilosort import Kilosort
    return Kilosort(PATH_SM4_SI, "first", rec_si())


def chans_rms_probe_c():
    return np.load("/data/MEAprojects/buzsaki/SiegleJ/AllenInstitute_744912849/session_766640955/probe_773592320/chans_rms.npy")


def chans_rms_si():
    return np.load("/data/MEAprojects/dandi/000034/sub-mouse412804/chans_rms.npy")


def calc_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def pickle_dump(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
        
def pickle_load(path):
    with open(path, "rb") as file:
        return pickle.load(file)
