import numpy as np
import pandas as pd
from scipy import signal as scipsign
import importlib.util
from matplotlib import pyplot as plt

spec = importlib.util.spec_from_file_location("performance", "../scripts/performance.py")
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


"""
Assumes timestamped signal

"""

def correlate(filter, signal, normalize=True):
    xcorr = scipsign.correlate(signal.values, filter, mode='same')
    if normalize:
        xcorr /= np.max(xcorr) if np.max(xcorr) > 0 else 1

    return xcorr

def locate_signals(correlation, timeline, threshold):

    pred = timeline[np.where(correlation > threshold)]

    return pred

def match_filters(filters, signal, threshold):
    """
    returns the estimated signal times
    """
    signature_times = []

    for i in range(len(filters)):
        length_n_filters = filters[i]

        for filter in length_n_filters:
            correlation = correlate(filter, signal)
            signature_times.extend(locate_signals(correlation, signal.index, threshold))

    signature_times = pd.DatetimeIndex(np.unique(signature_times))

    return signature_times

def n_filters(filters):
    """
    returns the size of a 2D list
    """
    return np.sum([len(length_n_filters) for length_n_filters in filters])


def get_house_correlations(filters, signal, normalize=True):

    correlations = np.zeros(len(signal))


    for i in range(len(filters)):
        length_n_filters = filters[i]

        for filter in length_n_filters:
            correlation = correlate(filter, signal, normalize=normalize)
            update = np.where(correlation > correlations)[0]
            correlations[update] = correlation[update]


    return correlations

def get_correlations(filters, signals, normalize=True):
    """
    returns correlations from multiple houses and filters
    """

    correlations = {}
    filters = np.array(filters)

    for house in signals.columns:
        house_signal = signals[house]
        corrs = get_house_correlations(filters, house_signal, normalize=normalize)
        correlations[house] = corrs

    return correlations

def test_filters(filters, signal, gold, threshold, signal_min_length=1):
    sample_rate = signal.index[1] - signal.index[0]
    signature_times = match_filters(filters, signal, threshold)
    gold = gold.index.where(gold > threshold).dropna()

    filter_p = performance.f1(signature_times, gold, sample_rate=sample_rate,signal_min_length=signal_min_length)

    return filter_p

def test_filters_multiple(filters, signal, gold, threshold, signal_min_length=1):
    sample_rate = signal.index[1] - signal.index[0]
    signature_times = match_filters(filters, signal, threshold)

    gold = gold.index.where(gold > 0.5).dropna()

    pred_list = performance.divide_signal(signature_times, sample_rate=sample_rate, min_length=signal_min_length)
    gold_list = performance.divide_signal(gold, sample_rate=sample_rate, min_length=signal_min_length)

    n_gold = len(gold_list)
    n_pred = len(pred_list)

    tp, fp = performance.positives(pred_list,gold_list)

    return tp, fp, n_gold, n_pred

def test_thresholds(filters, signal, gold, thresholds, signal_min_length=1):
    """
    Returns a matrix for each measure where each row corresponds to a filter and each column
    corresponds to a threshold.
    """

    f1s = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))
    precisions = np.zeros(len(thresholds))

    for i in range(len(thresholds)):
        t = thresholds[i]
        threshold_p = test_filters(filters, signal, gold, t, signal_min_length=signal_min_length)
        precisions[i] = threshold_p["precision"]
        recalls[i] = threshold_p["recall"]
        f1s[i] = threshold_p["f1"]

    return {"f1" : f1s, "recall" : recalls, "precision" : precisions}

def test_thresholds_multiple(filters, signal, gold, thresholds, signal_min_length=1, verbose=False):
    """
    signal: a dataframe with each column representing time series of measurements.
    """

    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))
    f1s = np.zeros(len(thresholds))
    n_preds = np.zeros(len(thresholds))

    for i in range(len(thresholds)):
        t = thresholds[i]
        tps = 0
        fps = 0
        n_gold = 0
        if verbose:
            print("Running threshold ", thresholds[i])
        for house in signal.columns:
            tp, fp, n_g, n_pred = test_filters_multiple(filters, signal[house], gold[house], t, signal_min_length=signal_min_length)
            tps += tp
            fps += fp
            n_gold += n_g
            n_preds[i] += n_pred


        precisions[i] = performance.precision(None,np.zeros(n_gold),tps,fps)
        recalls[i] = performance.recall(None,np.zeros(n_gold),tps,fps)
        f1s[i] = 2*(recalls[i]*precisions[i]) / (recalls[i]+precisions[i]) if recalls[i] + precisions[i] > 0 else 0

    return {"f1" : f1s, "recall" : recalls, "precision" : precisions, "n_pred": n_preds}
