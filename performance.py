import numpy as np
import pandas as pd


def divide_signal(signal,sample_rate=1,min_length=1):
    """
    Divides a 1d array of signal times into a list of signals.
    """
    gaps = []
    s_list = []
    for i in range(1,len(signal)):
        if signal[i] - signal[i-1] > sample_rate:
            gaps.append(i)
    i = 0
    for g in gaps:
        if len(signal[i:g]) >= min_length:
            s_list.append(signal[i:g])
        i = g
    if len(signal) >= min_length:
        s_list.append(signal[i:])

    return s_list


def positives(pred,gold):
    """
    Assumes input in the form of lists of signals
    """
    gold = gold.copy() #redefine scope


    tp = 0
    fp = 0
    match = False

    discovered_signals = []

    for p in pred:
        for i in range(len(gold)):
            g = gold[i]

            if np.in1d(g,p).any():
                if i not in discovered_signals:
                    tp +=1
                    discovered_signals.append(i)

                match = True

        fp += 0 if match else 1
        match = False

    return tp, fp

def precision(pred, gold, tp=None, fp=None):
    if tp is None or fp is None:
        tp, fp = positives(pred,gold)

    return tp / (tp + fp) if (tp + fp) > 0 else 1


def recall(pred, gold, tp=None, fp=None):
    if tp is None or fp is None:
        tp, fp = positives(pred,gold)

    return tp / len(gold) if len(gold) > 0 else 1

def f1(pred,gold,sample_rate=1,signal_min_length=1):
    """
    returns precision, recall, and f1
    pred: the predicted times of signal activity
    gold: the actual times of signal activity
    """

    pred_list = divide_signal(pred,sample_rate=sample_rate,min_length=signal_min_length)
    gold_list = divide_signal(gold,sample_rate=sample_rate,min_length=signal_min_length)

    tp, fp = positives(pred_list,gold_list)

    r = recall(pred_list,gold_list,tp,fp)
    p = precision(pred_list,gold_list,tp,fp)
    f1 = 2*(r*p) / (r+p) if r + p > 0 else 0

    return {"precision" : p, "recall" : r, "f1" : f1}
