import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    pred = pred.copy()

    tp = 0
    fp = 0
    match = False

    while len(pred) > 0:
        p = pred[0]
        pred.remove(p)

        for i in range(len(gold)):
            g = gold[i]

            if np.in1d(g,p).any():
                tp +=1
                gold = gold[:i] + gold[i+1:] #remove the gold signal (for efficiency)

                #remove any other signals in pred corresponding with the discovered signal (so they don't count as false positives)
                j = 0
                while j < len(pred):
                    p2 = pred[j]

                    if np.in1d(p2,g).any():
                        pred = pred[:j] + pred[j+1:] #remove() was bugging up
                    else:
                        j+=1

                match = True
                break

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

def threshold_testing(pred, true, thresholds, verbose=False):

    sample_rate = true[true.columns[0]].index[1] - true[true.columns[0]].index[0]

    f1s = np.zeros(len(thresholds))
    rs = np.zeros(len(thresholds))
    ps = np.zeros(len(thresholds))
    n_pred = np.zeros(len(thresholds))

    for i in range(len(thresholds)):
        t = thresholds[i]
        tps = 0
        fps = 0
        n_preds = 0
        n_golds = 0

        for j in range(len(true.columns)):
            house = true.columns[j]

            p = true.index[np.where(pred[house].flatten() > t)[0]]
            gold = true.index.where(true[house] > 0.5).dropna()

            p_list = divide_signal(p, min_length=2, sample_rate=sample_rate)
            gold_list = divide_signal(gold, min_length=2, sample_rate=sample_rate)

            tp,fp = positives(p_list, gold_list)

            tps += tp
            fps += fp

            n_preds += len(p_list)
            n_golds += len(gold_list)

        rs[i] = recall(None,np.zeros(n_golds),tp=tps,fp=fps)
        ps[i] = precision(None,np.zeros(n_golds),tp=tps,fp=fps)
        f1s[i] = 2*(rs[i]*ps[i]) / (rs[i]+ps[i]) if rs[i] + ps[i] > 0 else 0
        n_pred[i] = n_preds

        if verbose:
            print("threshold {} done".format(t))

    return {"f1": f1s, "recall": rs, "precision": ps, "n_pred": n_pred}

def plot_scores(score_dict, thresholds, out_file=None, model_name="", plot_n=False, grid=True, rotation=0):
    fig, ax1 = plt.subplots()
    ax1.grid(axis='both')
    plt.rc('axes', axisbelow=True)

    ax1.plot(thresholds, score_dict["f1"], label='F1 (max {0:.2f})'.format(score_dict["f1"].max()))
    ax1.plot(thresholds, score_dict["recall"], label='Recall')
    ax1.plot(thresholds, score_dict["precision"], label='Precision')
    ax1.set_ylabel('Scores')
    ax1.set_ylim(0,1)
    ax1.set_xlim(min(thresholds),max(thresholds))

    if plot_n:
        ax2 = ax1.twinx()

        ax2.bar(thresholds, score_dict["n_pred"], 0.1*max(thresholds), alpha=0.2, color='C0')
        ax2.set_ylabel('Number of signals discovered')

    xticks = ["{0:.2f}".format(t) for t in thresholds]
    plt.xticks(thresholds, xticks, rotation=rotation)
    ax1.set_xlabel("Threshold")
    ax1.set_title("{} scores".format(model_name))

    fig.legend(bbox_to_anchor=(0.93,0.90))
    fig.tight_layout()
    if out_file is not None:
        plt.savefig(out_file)

    plt.show()

def roc_curve(pred, true, thresholds, plot=False):
    sample_rate = true[true.columns[0]].index[1] - true[true.columns[0]].index[0]

    roc = pd.DataFrame(columns=thresholds)

    for i in range(len(thresholds)):
        t = thresholds[i]
        tps = 0
        fps = 0
        n_positives = 0
        n_negatives = 0

        for j in range(len(true.columns)):
            house = true.columns[j]

            p = true.index[np.where(pred[house].flatten() > t)[0]]
            gold = true.index.where(true[house] > 0.5).dropna()

            p_list = divide_signal(p, min_length=2, sample_rate=sample_rate)
            gold_list = divide_signal(gold, min_length=2, sample_rate=sample_rate)

            tp,fp = positives(p_list, gold_list)

            tps += tp
            fps += fp

            n_positives += len(gold_list)
            n_negatives += (len(true[house]) - len(gold))

        print(tps, fps)

        tp_rate = tps / n_positives if n_positives > 0 else 1
        fp_rate = fps / n_negatives if n_negatives > 0 else 1

        roc[t] = [tp_rate,fp_rate]

    if plot:
        plt.plot(roc.iloc[1],roc.iloc[0])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.show()

    return roc
