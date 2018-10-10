import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import importlib.util

spec = importlib.util.spec_from_file_location("performance", "../scripts/performance.py")
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)

spec2 = importlib.util.spec_from_file_location("pattern_matching", "../scripts/pattern_matching.py")
pattern_matching = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(pattern_matching)

spec3 = importlib.util.spec_from_file_location("data_preparation", "../scripts/data_preparation.py")
data_preparation = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(data_preparation)


def divide_by_length(signals,significance=100):

    """
    Returns a (sorted) list of lists, each sublist containing signals of a
    certain length, given that the number of such signals exceed significance.
    """
    signals_by_length = []

    for i in range(np.max([len(s) for s in signals])):
        length_n_signals = [s for s in signals if len(s) == i]
        if len(length_n_signals) > significance:
            signals_by_length.append(length_n_signals)
            print("No. length {} signals: {}".format(i,len(length_n_signals)))
    return signals_by_length

def calculate_k(signals_by_length, threshold=0.02, max_clusters=10000):
    """
    Calculates the optimal value for k given a set of signals by clustering for
    each k from 2 and calculating the silhouette score. The clustering where a
    significant loss in silhouette score is still acheived by increasing k is
    chosen.
    """

    silhouette_means = [2]

    for k in range(2,max_clusters):
        silhouette_scores = []
        #Generate clusterings
        for length_n_signals in signals_by_length:
            clustering = KMeans(n_clusters=k)
            clustering.fit(length_n_signals)
            silhouette_scores.append(silhouette_score(length_n_signals, clustering.labels_))

        #Calculate mean silhouette score
        silhouette_means.append(np.mean(silhouette_scores))


        #Compare silhouette score with previous silhouette score
        if silhouette_means[-2] - silhouette_means[-1] < threshold:
            return k-1

    return k

def calculate_single_k(length_n_signals, threshold=0.02, max_clusters=10000):
    """
    Calculates the optimal value for k given a set of signals of the same length by clustering for
    each k from 2 and calculating the silhouette score. The clustering where a
    significant loss in silhouette score is still acheived by increasing k is
    chosen.
    """

    silhouette_scores = [2]

    for k in range(2,max_clusters):

        #Generate clusterings
        clustering = KMeans(n_clusters=k)
        clustering.fit(length_n_signals)
        silhouette_scores.append(silhouette_score(length_n_signals, clustering.labels_))

        #Compare silhouette score with previous silhouette score
        if silhouette_scores[-2] - silhouette_scores[-1] < threshold:
            return k-1

    return k

def create_templates(signal_file, template_column='car1', significance=100, silhouette_threshold=0.02, max_templates=10000, return_library=False, separate_ks=False):
    """
    Creates typical templates from a list of signals by creating clusterings for
    each signal length. Returns a list of length equal to the number of signal
    lengths, each consisting of k templates, where k is automatically calculated
    using the silhouette score (see calcuate_k()).
    """
    all_signals = data_preparation.read_dataport_file(signal_file, template_column)
    signals = []
    for house in all_signals.columns:
        signals.extend(data_preparation.find_signals(all_signals[house], return_values=True))

    signals_by_length = divide_by_length(signals, significance=significance)

    n_lengths = len(signals_by_length)
    if not separate_ks:
        k = calculate_k(signals_by_length, threshold=silhouette_threshold, max_clusters=max_templates)

        templates = []

        for i in range(n_lengths):
            length_n_templates = []

            clustering = KMeans(n_clusters=k)
            clustering.fit(signals_by_length[i])

            for cluster in range(k):
                cluster_indexes = np.where(clustering.labels_ == cluster)[0]
                cluster_signals = np.array([signals_by_length[i][c] for c in cluster_indexes])
                cluster_mean = np.mean(cluster_signals, axis=0)
                length_n_templates.append(cluster_mean)

            templates.append(length_n_templates)
    else:
        templates = []

        for i in range(n_lengths):
            length_n_templates = []
            k = calculate_single_k(signals_by_length[i], threshold=silhouette_threshold, max_clusters=max_templates)
            clustering = KMeans(n_clusters=k)
            clustering.fit(signals_by_length[i])

            for cluster in range(k):
                cluster_indexes = np.where(clustering.labels_ == cluster)[0]
                cluster_signals = np.array([signals_by_length[i][c] for c in cluster_indexes])
                cluster_mean = np.mean(cluster_signals, axis=0)
                length_n_templates.append(cluster_mean)

            templates.append(length_n_templates)

    if return_library:
        return templates, signals_by_length

    return templates
