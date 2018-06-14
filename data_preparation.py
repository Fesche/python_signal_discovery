import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("performance", "../scripts/performance.py")
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


def find_signals(signal, threshold=0.5, return_values=False):
    """
    Returns timestamps of when the signal is active.
    Active is defined as anywhere the signal has a magnitude above threshold.
    """
    if signal.ndim > 1:
        if return_values:
            signal_index = np.where(np.sum(signal,axis=1) > threshold)
            signal_indexes = performance.divide_signal(signal_index)
            signals = []

            for s in signal_indexes:
                signals.append(signal.values[s])

            return signals

        return signal.index[np.where(np.sum(signal,axis=1) > threshold)]

    else:
        if return_values:
            signal_index = np.where(signal > threshold)[0]
            signal_indexes = performance.divide_signal(signal_index)
            signals = []

            for s in signal_indexes:
                signals.append(signal.values[s])

            return signals

        return signal.index[np.where(signal > threshold)]


def find_intervals(data_series, sample_rate=None, window_size=24, return_index=True):
    """
    takes a pandas timeseries with gaps and returns a list of intervals
    containing successive samples.
    """

    intervals = []

    if sample_rate is None:
        sample_rate = data_series.index[1] - data_series.index[0]

    current_interval = [data_series.index[0]]

    for i in range(1,len(data_series)):
        if data_series.index[i] - current_interval[-1] > sample_rate:
            if len(current_interval) > window_size:
                if return_index:
                    intervals.append(data_series.loc[current_interval].copy())
                else:
                    intervals.append(data_series.loc[current_interval].values)
                current_interval = [data_series.index[i]]
        else:
            current_interval.append(data_series.index[i])

    return intervals

def discrete_output_training_set(aggregate, submeter, threshold=0.5, window_size=24, step_size=None):
    """
    Makes a training set consisting of aggregate windows and labeled responses.
    The responses are the same size as the inputs, with 0s where there is no
    active signal, and 1s where there is an active signal.

    NOTE:
    Should the data be cleaned? e.g. by removing signals of length 1?
    """

    N = len(aggregate)


    if step_size is None:
        step_size = int(window_size/5)

    signal_times = find_signals(submeter, threshold=threshold)

    labels = pd.DataFrame(np.zeros(len(submeter)), index=submeter.index)

    labels.loc[signal_times] += 1

    X = np.zeros((N-window_size,window_size))
    y = np.zeros((N-window_size,window_size))

    for i in range(0,N-window_size):
        X[i] = aggregate[i:i+window_size]
        y[i] = labels[i:i+window_size].values.flatten()

    return {"X" : X, "y": y}

def continuous_output_training_set(aggregate, submeter, threshold=0.5, window_size=24, step_size=None):
    """
    Makes a training set consisting of aggregate windows and the isolated submeter signal.
    """

    N = len(aggregate)


    if step_size is None:
        step_size = int(window_size/5)

    signal_times = find_signals(submeter, threshold=threshold)

    labels = submeter

    X = np.zeros((N-window_size,window_size))
    y = np.zeros((N-window_size,window_size))

    for i in range(0,N-window_size):
        X[i] = aggregate[i:i+window_size]
        y[i] = labels[i:i+window_size].values.flatten()

    return {"X" : X, "y": y}


def syntethic_discrete_output_training_set(aggregate, submeter, n_data, threshold=0.5, window_size=24, verbose=False):
    """
    Creates n_data syntethic data points in the discrete output format by
    embedding submeter signals in aggregate signals.
    """

    synth_X = np.zeros((1,window_size))
    synth_y = np.zeros((1,window_size))

    sample_rate = aggregate.index[1] - aggregate.index[0]

    signal_times = find_signals(submeter)

    if len(signal_times) == 0:
        return {'X' : synth_X, 'y' : synth_y}

    signals = submeter.loc[signal_times]
    signals = find_intervals(signals, window_size=0, sample_rate=sample_rate, return_index=False)

    clean_aggregate = aggregate.drop(signal_times)
    clean_intervals = find_intervals(clean_aggregate, window_size=window_size, sample_rate=sample_rate)
    n_intervals = len(clean_intervals)

    if n_intervals < (n_data / 10):
        return {'X' : synth_X, 'y' : synth_y}

    for i in range(n_data):

        #pick a random interval
        interval = clean_intervals[np.random.randint(0,n_intervals)].copy()
        y = np.zeros(window_size)

        #pick a random window from interval
        d = np.random.randint(len(interval) - window_size)
        window = interval[d:d+window_size].values

        #Generates empty samples with probability 0.5
        if np.random.random() > 0.5:
            #pick a random signal
            d = np.random.randint(0,len(signals))
            signal = signals[d]

            #pick a random placement position in the window
            d = np.random.randint(-len(signal), window_size)
            if d < 0:
                if -d < len(signal):
                    signal = signal[-d:]
                    window[0:len(signal)] += signal
                    y[0:len(signal)] += 1

            elif d + len(signal) > window_size:
                signal = signal[:window_size - (d + len(signal))]
                window[d:] += signal
                y[d:] += 1

            else:
                window[d:d+len(signal)] += signal
                y[d:d+len(signal)] += 1

        synth_X = np.vstack((synth_X,window.flatten()))
        synth_y = np.vstack((synth_y,y))

    return {'X' : synth_X, 'y' : synth_y}


def syntethic_continuous_output_training_set(aggregate, submeter, n_data, threshold=0.5, window_size=24, verbose=False):
    """
    Creates n_data syntethic data points in the discrete output format by
    embedding submeter signals in aggregate signals.
    """

    synth_X = np.zeros((1,window_size))
    synth_y = np.zeros((1,window_size))

    sample_rate = aggregate.index[1] - aggregate.index[0]

    signal_times = find_signals(submeter)

    if len(signal_times) == 0:
        return {'X' : synth_X, 'y' : synth_y}

    signals = submeter.loc[signal_times]
    signals = find_intervals(signals, window_size=0, sample_rate=sample_rate, return_index=False)

    clean_aggregate = aggregate.drop(signal_times)
    clean_intervals = find_intervals(clean_aggregate, window_size=window_size, sample_rate=sample_rate)
    n_intervals = len(clean_intervals)
    if n_intervals < (n_data / 10):
        return {'X' : synth_X, 'y' : synth_y}

    for i in range(n_data):

        #pick a random interval
        interval = clean_intervals[np.random.randint(0,n_intervals)].copy()
        y = np.zeros(window_size)

        #pick a random window from interval
        d = np.random.randint(len(interval) - window_size)
        window = interval[d:d+window_size].values

        #Generates empty samples with probability 0.5
        if np.random.random() > 0.5:
            #pick a random signal
            d = np.random.randint(0,len(signals))
            signal = signals[d]

            #pick a random placement position in the window
            d = np.random.randint(-len(signal), window_size)
            if d < 0:
                if -d < len(signal):
                    signal = signal[-d:]
                    window[0:len(signal)] += signal
                    y[0:len(signal)] += signal

            elif d + len(signal) > window_size:
                signal = signal[:window_size - (d + len(signal))]
                window[d:] += signal
                y[d:] += signal

            else:
                window[d:d+len(signal)] += signal
                y[d:d+len(signal)] += signal

        synth_X = np.vstack((synth_X,window.flatten()))
        synth_y = np.vstack((synth_y,y))

    return {'X' : synth_X, 'y' : synth_y}

def read_dataport_file(filepath, column, upsample=None, downsample=None):
    """
    Returns data indexed by datetime, with each column corresponding to a house.
    Resample needs to be a valid code for the pd.resample function, e.g.
    '15m'.

    column: which column to read
    """

    data = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True)
    data = data.sort_index()

    data_n = pd.DataFrame(columns=np.unique(data['dataid']), index=np.unique(data.index))

    for house in data_n.columns:
        data_n[house] = data[column].where(data['dataid'] == house).dropna()

    if upsample is not None:
        data_n.resample(upsample).bfill()

    if downsample is not None:
        data_n.resample(downsample).mean()

    return data_n


def make_training_set(aggregate_data_file, submeter_data_file, type='discrete', n_synth=0, window_size=24, submeter_threshold=0.5, aggregate_col='use', submeter_col='car1', verbose=False, window_step=None, output_file_X='training_set_X.csv', output_file_y='training_set_y.csv'):
    """
    n_synth: number of synthetic data points to make
    window_size: size of input/output windows
    submeter_threshold: threshold at which a submeter measurement is considered "on"
    aggregate_col: name of the column in the input file to use as aggregate data
    submeter_col: name of the column in the input file to use as submeter data
    verbose: toggle verbosity (only displays progress)
    window_step: size of the steps between each input/output window generated
    output_file_[X|y]: files into which the data is written

    WARNING:
    The training set is returned unnormalized and should be normalized before training.
    """
    aggregate_data = read_dataport_file(aggregate_data_file, aggregate_col)
    submeter_data = read_dataport_file(submeter_data_file, submeter_col)
    open(output_file_X, 'w').close()
    open(output_file_y, 'w').close()

    if verbose:
        print("Data files read, beginning training set extraction")
    if type == 'discrete':
        for house in aggregate_data.columns:
            real_training_set = discrete_output_training_set(aggregate_data[house], submeter_data[house], window_size=24, threshold=submeter_threshold, step_size=window_step)
            X = pd.DataFrame(real_training_set['X'], columns=np.arange(window_size))
            y = pd.DataFrame(real_training_set['y'], columns=np.arange(window_size))

            with open(output_file_X, 'a') as f:
                X.to_csv(f, index=False)

            with open(output_file_y, 'a') as f:
                y.to_csv(f, index=False)

            if verbose:
                print("House {} done".format(house))

        if n_synth > 0:
            for house in aggregate_data.columns:
                synth_training_set = syntethic_discrete_output_training_set(aggregate_data[house], submeter_data[house], int(n_synth/len(aggregate_data.columns)), threshold=submeter_threshold, window_size=window_size, verbose=verbose)

                X = pd.DataFrame(synth_training_set['X'], columns=np.arange(window_size))
                y = pd.DataFrame(synth_training_set['y'], columns=np.arange(window_size))

                with open(output_file_X, 'a') as f:
                    X.to_csv(f, index=False, header=False)

                with open(output_file_y, 'a') as f:
                    y.to_csv(f, index=False, header=False)

                if verbose:
                    print("House {} synthetic done".format(house))

    elif type == 'continuous':
        for house in aggregate_data.columns:
            real_training_set = continuous_output_training_set(aggregate_data[house], submeter_data[house], window_size=24, threshold=submeter_threshold, step_size=window_step)

            X = pd.DataFrame(real_training_set['X'], columns=np.arange(window_size))
            y = pd.DataFrame(real_training_set['y'], columns=np.arange(window_size))

            with open(output_file_X, 'a') as f:
                X.to_csv(f, index=False)

            with open(output_file_y, 'a') as f:
                y.to_csv(f, index=False)

            if verbose:
                print("House {} done".format(house))

        if n_synth > 0:
            for house in aggregate_data.columns:
                synth_training_set = syntethic_continuous_output_training_set(aggregate_data[house], submeter_data[house], int(n_synth/len(aggregate_data.columns)), threshold=submeter_threshold, window_size=window_size, verbose=verbose)

                X = pd.DataFrame(synth_training_set['X'], columns=np.arange(window_size))
                y = pd.DataFrame(synth_training_set['y'], columns=np.arange(window_size))

                with open(output_file_X, 'a') as f:
                    X.to_csv(f, index=False)

                with open(output_file_y, 'a') as f:
                    y.to_csv(f, index=False)

                if verbose:
                    print("House {} synthetic done".format(house))

    return
