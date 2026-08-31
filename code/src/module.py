import numpy as np

def get_binned_triggered_spike_counts_fast(spike_times, stim_times, bins):
    """
    Fast peri-stimulus time histogram using searchsorted.

    Parameters
    ----------
    spike_times : 1D array_like, sorted
        Times of all spikes (e.g. in seconds).
    stim_times : 1D array_like
        Times of stimulus onsets.
    bins : 1D array_like
        Bin edges *relative* to stimulus (e.g. np.linspace(-0.1, 0.5, 61)).

    Returns
    -------
    counts : 2D ndarray, shape (n_trials, len(bins)-1)
        counts[i, j] is the number of spikes in bin j of trial i.
    """
    # ensure numpy arrays
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)
    bins = np.asarray(bins)

    # If your spike_times isn't already sorted, uncomment:
    # spike_times = np.sort(spike_times)

    n_trials = stim_times.size
    n_bins = bins.size - 1
    counts = np.zeros((n_trials, n_bins), dtype=int)

    for i, stim in enumerate(stim_times):
        # compute the absolute edges for this trial
        edges = stim + bins
        # find the insertion indices for each edge
        spikes_before_edge = np.searchsorted(spike_times, edges, side='left')
        # differences between successive indices = counts per bin
        counts[i, :] = np.diff(spikes_before_edge)

    return counts

def neuron_psth(units_table, neuron_row_index, stim_time_array, psth_bin_edge_array):
    spike_time_array = units_table.spike_times.values[neuron_row_index]
    count_array = get_binned_triggered_spike_counts_fast(
        spike_time_array, stim_time_array, psth_bin_edge_array)
    return count_array.mean(axis=0) / 0.05

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, label_array, title):
    count_array = confusion_matrix(y_true, y_pred, labels=label_array)          # raw counts
    fraction_array = count_array / count_array.sum(axis=1, keepdims=True)       # row-normalised (each true class sums to 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    mesh = ax.pcolormesh(fraction_array, cmap='viridis', alpha=0.7,
                         edgecolor='white', linewidth=2)                        # colored cells, white edges
    for r in range(len(label_array)):                                          # write count + percent in each cell
        for c in range(len(label_array)):
            ax.text(c + 0.5, r + 0.5, '%d\n%.0f%%'
                    % (count_array[r, c], 100 * fraction_array[r, c]),
                    ha='center', va='center', color='black')
    ax.set_xticks(np.arange(len(label_array)) + 0.5)
    ax.set_yticks(np.arange(len(label_array)) + 0.5)
    ax.set_xticklabels(label_array)
    ax.set_yticklabels(label_array)
    ax.invert_yaxis()                                                          # put first class at top
    ax.set_xlabel('stimulus predicted by the decoder')
    ax.set_ylabel('stimulus actually shown')
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('fraction of true-class\n trials assigned to this class')
    fig.tight_layout()
    plt.show()