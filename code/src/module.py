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
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict,
    permutation_test_score)

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

def get_stim_times(trials, stim_name, block, outcome=None, match_block=True):
    mask = trials.stim_name == stim_name
    if match_block:
        mask &= trials.rewarded_modality == block
    if outcome is not None:
        mask &= trials[outcome]
    return mask


# stimulus that is rewarded / unrewarded in each block context
_BLOCK_STIM = {
    'aud': ('sound1', 'vis1'),
    'vis': ('vis1', 'sound1'),
}


def decode_trial_identity(
    nwbfile,
    region,
    block='aud',
    outcome_filter=1,
    window_bin_slice=slice(3, 9),
    decode_bin_edge_array=None,
    n_neurons=None,
    n_trials=None,
    min_units=5,
    min_trials_per_class=5,
    cv=5,
    max_iter=5000,
    random_state=0,
    n_permutations=200,
    n_importance_repeats=10,
    importance_top_frac=0.10,
    importance_test_size=0.25,
    stim_pair=None,
    class_outcomes=None,
    match_block=True,
    mouse_id=None,
):
    """Decode which stimulus was shown on each trial from population spike counts.

    Reproduces the single-session / single-region decoder built in
    ``Decode_ErrorTrials.ipynb`` so it can be called in a loop over mice and
    brain regions.

    A linear SVM is trained on a (trials x neurons) matrix of spike counts taken
    from ``window_bin_slice`` of ``decode_bin_edge_array`` (relative to stimulus
    onset). The two classes are the block-rewarded stimulus (e.g. ``sound1`` in
    an auditory block) and the block-unrewarded stimulus (e.g. ``vis1``).

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        An already-opened session, e.g. ``pynwb.read_nwb(nwb_path)``. Open it
        once per mouse and call this function once per region.
    region : str
        Value of ``units.structure`` to decode from (e.g. ``'MOs'``).
    block : {'aud', 'vis'}
        Task-context block to restrict trials to.
    outcome_filter : {0, 1}
        0 uses all target trials; 1 restricts to hits (rewarded stimulus) and
        false alarms (unrewarded stimulus), matching the notebook.
    window_bin_slice : slice
        Which bins of ``decode_bin_edge_array`` to sum into the decoder input.
        Default ``slice(3, 9)`` is 0 to 0.5 s after onset for the default edges.
    decode_bin_edge_array : 1D array, optional
        Bin edges relative to stimulus onset. Defaults to
        ``np.arange(-0.3, 2.0, 0.1)``.
    n_neurons : int, optional
        If set, randomly subsample this many QC-passing units in the region
        (seeded by ``random_state``) so feature count is matched across regions.
        Regions with fewer units are skipped.
    n_trials : int, optional
        If set, randomly subsample this many trials per class so sample count is
        matched. Classes with fewer trials cause the region to be skipped.
    min_units, min_trials_per_class : int
        Lower bounds used when ``n_neurons`` / ``n_trials`` are not given. If the
        available count is below the bound the region is skipped.
    cv : int
        Requested number of cross-validation folds; reduced automatically if a
        class has fewer samples than folds.
    max_iter : int
        ``LinearSVC`` iteration cap.
    random_state : int
        Seeds unit/trial subsampling, the train/test split, the SVM, and the
        label shuffles used to build the chance distribution.
    n_permutations : int
        Number of times the class labels are shuffled to estimate the chance
        level empirically. For each shuffle the full cross-validation is re-run
        on the permuted labels; ``chance`` is the mean of that null
        distribution. Set to 0 to skip the permutation test and fall back to
        the majority-class fraction.
    n_importance_repeats : int
        Number of independent stratified train/test splits used for the
        feature-importance analysis. For each repeat a fresh SVM is fit on the
        train trials, scored on the test trials, and its top
        ``importance_top_frac`` of neurons (by absolute weight) are flagged as
        important. Set to 0 to skip this analysis.
    importance_top_frac : float
        Fraction of neurons flagged as important in each repeat, fixed in
        advance (e.g. 0.10 flags the top 10% by ``|w_r|``).
    importance_test_size : float
        Held-out fraction in each importance repeat's train/test split.
    stim_pair : (str, str), optional
        The two ``stim_name`` values to decode. Defaults to the block's
        target / distractor target stimuli (``_BLOCK_STIM[block]``). Pass e.g.
        ``('sound2', 'vis2')`` to build a non-target stimulus decoder.
    class_outcomes : (str, str) or (None, None), optional
        Per-class boolean trial columns to filter on. Defaults follow
        ``outcome_filter`` (``('is_hit', 'is_false_alarm')`` when it is 1).
        Pass ``('is_correct_reject', 'is_correct_reject')`` to restrict both
        classes to trials where licking was correctly withheld.
    match_block : bool
        If True (default) only trials with ``rewarded_modality == block`` are
        used. Set False to pool a stimulus across every task block (e.g. a
        non-target decoder run in both the visual and auditory blocks).
    mouse_id : hashable, optional
        Copied into the result dict for bookkeeping in the loop.

    Returns
    -------
    dict
        Always contains ``mouse_id``, ``region``, ``block``, ``n_units``,
        ``n_rew_trials``, ``n_unrew_trials``, ``n_trials`` and ``status``.
        When ``status == 'ok'`` it also contains ``split_accuracy``,
        ``cv_accuracy_mean``, ``cv_accuracy_folds`` (array), ``chance``
        (mean of the label-shuffle null distribution), ``chance_std`` (its
        standard deviation), ``chance_p`` (permutation p-value: fraction of
        shuffles whose accuracy was >= the observed accuracy),
        ``null_accuracy`` (the ``n_permutations``-long array of shuffled-label
        accuracies), ``chance_majority_class`` (the old majority-class
        fraction, kept for reference),
        ``confusion_matrix`` (2x2 array of counts, cross-validated),
        ``confusion_labels`` (``[rew_stim, unrew_stim]``), ``coef_weights``
        (abs SVM weight per unit, from a fit on all rows), ``unit_ids``
        (index into ``units`` for each column) and ``n_cv_folds``.

        The repeated-split feature-importance analysis adds (all per-unit
        arrays aligned with ``unit_ids``): ``rep_accuracy`` (test accuracy of
        each repeat), ``rep_accuracy_mean`` / ``rep_accuracy_std``,
        ``weight_abs_mean`` / ``weight_abs_std`` (mean and SD of ``|w_r|``
        across repeats), ``weight_stability`` (fraction of repeats in which the
        unit was flagged as important -- the feature-importance ranking),
        ``n_flagged_per_rep`` (how many units are flagged each repeat),
        ``single_unit_stat`` (circularity-safe firing statistic: for each unit
        the mean over repeats in which it was flagged *and* had held-out
        trials of both classes, of ``mean rate on hit trials - mean rate on
        FA trials`` computed on that repeat's test trials only), and
        ``single_unit_stat_n_reps`` (number of repeats contributing to each
        unit's ``single_unit_stat``).

        Otherwise the metric keys are ``None`` and ``status`` explains why
        (``'insufficient_units'`` or ``'insufficient_trials'``).
    """
    if block not in _BLOCK_STIM:
        raise ValueError("block must be 'aud' or 'vis', got %r" % (block,))
    if decode_bin_edge_array is None:
        decode_bin_edge_array = np.arange(-0.3, 2.0, 0.1)
    # the two stimulus classes to decode; default = the block's target /
    # non-target target stimuli (sound1 vs vis1 style). ``stim_pair`` overrides
    # this, e.g. ('sound2', 'vis2') for a non-target stimulus decoder.
    rew_stim, unrew_stim = stim_pair if stim_pair is not None else _BLOCK_STIM[block]
    # per-class trial-outcome filter. Default follows ``outcome_filter``
    # (hits vs false alarms). Pass e.g. ('is_correct_reject', 'is_correct_reject')
    # to decode from trials where licking was correctly withheld.
    if class_outcomes is not None:
        outcome_a, outcome_b = class_outcomes
    elif outcome_filter == 1:
        outcome_a, outcome_b = 'is_hit', 'is_false_alarm'
    else:
        outcome_a, outcome_b = None, None
    rng = np.random.default_rng(random_state)

    result = {
        'mouse_id': mouse_id,
        'region': region,
        'block': block,
        'n_units': 0,
        'n_rew_trials': 0,
        'n_unrew_trials': 0,
        'n_trials': 0,
        'status': 'ok',
        'split_accuracy': None,
        'cv_accuracy_mean': None,
        'cv_accuracy_folds': None,
        'n_cv_folds': None,
        'chance': None,
        'chance_std': None,
        'chance_p': None,
        'chance_majority_class': None,
        'null_accuracy': None,
        'confusion_matrix': None,
        'confusion_labels': [rew_stim, unrew_stim],
        'classes': [rew_stim, unrew_stim],
        'match_block': match_block,
        'coef_weights': None,
        'unit_ids': None,
        'rep_accuracy': None,
        'rep_accuracy_mean': None,
        'rep_accuracy_std': None,
        'weight_abs_mean': None,
        'weight_abs_std': None,
        'weight_stability': None,
        'n_flagged_per_rep': None,
        'single_unit_stat': None,
        'single_unit_stat_n_reps': None,
    }

    # --- units: default QC, in this region, optionally subsampled ------------
    units_table = nwbfile.units.to_dataframe()
    region_units = units_table[units_table.default_qc
                               & (units_table.structure == region)]
    required_units = n_neurons if n_neurons is not None else min_units
    if len(region_units) < max(required_units, 1):
        result['n_units'] = len(region_units)
        result['status'] = 'insufficient_units'
        return result
    if n_neurons is not None:
        pick = rng.choice(len(region_units), size=n_neurons, replace=False)
        region_units = region_units.iloc[np.sort(pick)]
    result['n_units'] = len(region_units)
    result['unit_ids'] = region_units.index.to_numpy()

    # --- trials: two stimulus classes, optional block match + outcome filter -
    trials = nwbfile.trials.to_dataframe()
    rew_mask = get_stim_times(
        trials, rew_stim, block, outcome_a, match_block=match_block).to_numpy()
    unrew_mask = get_stim_times(
        trials, unrew_stim, block, outcome_b, match_block=match_block).to_numpy()

    required_trials = (n_trials if n_trials is not None
                       else min_trials_per_class)
    if (rew_mask.sum() < max(required_trials, 1)
            or unrew_mask.sum() < max(required_trials, 1)):
        result['n_rew_trials'] = int(rew_mask.sum())
        result['n_unrew_trials'] = int(unrew_mask.sum())
        result['status'] = 'insufficient_trials'
        return result
    if n_trials is not None:
        rew_mask = _subsample_mask(rew_mask, n_trials, rng)
        unrew_mask = _subsample_mask(unrew_mask, n_trials, rng)
    result['n_rew_trials'] = int(rew_mask.sum())
    result['n_unrew_trials'] = int(unrew_mask.sum())

    decode_trial_mask = rew_mask | unrew_mask
    decode_stim_time_array = trials.loc[
        decode_trial_mask, 'stim_start_time'].to_numpy()
    decode_label_array = trials.loc[decode_trial_mask, 'stim_name'].to_numpy()
    result['n_trials'] = len(decode_label_array)

    # --- population spike-count matrix: (trials, neurons) -------------------
    n_bins = len(decode_bin_edge_array) - 1
    population_tensor = np.zeros(
        (len(region_units), len(decode_stim_time_array), n_bins))
    for row, spike_times in enumerate(region_units.spike_times.values):
        population_tensor[row] = get_binned_triggered_spike_counts_fast(
            spike_times, decode_stim_time_array, decode_bin_edge_array)
    population_count_array = np.sum(
        population_tensor[:, :, window_bin_slice], axis=2).T
    assert population_count_array.shape[0] == len(decode_label_array)

    # spike counts -> firing rate (Hz) over the decode window, for the
    # single-unit firing statistic
    _start = window_bin_slice.start or 0
    _stop = (window_bin_slice.stop if window_bin_slice.stop is not None
             else len(decode_bin_edge_array) - 1)
    window_duration = float(decode_bin_edge_array[_stop]
                            - decode_bin_edge_array[_start])
    population_rate_array = population_count_array / window_duration

    # --- decode -----------------------------------------------------------
    stim_name_array = np.array([rew_stim, unrew_stim])
    _, class_counts = np.unique(decode_label_array, return_counts=True)
    result['chance_majority_class'] = class_counts.max() / class_counts.sum()
    # provisional; replaced by the label-shuffle estimate below when CV runs
    result['chance'] = result['chance_majority_class']

    x_train, x_test, y_train, y_test = train_test_split(
        population_count_array, decode_label_array,
        random_state=random_state, stratify=decode_label_array)
    fitted_svc = LinearSVC(max_iter=max_iter, random_state=random_state)
    fitted_svc.fit(x_train, y_train)
    result['split_accuracy'] = fitted_svc.score(x_test, y_test)

    n_folds = int(min(cv, class_counts.min()))
    if n_folds >= 2:
        cv_scores = cross_val_score(
            LinearSVC(max_iter=max_iter, random_state=random_state),
            population_count_array, decode_label_array, cv=n_folds)
        cv_pred = cross_val_predict(
            LinearSVC(max_iter=max_iter, random_state=random_state),
            population_count_array, decode_label_array, cv=n_folds)
        result['cv_accuracy_folds'] = cv_scores
        result['cv_accuracy_mean'] = cv_scores.mean()
        result['n_cv_folds'] = n_folds
        result['confusion_matrix'] = confusion_matrix(
            decode_label_array, cv_pred, labels=stim_name_array)

        # empirical chance: shuffle the hit / false-alarm labels n_permutations
        # times and re-run the same cross-validation on each shuffle.
        if n_permutations and n_permutations > 0:
            _, null_scores, perm_p = permutation_test_score(
                LinearSVC(max_iter=max_iter, random_state=random_state),
                population_count_array, decode_label_array,
                cv=n_folds, n_permutations=n_permutations,
                random_state=random_state, n_jobs=-1)
            result['null_accuracy'] = null_scores
            result['chance'] = float(null_scores.mean())
            result['chance_std'] = float(null_scores.std(ddof=1))
            result['chance_p'] = float(perm_p)

    # reference: weights from a single fit on every trial (as in the notebook)
    full_svc = LinearSVC(max_iter=max_iter, random_state=random_state)
    full_svc.fit(population_count_array, decode_label_array)
    result['coef_weights'] = np.abs(full_svc.coef_.ravel())

    # repeated train/test splits: per-repeat weights, flagged "important"
    # neurons, and a circularity-safe single-unit firing statistic computed
    # only on each repeat's held-out trials
    if n_importance_repeats and n_importance_repeats > 0:
        result.update(_repeated_split_importance(
            population_count_array, population_rate_array, decode_label_array,
            rew_stim=rew_stim, unrew_stim=unrew_stim,
            n_repeats=n_importance_repeats, top_frac=importance_top_frac,
            test_size=importance_test_size, max_iter=max_iter,
            random_state=random_state))

    return result


def _repeated_split_importance(count_array, rate_array, label_array, *,
                               rew_stim, unrew_stim, n_repeats, top_frac,
                               test_size, max_iter, random_state):
    """Repeated stratified train/test splits for feature importance.

    For each repeat ``r``:

    1. split trials into train/test (stratified, seed ``random_state + r``);
    2. fit a linear SVM on train -> weight vector ``w_r``;
    3. score accuracy on test -> ``accuracy_r``;
    4. flag the ``top_frac`` neurons with the largest ``|w_r|`` as important;
    5. on that repeat's test trials only, compute the single-unit statistic
       (mean firing rate on hit trials - mean rate on false-alarm trials) for
       the flagged neurons.

    Returns per-neuron arrays aligned with the columns of ``count_array``:
    ``weight_stability`` (fraction of repeats flagged), ``weight_abs_mean`` /
    ``weight_abs_std``, ``single_unit_stat`` (mean of step 5 across the repeats
    where the neuron was both flagged and had test trials of both classes) and
    ``single_unit_stat_n_reps``; plus ``rep_accuracy`` and summaries.
    """
    n_trials, n_units = count_array.shape
    n_flag = max(1, int(np.ceil(top_frac * n_units)))

    rep_acc = np.full(n_repeats, np.nan)
    abs_w = np.zeros((n_repeats, n_units))
    flagged = np.zeros((n_repeats, n_units), dtype=bool)
    su_stat = np.full((n_repeats, n_units), np.nan)  # per (repeat, neuron)

    for r in range(n_repeats):
        x_tr, x_te, y_tr, y_te, rate_tr, rate_te = train_test_split(
            count_array, label_array, rate_array,
            test_size=test_size, stratify=label_array,
            random_state=random_state + r)

        svc = LinearSVC(max_iter=max_iter, random_state=random_state)
        svc.fit(x_tr, y_tr)
        rep_acc[r] = svc.score(x_te, y_te)

        w = np.abs(svc.coef_.ravel())
        abs_w[r] = w
        cutoff = np.partition(w, n_units - n_flag)[n_units - n_flag]
        flag_r = w >= cutoff
        flagged[r] = flag_r

        hit_te = y_te == rew_stim
        fa_te = y_te == unrew_stim
        if hit_te.any() and fa_te.any():
            diff = (rate_te[hit_te].mean(axis=0)
                    - rate_te[fa_te].mean(axis=0))
            su_stat[r, flag_r] = diff[flag_r]

    su_mean = np.full(n_units, np.nan)
    has_data = np.any(~np.isnan(su_stat), axis=0)
    su_mean[has_data] = np.nanmean(su_stat[:, has_data], axis=0)

    ddof = 1 if n_repeats > 1 else 0
    return {
        'rep_accuracy': rep_acc,
        'rep_accuracy_mean': float(np.nanmean(rep_acc)),
        'rep_accuracy_std': float(np.nanstd(rep_acc, ddof=ddof)),
        'weight_abs_mean': abs_w.mean(axis=0),
        'weight_abs_std': abs_w.std(axis=0, ddof=ddof),
        'weight_stability': flagged.mean(axis=0),
        'n_flagged_per_rep': int(n_flag),
        'single_unit_stat': su_mean,
        'single_unit_stat_n_reps': np.sum(~np.isnan(su_stat), axis=0),
    }


def _subsample_mask(mask, n, rng):
    """Return a copy of boolean ``mask`` with only ``n`` of its True entries set."""
    true_ind = np.flatnonzero(mask)
    keep = rng.choice(true_ind, size=n, replace=False)
    out = np.zeros_like(mask)
    out[keep] = True
    return out