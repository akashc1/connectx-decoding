from collections import defaultdict
import math
from typing import Tuple

from brainbox.io.one import SpikeSortingLoader
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
from ibllib.atlas import AllenAtlas
import matplotlib.pyplot as plt
import numpy as np
from one.api import ONE
import pandas as pd


def remove_nans(lis):
    tes = np.array([[i, val] for i, val in enumerate(lis)])
    row_remove = np.argwhere(np.isnan(tes))
    tes = tes[~np.isnan(tes).any(axis=1)]

    return tes, row_remove[:, 0]


def get_spikedata(eid):
    PIDlist = one.eid2pid(eid)
    datalist = []
    print(PIDlist)
    for PID in PIDlist[0]:
        sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        datalist.append([spikes, clusters, channels])
    return datalist


def select_trials(trial_data, trial_condition, condition):
    indices = [
        i
        for i, trial in enumerate(trial_data)
        if trial_condition[i] == condition and not math.isnan(trial)
    ]
    result = [trial for i, trial in enumerate(trial_data) if i in indices]
    return result, indices


def get_matrices(trials, spikes, relev_neur, param='choice', param_condition=-1):

    move_trials, trial_indices = select_trials(
        trials['firstMovement_times'],
        trials[param],
        param_condition,
    )
    print(len(move_trials), move_trials)
    print(len(trial_indices), trial_indices)

    # 0D behavioral parameters
    movement_choices = trials.choice[trial_indices]
    trial_results = trials.feedbackType[trial_indices]
    print(len(movement_choices))

    peth, spike_counts = calculate_peths(
        spikes.times, spikes.clusters, relev_neur,
        move_trials,
        pre_time=3, post_time=3, bin_size=0.05, smoothing=0,
    )

    print('peth["tscale"] contains the timebin centers relative to the event')
    print(f'\npeth["means"] is shaped: {peth["means"].shape}')
    print('This variable is NxB (neurons x timebins) and contains the mean spike rates over trials')
    print(f'\nspike_counts is shaped: {spike_counts.shape}')
    print('This variable is TxNxB (trials x neurons x timebins) and contains all spike rates per trial')

    # If you just want all the spikes over the entire 0-300 ms window you can sum like this:
    whole_window = np.sum(spike_counts, axis=2)

    print(np.linalg.norm(whole_window))
    print(f'\nwhole_window is shaped: {whole_window.shape}')
    print('This variable is TxN (trials x neurons) and contains summed spike rates per trial')

    trial_data = {}
    trial_data['movement_init_times'] = move_trials
    trial_data['choices'] = movement_choices
    trial_data['feedback'] = trial_results
    trial_data['trial_indices'] = trial_indices

    return peth, spike_counts, trial_data


def gen_eidlist(roi_name):
    ses = one.alyx.rest('sessions', 'list', atlas_acronym=roi_name)
    return [i['id'] for i in ses]


def check_num_ses(list_of_eids):
    roi_intersection = set(list_of_eids[0]).intersection(*list_of_eids[1:])
    print('Found ' + str(len(list(roi_intersection))) + 'recording sessions')
    return len(list(roi_intersection))


def get_region_mapping(csv_path='proj_brainregions.csv') -> Tuple[dict, dict]:
    regions = pd.read_csv(csv_path).columns
    region2ind, ind2region = {}, {}
    for i, region in enumerate(regions):
        if region == 'fiber tracts':  # strange one we can ignore
            continue

        region = region.split('.')[0]
        region2ind[region] = i
        ind2region[i] = region

    return region2ind, ind2region


if __name__ == '__main__':
    one = ONE(
        cache_dir='data',
        base_url='https://openalyx.internationalbrainlab.org',
        password='international',
        silent=True,
    )
    ba = AllenAtlas()

    r2i, i2r = get_region_mapping()

    roi2eids = {}
    eids2rois = defaultdict(list)
    for roi in r2i:
        eids = gen_eidlist(roi)
        roi2eids[roi] = eids
        for e in eids:
            eids2rois[e].append(roi)

    most_rois_eid = max(eids2rois, key=lambda eid: len(eids2rois[eid]))
    print(f"EID with most number of ROIs: {most_rois_eid}")
    print(f"{len(eids2rois)=} total EIDs")


