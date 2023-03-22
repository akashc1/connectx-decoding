from pathlib import Path
import pickle
import math

from sklearn import svm
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from ibllib.atlas import AllenAtlas
import numpy as np
from one.api import ONE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

ROIS = ['STRv', 'STRd', 'MOp', 'MOs']
EIDS = [
    'ee8b36de-779f-4dea-901f-e0141c95722b',
    '81a78eac-9d36-4f90-a73a-7eb3ad7f770b',
    '88d24c31-52e4-49cc-9f32-6adbeb9eba87'
]
FAMILY_DIC = {'MOp': [], 'STRd': ['CP'], 'STRv': ['ACB', 'FS']}


def remove_nans(lis, indices):
    tes = np.array([[i, val] for i, val in enumerate(lis)])
    row_remove = np.argwhere(np.isnan(tes))
    tes = tes[~np.isnan(tes).any(axis=1)]

    return tes, row_remove[:, 0]


def get_spikedata_pid(PID):
    sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    return spikes, clusters, channels


def get_spikedata_eid(eid):
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


def get_matrices(
    trials,
    spikes,
    relev_neur,
    param='choice',
    param_condition=-1,
    trial_def=(3, 3, 0.05),
    trial_timing='firstMovement_times',
):

    move_trials, trial_indices = select_trials(trials[trial_timing], trials[param], param_condition)

    # 0D behavioral parameters
    movement_choices = trials.choice[trial_indices]
    trial_results = trials.feedbackType[trial_indices]

    peth, spike_counts = calculate_peths(
        spikes.times, spikes.clusters, relev_neur,
        move_trials,
        pre_time=trial_def[0],
        post_time=trial_def[1],
        bin_size=trial_def[2],
        smoothing=0,
    )

    print('peth["tscale"] contains the timebin centers relative to the event')
    print(f'\npeth["means"] is shaped: {peth["means"].shape}')
    print('This variable is NxB (neurons x timebins) and contains the mean spike rates over trials')
    print(f'\nspike_counts is shaped: {spike_counts.shape}')
    print('This variable is (trials x neurons x timebins) and contains all spike rates per trial')

    # If you just want all the spikes over the entire 0-300 ms window you can sum like this:
    whole_window = np.sum(spike_counts, axis=2)

    print(np.linalg.norm(whole_window))
    print(f'\nwhole_window is shaped: {whole_window.shape}')
    print('This variable is TxN (trials x neurons) and contains summed spike rates per trial')

    trial_data = {}
    trial_data['movement_init_times'] = move_trials
    print(type(move_trials), len(move_trials))
    trial_data['choices'] = movement_choices
    trial_data['feedback'] = trial_results
    trial_data['trial_indices'] = trial_indices

    return peth, spike_counts, trial_data


def gen_eidlist(roi_name):
    ses = one.alyx.rest('sessions', 'list', atlas_acronym=roi_name)
    eids = [i['id'] for i in ses]
    return eids


def gen_pidlist(roi_name):
    ses = one.alyx.rest('insertions', 'list', atlas_acronym=roi_name)
    pids = [i['id'] for i in ses]
    return pids


def check_num_ses(list_of_eids):
    roi_intersection = set(list_of_eids[0]).intersection(*list_of_eids[1:])
    print('Found ' + str(len(list(roi_intersection))) + 'recording sessions')
    return len(list(roi_intersection))


def is_child_of(child, parent, family_dictionary=FAMILY_DIC):
    if parent in child:
        return True

    if parent not in family_dictionary.keys():
        return False
    elif child in family_dictionary[parent]:
        return True
    else:
        return False


def get_connectome_weights(file='proj_strengths.xlsx'):
    projectome = pd.read_excel(file, sheet_name='W_ipsi', index_col=0)
    strengths = np.zeros([7, 7])
    for i, source in enumerate(ROIS):
        for j, target in enumerate(ROIS):
            # print(round(projectome.loc[source, target], 2))
            strengths[i, j] = projectome.loc[source, target]
        if np.sum(strengths[i, :]) > 0:
            strengths[i, :] /= np.sum(strengths[i, :])

    return strengths


def get_data_per_recording(
    eid,
    corr_regions,
    trial_def=(3, 3, 0.05),
    trial_timing='firstMovement_times',
):
    datalist = get_spikedata_eid(eid)
    trials = one.load_object(eid, 'trials')

    region_2_data = {}

    for region in corr_regions:
        relev_neur_list = []

        probe = 0
        relev_neur_0 = [
            i
            for i, acronym in enumerate(datalist[probe][1]['acronym'])
            if is_child_of(acronym, region)
        ]

        if len(relev_neur_0) > 0:  # will be performed if datalist has one probe OR has two probes
            relev_neur_list.append(relev_neur_0)

        if len(datalist) > 1:  # if two probes
            probe = 1
            relev_neur_1 = [
                i
                for i, acronym in enumerate(datalist[probe][1]['acronym'])
                if is_child_of(acronym, region)
            ]

            if len(relev_neur_1) > 0:
                relev_neur_list.append(relev_neur_1)

        dic = {}
        for condition in [-1, 1]:
            mini_dic = {}
            probe = 0
            peth, spike_counts, trial_data = get_matrices(
                trials,
                datalist[probe][0],
                relev_neur_list[probe],
                param='choice',
                param_condition=condition,
                trial_def=trial_def,
                trial_timing=trial_timing,
            )

            if len(relev_neur_list) > 1:
                probe = 1
                peth1, spike_counts1, trial_data1 = get_matrices(
                    trials,
                    datalist[probe][0],
                    relev_neur_list[probe],
                    param='choice',
                    param_condition=condition,
                    trial_def=trial_def,
                    trial_timing=trial_timing,
                )

            mini_dic['peth'] = peth  # not updating this cuz we never use it
            # (trials x neurons x timebins)
            if len(relev_neur_list) > 1:
                conglom_spikes = np.hstack([spike_counts, spike_counts1])
                print('spike_counts CONGLOM.shape: ', conglom_spikes.shape)
                mini_dic['spike_counts'] = conglom_spikes
                mini_dic['trial_data'] = trial_data
            else:
                mini_dic['spike_counts'] = spike_counts
                mini_dic['trial_data'] = trial_data

            dic[condition] = mini_dic
        region_2_data[region] = dic

    return region_2_data


def get_all_recording_data():
    whole_dataset = {}
    for eid in EIDS:
        region_2_dat = get_data_per_recording(eid, ROIS)
        whole_dataset[eid] = region_2_dat

    return whole_dataset


def get_data_array():
    if (raw_data_pkl := Path.cwd() / 'data' / 'raw_data.pkl').exists():
        print(f"Loading cached data from {str(raw_data_pkl)}")
        with open(raw_data_pkl, 'rb') as f:
            eid_to_data = pickle.load(f)
    else:
        eid_to_data = get_all_recording_data()
        print(f"Writing data to cache at {str(raw_data_pkl)}")
        with open(raw_data_pkl, 'wb') as f:
            pickle.dump(eid_to_data, f)

    num_blank_trials = 0
    all_inputs, all_outputs = [], []
    for eid, recording in eid_to_data.items():
        missing_rois = {roi for roi in ROIS if roi not in recording}
        if missing_rois:
            print(f"Skipping {eid} since it is missing {missing_rois=}")
            continue

        num_trials = (
            recording[ROIS[0]][-1]['trial_data']['choices'].shape[0]
            + recording[ROIS[0]][1]['trial_data']['choices'].shape[0]
        )
        T = recording[ROIS[0]][1]['spike_counts'].shape[-1]
        print(f"Recording {eid}: {num_trials} trials, {T=}")

        region_inputs, region_outputs = [], []
        for ex_roi in ROIS:
            whole_data = []
            whole_outputs = []

            for condition in [-1, 1]:
                spike_counts = recording[ex_roi][condition]['spike_counts']
                if spike_counts.shape[1] < 1:
                    spike_counts = -1 * np.ones(
                        (recording[ex_roi][condition]['trial_data']['choices'].shape[0], 5, T)
                    )
                    num_blank_trials += (
                        recording[ex_roi][condition]['trial_data']['choices'].shape[0]
                    )

                dat = np.mean(spike_counts, axis=1)  # averaging over neurons

                whole_data.append(dat)
                whole_outputs.append(recording[ex_roi][condition]['trial_data']['choices'])

            whole_data = np.concatenate(whole_data)
            whole_outputs = np.concatenate(whole_outputs)
            print(whole_data.shape, whole_outputs.shape)

            region_inputs.append(whole_data)
            region_outputs.append(whole_outputs)

        all_inputs.append(np.stack(region_inputs, 1))
        all_outputs.append(np.stack(region_outputs, 1))
        print(f"{eid=} {all_inputs[-1].shape=}")

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)

    print(all_inputs.shape, all_outputs.shape)
    print("Num blank trials: ", num_blank_trials)

    return all_inputs, all_outputs[:, 0]


if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)  # used for caching IBL data
    one = ONE(
        cache_dir='data',
        base_url='https://openalyx.internationalbrainlab.org',
        password='international',
        silent=True,
    )
    ba = AllenAtlas()
    all_inputs, all_outputs = get_data_array()
    print(f"{all_inputs.shape=}, {all_outputs.shape=}")
    X_train, X_test, y_train, y_test = train_test_split(all_inputs, all_outputs, test_size=0.2)
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train[:, 0])
    y_pred = clf.predict(X_test.reshape(X_test.shape[0], -1))
    print(" accuracy:", metrics.accuracy_score(y_test[:, 0], y_pred))
