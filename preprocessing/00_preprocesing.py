import pickle
import tqdm
import pandas as pd
import numpy as np
from scipy.stats import stats
from datetime import timedelta
from utils import TEMPORAL_BASE_PATH, STATIC_PATH, PREPROCESSED_DATA_PATH

static_df = pd.read_csv(STATIC_PATH)
static_df = static_df.set_index('ID')

temporal_patient_ids = ...


def load_patient_temporal_data(patient_id, static_df, temporal_base_path, days_to_load=7, hours_to_drop=4):
    with open(fr'{temporal_base_path}\{patient_id}', 'rb') as f:
        temporal_pat = pickle.load(f)

    temporal_data = pd.DataFrame(temporal_pat['data'][:temporal_pat['timeCounter']],
                                 columns=temporal_pat['parNames'])

    temporal_data = temporal_data.sort_values(by='time')
    temporal_data.reset_index(inplace=True, drop=True)
    birthdate = static_df.loc[patient_id].birthdate
    temporal_data['time'] = temporal_data['time'] - birthdate
    l = (temporal_data['time'] - birthdate) / (1/24/60)
    temporal_data['time_from_birth'] = round_seconds([timedelta(minutes=x) for x in l])
    temporal_data = temporal_data[temporal_data['time_from_birth'] >= timedelta(hours=hours_to_drop)]
    temporal_data = temporal_data.assign(day_num=lambda x: (x['time'] - birthdate + 1).apply(lambda x: x.floor()))
    temporal_data = temporal_data[temporal_data.day_num <= days_to_load]
    full_time_dataframe = pd.DataFrame([(timedelta(minutes=x) for x in range(hours_to_drop*60, days_to_load*24*60*1))],
                                       columns=['time_from_birth'])

    temporal_data = full_time_dataframe.merge(temporal_data, how='left')

    if 'FiO2' not in temporal_data.columns:
        temporal_data['FiO2'] = np.nan

    temporal_data.loc[:, 'FiO2'] = temporal_data.loc[:, 'FiO2'].fillna(method='ffill', limit=180)

    return temporal_data

def has_30_percent_missing_per_column(sequence):
    for column in sequence.columns:
        total_values = sequence[column].size
        missing_values = sequence[column].isna().sum()
        missing_percentage = (missing_values / total_values) * 100
        if missing_percentage > 30:
            return True
    return False


def is_valid(temporal_df):
    missing = []
    sequence_length = 120
    num_sequences = len(temporal_df) // sequence_length
    prev_three_missing = [0, 0, 0]

    for i in range(num_sequences):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        sequence = temporal_df.iloc[start_idx:end_idx]

        if has_30_percent_missing_per_column(sequence):
            prev_three_missing.pop(0)
            prev_three_missing.append(1)
            missing.append(1)
            if prev_three_missing == [1, 1, 1]:
                return False
        else:
            prev_three_missing.pop(0)
            prev_three_missing.append(0)
            missing.append(0)
    return True

def append_df_to_hdf5(file_path, df, key):
    with pd.HDFStore(file_path, mode='a') as store:
        store.append(key, df, data_columns=True, format='table', index=False)

def compute_features(measurements, measurement_name):
    features = {}
    features[f'Mean_{measurement_name}'] = np.mean(measurements)
    features[f'Median_{measurement_name}'] = np.median(measurements)
    features[f'Standard Deviation_{measurement_name}'] = np.std(measurements)
    features[f'Variance_{measurement_name}'] = np.var(measurements)
    features[f'Minimum_{measurement_name}'] = np.min(measurements)
    features[f'Maximum_{measurement_name}'] = np.max(measurements)
    features[f'Range_{measurement_name}'] = features[f'Maximum_{measurement_name}'] - features[f'Minimum_{measurement_name}']
    features[f'Skewness_{measurement_name}'] = stats.skew(measurements)
    features[f'Kurtosis_{measurement_name}'] = stats.kurtosis(measurements)
    features[f'IQR_{measurement_name}'] = stats.iqr(measurements)
    features[f'Entropy_{measurement_name}'] = stats.entropy(measurements)
    features[f'Autocorrelation(lag=1)_{measurement_name}'] = np.corrcoef(measurements[:-1], measurements[1:])[0, 1]
    return features



valid_temporal = list()

pulse = []
spo2 = []
adem = []
fio2 = []

positive_ids = []
negative_ids = []
feature_dicts = []

err_counter = 0

missing_Fio2 = []

for patient_id in tqdm.tqdm(temporal_patient_ids):
    try:
        pat = load_patient_temporal_data(patient_id, static_df, TEMPORAL_BASE_PATH)
        pat.loc[pat['SpO2'] < 50, 'SpO2'] = np.nan
        pat.loc[pat['SpO2'] > 100, 'SpO2'] = np.nan
        pat.loc[pat['Pulse'] < 20, 'Pulse'] = np.nan
        pat.loc[pat['Pulse'] > 250, 'Pulse'] = np.nan
        pat.loc[pat['Ademfrequentie'] < 1, 'Ademfrequentie'] = np.nan
        pat.loc[pat['Ademfrequentie'] > 100, 'Ademfrequentie'] = np.nan
        pat.loc[pat['FiO2'] > 100, 'FiO2'] = np.nan
        pat.loc[pat['FiO2'] < 0, 'FiO2'] = np.nan

        sequence = pat[['SpO2', 'Ademfrequentie', 'Pulse', 'FiO2']]

        for column in pat[['SpO2', 'Ademfrequentie', 'Pulse', 'FiO2']].columns:
            total_values = sequence[column].size
            missing_values = sequence[column].isna().sum()
            missing_percentage = (missing_values / total_values)
            if missing_percentage > .3:
                continue

        if is_valid(pat[['SpO2', 'Ademfrequentie', 'Pulse', 'FiO2']]):
            valid_temporal.append(patient_id)

            pat_filtered = pat[['SpO2', 'Ademfrequentie', 'Pulse', 'FiO2']].interpolate(method='linear', limit_direction='both')

            features = {}
            features.update(compute_features(pat_filtered['SpO2'].to_numpy(), 'SpO2'))
            features.update(compute_features(pat_filtered['Ademfrequentie'].to_numpy(), 'Ademfrequentie'))
            features.update(compute_features(pat_filtered['Pulse'].to_numpy(), 'Pulse'))
            features.update(compute_features(pat_filtered['FiO2'].to_numpy(), 'FiO2'))
            features['ID'] = patient_id
            feature_dicts.append(features)

            spo2.extend(pat_filtered['SpO2'].to_numpy())
            pulse.extend(pat_filtered['Pulse'].to_numpy())
            adem.extend(pat_filtered['Ademfrequentie'].to_numpy())
            fio2.extend(pat_filtered['FiO2'].to_numpy())

            if static_df.loc[patient_id]['bpd_label'] == 1:
                positive_ids.append(patient_id)
            else:
                negative_ids.append(patient_id)

    except Exception as e:
        err_counter += 1
        pass

min_spo2 = np.min(np.array(spo2))
min_pulse = np.min(np.array(pulse))
min_adem = np.min(np.array(adem))
min_fio2 = np.min(np.array(fio2))
max_spo2 = np.max(np.array(spo2))
max_pulse = np.max(np.array(pulse))
max_adem = np.max(np.array(adem))
max_fio2 = np.max(np.array(fio2))


for patient_id in tqdm(valid_temporal):

    pat = load_patient_temporal_data(patient_id, static_df, TEMPORAL_BASE_PATH)
    pat.loc[pat['SpO2'] < 50, 'SpO2'] = np.nan
    pat.loc[pat['SpO2'] > 100, 'SpO2'] = np.nan
    pat.loc[pat['Pulse'] < 20, 'Pulse'] = np.nan
    pat.loc[pat['Pulse'] > 250, 'Pulse'] = np.nan
    pat.loc[pat['Ademfrequentie'] < 1, 'Ademfrequentie'] = np.nan
    pat.loc[pat['Ademfrequentie'] > 100, 'Ademfrequentie'] = np.nan
    pat.loc[pat['FiO2'] > 100, 'FiO2'] = np.nan
    pat.loc[pat['FiO2'] < 0, 'FiO2'] = np.nan

    pat_filtered = pat[['SpO2', 'Ademfrequentie', 'Pulse', 'FiO2']].interpolate(method='linear', limit_direction='both')

    pat_filtered['SpO2'] = (pat_filtered['SpO2'] - min_spo2) / (max_spo2 - min_spo2)
    pat_filtered['Ademfrequentie'] = (pat_filtered['Ademfrequentie'] - min_adem) / (max_adem - min_adem)
    pat_filtered['Pulse'] = (pat_filtered['Pulse'] - min_pulse) / (max_pulse - min_pulse)
    pat_filtered['FiO2'] = (pat_filtered['FiO2'] - min_fio2) / (max_fio2 - min_fio2)

    append_df_to_hdf5(f"{PREPROCESSED_DATA_PATH}/temporal_data.h5", pat_filtered, patient_id)


