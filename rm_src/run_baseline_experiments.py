from enum import Enum
import random
from pathlib import Path

# Math and graphing.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn imports.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, r2_score, ConfusionMatrixDisplay

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pprint import pprint
from sklearn.inspection import permutation_importance
from time import perf_counter
from sklearn.ensemble import RandomForestClassifier


# Global experiment flags and variables.
SEED = 19348
TARGETS = ['p_micro', 'no_trip', 's_car', 'transit', 'car', 's_micro', 'ridehail', 'walk', 'unknown']
CV = False

# Set the Numpy seed too.
random.seed(SEED)
np.random.seed(SEED)

class SPLIT_TYPE(Enum):
    INTRA_USER = 0
    INTER_USER = 1
    TARGET = 2
    MODE = 3
    

class SPLIT(Enum):
    TRAIN = 0
    TEST = 1

def get_splits(count_df: pd.DataFrame, n:int, test_size=0.2):
    maxsize = int(n * test_size)

    max_threshold = int(maxsize * 1.05)
    min_threshold = int(maxsize * 0.95)

    print(f"{min_threshold}, {max_threshold}")
    
    # Allow a 10% tolerance
    def _dp(ix, curr_size, ids, cache):
        
        if ix >= count_df.shape[0]:
            return []

        key = ix

        if key in cache:
            return cache[key]

        if curr_size > max_threshold:
            return []

        if min_threshold <= curr_size <= max_threshold:
            return ids

        # two options - either pick the current id or skip it.
        branch_a = _dp(ix, curr_size+count_df.loc[ix, 'count'], ids+[count_df.loc[ix, 'index']], cache)
        branch_b = _dp(ix+1, curr_size, ids, cache)
        
        curr_max = []
        if branch_a and len(branch_a) > 0:
            curr_max = branch_a
        
        if branch_b and len(branch_b) > len(branch_a):
            curr_max = branch_b
            
        cache[key] = curr_max
        return cache[key]
    
    return _dp(0, 0, ids=list(), cache=dict())


def get_train_test_splits(data: pd.DataFrame, how=SPLIT_TYPE, test_ratio=0.2, shuffle=True):

    n_users = list(data.user_id.unique())
    n = data.shape[0]
    
    if shuffle:
        data = data.sample(data.shape[0], random_state=SEED).reset_index(drop=True, inplace=False)

    if how == SPLIT_TYPE.INTER_USER:
        # Make the split, ensuring that a user in one fold is not leaked into the other fold.
        # Basic idea: we want to start with the users with the highest instances and place 
        # alternating users in each set.
        counts = data.user_id.value_counts().reset_index(drop=False, inplace=False, name='count')

        # Now, start with the user_id at the top, and keep adding to either split.
        # This can be achieved using a simple DP program.
        test_ids = get_splits(counts, data.shape[0])
        test_data = data.loc[data.user_id.isin(test_ids), :]
        train_index = data.index.difference(test_data.index)
        train_data = data.loc[data.user_id.isin(train_index), :]
        
        return train_data, test_data
    
    elif how == SPLIT_TYPE.INTRA_USER:
        
        # There are certain users with only one observation. What do we do with those?
        # As per the mobilitynet modeling pipeline, we randomly assign them to either the
        # training or test set.
        
        value_counts = data.user_id.value_counts()
        single_count_ids = value_counts[value_counts == 1].index
        
        data_filtered = data.loc[~data.user_id.isin(single_count_ids), :].reset_index(drop=True)
        data_single_counts = data.loc[data.user_id.isin(single_count_ids), :].reset_index(drop=True)
        
        X_tr, X_te = train_test_split(
            data_filtered, test_size=test_ratio, shuffle=shuffle, stratify=data_filtered.user_id,
            random_state=SEED
        )
        
        data_single_counts['assigned'] = np.random.choice(['train', 'test'], len(data_single_counts))
        X_tr_merged = pd.concat(
            [X_tr, data_single_counts.loc[data_single_counts.assigned == 'train', :].drop(
                columns=['assigned'], inplace=False
            )],
            ignore_index=True, axis=0
        )
        
        X_te_merged = pd.concat(
            [X_te, data_single_counts.loc[data_single_counts.assigned == 'test', :].drop(
                columns=['assigned'], inplace=False
            )],
            ignore_index=True, axis=0
        )
        
        return X_tr_merged, X_te_merged
    
    elif how == SPLIT_TYPE.TARGET:
        
        X_tr, X_te = train_test_split(
            data, test_size=test_ratio, shuffle=shuffle, stratify=data.target,
            random_state=SEED
        )
        
        return X_tr, X_te
    
    elif how == SPLIT_TYPE.MODE:
        X_tr, X_te = train_test_split(
            data, test_size=test_ratio, shuffle=shuffle, stratify=data.section_mode_argmax,
            random_state=SEED
        )
        
        return X_tr, X_te
    
    raise NotImplementedError("Unknown split type")

    
def get_duration_estimate(df: pd.DataFrame, dset: SPLIT, model_dict: dict):
    
    X_features = ['section_distance_argmax', 'age']
    
    if 'mph' in df.columns:
        X_features += ['mph']
    
    if dset == SPLIT.TRAIN and model_dict is None:
        model_dict = dict()
    
    if dset == SPLIT.TEST and model_dict is None:
        raise AttributeError("Expected model dict for testing.")
    
    if dset == SPLIT.TRAIN:
        for section_mode in df.section_mode_argmax.unique():
            section_data = df.loc[df.section_mode_argmax == section_mode, :]
            if section_mode not in model_dict:
                model_dict[section_mode] = dict()

                model = LinearRegression(fit_intercept=True)

                X = section_data[
                    X_features
                ]
                Y = section_data[['section_duration_argmax']]

                model.fit(X, Y.values.ravel())

                r2 = r2_score(y_pred=model.predict(X), y_true=Y.values.ravel())
                print(f"\t-> Train R2 for {section_mode}: {r2}")

                model_dict[section_mode]['model'] = model
                
    elif dset == SPLIT.TEST:
        for section_mode in df.section_mode_argmax.unique():
            section_data = df.loc[df.section_mode_argmax == section_mode, :]
            X = section_data[
                X_features
            ]
            Y = section_data[['section_duration_argmax']]
            
            y_pred = model_dict[section_mode]['model'].predict(X)
            r2 = r2_score(y_pred=y_pred, y_true=Y.values.ravel())
            print(f"\t-> Test R2 for {section_mode}: {r2}")
    
    # Create the new columns for the duration.
    new_columns = ['p_micro','no_trip','s_car','transit','car','s_micro','ridehail','walk','unknown']
    df[new_columns] = 0
    df['temp'] = 0
    
    for section in df.section_mode_argmax.unique():
        X_section = df.loc[df.section_mode_argmax == section, X_features]
        
        # broadcast to all columns.
        df.loc[df.section_mode_argmax == section, 'temp'] = model_dict[section]['model'].predict(X_section)
    
    for c in new_columns:
        df[c] = df['av_' + c] * df['temp']
    
    df.drop(columns=['temp'], inplace=True)
    
    df.rename(columns=dict([(x, 'tt_'+x) for x in new_columns]), inplace=True)
    
    # return model_dict, result_df
    return model_dict, df

# Some helper functions that will help ease redundancy in the code.

def drop_columns(df: pd.DataFrame):
    to_drop = [
        'source', 'end_ts', 'end_fmt_time', 'end_loc', 'raw_trip', 'start_ts', 
        'start_fmt_time', 'start_loc', 'duration', 'distance', 'start_place', 
        'end_place', 'cleaned_trip', 'inferred_labels', 'inferred_trip', 'expectation',
        'confidence_threshold', 'expected_trip', 'user_input', 'start:year', 'start:month', 
        'start:day', 'start_local_dt_minute', 'start_local_dt_second', 
        'start_local_dt_weekday', 'start_local_dt_timezone', 'end:year', 'end:month', 'end:day', 
        'end_local_dt_minute', 'end_local_dt_second', 'end_local_dt_weekday', 
        'end_local_dt_timezone', '_id', 'user_id', 'metadata_write_ts', 'additions', 
        'mode_confirm', 'purpose_confirm', 'Mode_confirm', 'Trip_purpose', 
        'original_user_id', 'program', 'opcode', 'Timestamp', 'birth_year', 
        'available_modes', 'section_coordinates_argmax', 'section_mode_argmax'
    ]
    
    # Drop section_mode_argmax and available_modes.
    return df.drop(
        columns=to_drop, 
        inplace=False
    )


def scale_values(df: pd.DataFrame, split: SPLIT, scalers=None):
    # Scale costs using StandardScaler.
    costs = df[[c for c in df.columns if 'cost_' in c]].copy()
    times = df[[c for c in df.columns if 'tt_' in c or 'duration' in c]].copy()
    distances = df[[c for c in df.columns if 'distance' in c]]
    
    print(
        "Cost columns to be scaled: ", costs.columns,"\nTime columns to be scaled: ", times.columns, \
        "\nDistance columns to be scaled: ", distances.columns
    )
    
    if split == SPLIT.TRAIN and scalers is None:
        cost_scaler = StandardScaler()
        tt_scaler = StandardScaler()
        dist_scaler = StandardScaler()
        
        cost_scaled = pd.DataFrame(
            cost_scaler.fit_transform(costs), 
            columns=costs.columns, 
            index=costs.index
        )
        
        tt_scaled = pd.DataFrame(
            tt_scaler.fit_transform(times),
            columns=times.columns,
            index=times.index
        )
        
        dist_scaled = pd.DataFrame(
            dist_scaler.fit_transform(distances),
            columns=distances.columns,
            index=distances.index
        )
    
    elif split == SPLIT.TEST and scalers is not None:
        
        cost_scaler, tt_scaler, dist_scaler = scalers
        
        cost_scaled = pd.DataFrame(
            cost_scaler.transform(costs), 
            columns=costs.columns, 
            index=costs.index
        )
        
        tt_scaled = pd.DataFrame(
            tt_scaler.transform(times), 
            columns=times.columns, 
            index=times.index
        )
        
        dist_scaled = pd.DataFrame(
            dist_scaler.transform(distances),
            columns=distances.columns,
            index=distances.index
        )
        
    else:
        raise NotImplementedError("Unknown split")
    
    # Drop the original columns.
    df.drop(
        columns=costs.columns.tolist() + times.columns.tolist() + distances.columns.tolist(), 
        inplace=True
    )
    
    df = df.merge(right=cost_scaled, left_index=True, right_index=True)
    df = df.merge(right=tt_scaled, left_index=True, right_index=True)
    df = df.merge(right=dist_scaled, left_index=True, right_index=True)
    
    return df, (cost_scaler, tt_scaler, dist_scaler)


def train(X_tr, Y_tr):
    if CV:

        model = RandomForestClassifier(random_state=SEED)

        # We want to build bootstrapped trees that would not always use all the features.
        param_set2 = {
            'n_estimators': [150, 200, 250],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'class_weight': ['balanced_subsample'],
            'max_features': [None, 'sqrt'],
            'bootstrap': [True]
        }

        cv_set2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

        clf_set2 = GridSearchCV(model, param_set2, cv=cv_set2, n_jobs=-1, scoring='f1_weighted', verbose=1)

        start = perf_counter()

        clf_set2.fit(
            X_tr,
            Y_tr
        )

        time_req = (perf_counter() - start)/60.

        best_model = clf_set2.best_estimator_
    else:
        best_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=SEED,
            n_jobs=-1
        ).fit(X_tr, Y_tr)
    
    return best_model


def predict(model, X_tr, Y_tr, X_te, Y_te):
    
    y_test_pred = model.predict(X_te)
    y_train_pred = model.predict(X_tr)
    
    train_f1 = f1_score(
        y_true=Y_tr,
        y_pred=y_train_pred,
        average='weighted'
    )
    
    test_f1 = f1_score(
        y_true=Y_te,
        y_pred=y_test_pred,
        average='weighted'
    )
    
    return y_train_pred, train_f1, y_test_pred, test_f1


def run_sampled_sweep(df: pd.DataFrame, dir_name: Path, **kwargs):
    
    targets = TARGETS.copy()
    
    drop_s_micro = kwargs.pop('drop_s_micro', None)
    
    if drop_s_micro:
        df.drop(
            index=df.loc[data.target == 6, :].index,
            inplace=True
        )
    
        # Shift all values after 6 by -1
        df.loc[data.target > 5, 'target'] -= 1
        
        # Update targets.
        targets.pop(targets.index('s_micro'))
        
    split = kwargs.pop('split', None)
    
    train_data, test_data = get_train_test_splits(data=df, how=split, shuffle=True)
    
    params, train_data = get_duration_estimate(train_data, SPLIT.TRAIN, None)
    _, test_data = get_duration_estimate(test_data, SPLIT.TEST, params)
    
    train_data = drop_columns(train_data)
    test_data = drop_columns(test_data)
    
    drop_location = kwargs.pop('drop_location', None)
    
    if drop_location:
        train_data.drop(columns=['start_lat', 'start_lng', 'end_lat', 'end_lng'], inplace=True)
        test_data.drop(columns=['start_lat', 'start_lng', 'end_lat', 'end_lng'], inplace=True)
    
    X_tr, Y_tr = train_data.drop(columns=['target'], inplace=False), train_data.target.values.ravel()
    X_te, Y_te = test_data.drop(columns=['target'], inplace=False), test_data.target.values.ravel()
    
    model = train(X_tr, Y_tr)
    tr_preds, tr_f1, te_preds, te_f1 = predict(model, X_tr, Y_tr, X_te, Y_te)
    
    print(f"\t-> Train F1: {tr_f1}, Test F1: {te_f1}")
    
    importance = sorted(
        zip(
            model.feature_names_in_, 
            model.feature_importances_
        ), 
        key=lambda x: x[-1], reverse=True
    )
    
    with open(dir_name / 'f1_scores.txt', 'w') as f:
        f.write(f"Train F1: {tr_f1}\nTest F1: {te_f1}")
    
    importance_df = pd.DataFrame(importance, columns=['feature_name', 'importance'])
    importance_df.to_csv(dir_name / 'feature_importance.csv', index=False)
    
    with open(dir_name / 'classification_report.txt', 'w') as f:
        f.write(classification_report(y_true=Y_te, y_pred=te_preds, target_names=targets))
    
    fig, ax = plt.subplots(figsize=(7, 7))
    cm = ConfusionMatrixDisplay.from_estimator(
        model,
        X=X_te,
        y=Y_te,
        ax=ax
    )
    ax.set_xticklabels(targets, rotation=45)
    ax.set_yticklabels(targets)
    fig.tight_layout()
    plt.savefig(dir_name / 'test_confusion_matrix.png')
    plt.close('all')


def save_metadata(dir_name: Path, **kwargs):
    with open(dir_name / 'metadata.txt', 'w') as f:
        for k, v in kwargs.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    data = pd.read_csv('../data/ReplacedMode_Fix_02142024.csv')
    data.drop_duplicates(inplace=True)
    
    print("Beginning sweeps.")
    
    start = perf_counter()
    sweep_number = 1
    
    for split in [SPLIT_TYPE.INTRA_USER, SPLIT_TYPE.TARGET, SPLIT_TYPE.MODE]:
        for drop in [True, False]:
            for location_drop in [True, False]:
                kwargs = {
                    'drop_s_micro': drop,
                    'split': split,
                    'drop_location': location_drop
                }
                dir_name = Path(f'../benchmark_results/benchmark_{sweep_number}')
                
                if not dir_name.exists():
                    dir_name.mkdir()
                
                print(f"\t-> Running sweep #{sweep_number}...")
                save_metadata(dir_name, **kwargs)
                run_sampled_sweep(data.copy(), dir_name, **kwargs)
                print(f"Completed benchmarking for {sweep_number} experiment.")
                print(50*'-')
                sweep_number += 1
                
    elapsed = perf_counter() - start
    
    print(f"Completed sweeps in {elapsed/60.} minutes")