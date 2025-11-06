from enum import Enum
import random
import warnings
import argparse
from pathlib import Path
from collections import Counter

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
from scipy.special import kl_div
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from pprint import pprint
from sklearn.inspection import permutation_importance
from time import perf_counter
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=Warning)

# Global experiment flags and variables.
SEED = 13210
TARGETS = ['p_micro', 'no_trip', 's_car', 'transit', 'car', 's_micro', 'ridehail', 'walk', 'unknown']
MAP = {ix+1:t for ix, t in enumerate(TARGETS)}

CV = False

# Set the Numpy seed too.
random.seed(SEED)
np.random.seed(SEED)

class SPLIT_TYPE(Enum):
    INTRA_USER = 0
    INTER_USER = 1
    TARGET = 2
    MODE = 3
    HIDE_USER = 4
    

class SPLIT(Enum):
    TRAIN = 0
    TEST = 1


def get_train_test_splits(data: pd.DataFrame, how=SPLIT_TYPE, test_ratio=0.2, shuffle=True):
    
    if how == SPLIT_TYPE.INTER_USER:

        X = data.drop(columns=['target'])
        y = data['target'].values
        groups = data.user_id.values

        # n_splits determines split size. So n=5, is 20% for each split, which is what we want.
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=shuffle, random_state=SEED)
        # splitter = GroupKFold(n_splits=5)
        
        for train_index, test_index in splitter.split(X, y, groups):
            X_tr = data.iloc[train_index, :]
            X_te = data.iloc[test_index, :]
            
            # Iterate only once and break.
            break

        return X_tr, X_te, None
    
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
        
        return X_tr_merged, X_te_merged, None
    
    elif how == SPLIT_TYPE.TARGET:
        
        X_tr, X_te = train_test_split(
            data, test_size=test_ratio, shuffle=shuffle, stratify=data.target,
            random_state=SEED
        )
        
        return X_tr, X_te, None
    
    elif how == SPLIT_TYPE.MODE:
        X_tr, X_te = train_test_split(
            data, test_size=test_ratio, shuffle=shuffle, stratify=data.section_mode_argmax,
            random_state=SEED
        )
        
        return X_tr, X_te, None
    
    
    elif how == SPLIT_TYPE.HIDE_USER:
        users = data.user_id.value_counts(normalize=True)
        percentiles = users.quantile([0.25, 0.5, 0.75])
        
        low_trip_users = users[users <= percentiles[0.25]].index
        mid_trip_users = users[(percentiles[0.25] <= users) & (users <= percentiles[0.5])].index
        high_trip_users = users[(percentiles[0.5] <= users) & (users <= percentiles[0.75])].index
        
        # select one from each randomly.
        user1 = np.random.choice(low_trip_users)
        user2 = np.random.choice(mid_trip_users)
        user3 = np.random.choice(high_trip_users)
        
        print(f"Users picked: {user1}, {user2}, {user3}")
        
        # Remove these users from the entire dataset.
        held_out = data.loc[data.user_id.isin([user1, user2, user3]), :].reset_index(drop=True)
        remaining = data.loc[~data.user_id.isin([user1, user2, user3]), :].reset_index(drop=True)
        
        # Split randomly.
        X_tr, X_te = train_test_split(
            remaining, test_size=test_ratio, shuffle=shuffle, random_state=SEED
        )
        
        return X_tr, X_te, held_out
    
    raise NotImplementedError("Unknown split type")

    
def get_duration_estimate(df: pd.DataFrame, dset: SPLIT, model_dict: dict):
    
    X_features = ['section_distance_argmax', 'mph']
    
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

                X = section_data[X_features]
                Y = section_data[['section_duration_argmax']]

                model.fit(X, Y.values.ravel())

                r2 = r2_score(y_pred=model.predict(X), y_true=Y.values.ravel())
                print(f"\t-> Train R2 for {section_mode}: {r2}")

                model_dict[section_mode]['model'] = model
                
    elif dset == SPLIT.TEST:
        for section_mode in df.section_mode_argmax.unique():
            section_data = df.loc[df.section_mode_argmax == section_mode, :]
            X = section_data[X_features]
            Y = section_data[['section_duration_argmax']]
            
            if section_mode not in model_dict.keys():
                print(f"Inference for section={section_mode} could not be done due to lack of samples. Skipping...")
                continue
                
            y_pred = model_dict[section_mode]['model'].predict(X)
            r2 = r2_score(y_pred=y_pred, y_true=Y.values.ravel())
            print(f"\t-> Test R2 for {section_mode}: {r2}")
    
    # Create the new columns for the duration.
    new_columns = ['p_micro','no_trip','s_car','transit','car','s_micro','ridehail','walk','unknown']
    df[TARGETS] = 0
    df['temp'] = 0
    
    for section in df.section_mode_argmax.unique():
        
        # Cannot predict because the mode is present in test but not in train.
        if section not in model_dict.keys():
            df.loc[df.section_mode_argmax == section, 'temp'] = 0.
            continue
        
        X_section = df.loc[df.section_mode_argmax == section, X_features]
        
        # broadcast to all columns.
        df.loc[df.section_mode_argmax == section, 'temp'] = model_dict[section]['model'].predict(X_section)
    
    for c in TARGETS:
        df[c] = df['av_' + c] * df['temp']
    
    df.drop(columns=['temp'], inplace=True)
    
    df.rename(columns=dict([(x, 'tt_'+x) for x in TARGETS]), inplace=True)
    
    # return model_dict, result_df
    return model_dict, df

# Some helper functions that will help ease redundancy in the code.

def drop_columns(df: pd.DataFrame):
    to_drop = ['section_mode_argmax', 'available_modes', 'user_id']
    
    # Drop section_mode_argmax and available_modes.
    return df.drop(
        columns=to_drop, 
        inplace=False
    )


def scale_values(df: pd.DataFrame, split: SPLIT, scalers=None):
    # Scale costs using StandardScaler.
    costs = df[[c for c in df.columns if 'cost_' in c]].copy()
    times = df[[c for c in df.columns if 'tt_' in c or 'duration' in c]].copy()
    distances = df[[c for c in df.columns if 'distance' in c or 'mph' in c]].copy()
    
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
        average='weighted',
        zero_division=0.
    )
    
    test_f1 = f1_score(
        y_true=Y_te,
        y_pred=y_test_pred,
        average='weighted',
        zero_division=0.
    )
    
    return y_train_pred, train_f1, y_test_pred, test_f1


def run_sampled_sweep(df: pd.DataFrame, dir_name: Path, **kwargs):
    
    targets = TARGETS.copy()
        
    split = kwargs.pop('split', None)
    
    try:
        train_data, test_data, hidden_data = get_train_test_splits(data=df, how=split, shuffle=True)
    except Exception as e:
        print(e)
        return
    
    params, train_data = get_duration_estimate(train_data, SPLIT.TRAIN, None)
    _, test_data = get_duration_estimate(test_data, SPLIT.TEST, params)
    
    train_data = drop_columns(train_data)
    test_data = drop_columns(test_data)
    
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
    
    # target_names = [MAP[x] for x in np.unique(Y_te)]
    
    with open(dir_name / 'classification_report.txt', 'w') as f:
        f.write(classification_report(y_true=Y_te, y_pred=te_preds))
    
    if split == SPLIT_TYPE.HIDE_USER and hidden_data is not None:
        _, hidden_data = get_duration_estimate(hidden_data, SPLIT.TEST, params)
        hidden_data = drop_columns(hidden_data)

        X_hid, Y_hid = hidden_data.drop(columns=['target'], inplace=False), hidden_data.target.values.ravel()

        tr_preds, tr_f1, te_preds, te_f1 = predict(model, X_tr, Y_tr, X_hid, Y_hid)
        print(f"\t\t ---> Hidden user F1: {te_f1} <---")
    
    fig, ax = plt.subplots(figsize=(7, 7))
    cm = ConfusionMatrixDisplay.from_estimator(
        model,
        X=X_te,
        y=Y_te,
        ax=ax
    )
    # ax.set_xticklabels(target_names, rotation=45) 
    # ax.set_yticklabels(target_names)
    fig.tight_layout()
    plt.savefig(dir_name / 'test_confusion_matrix.png')
    plt.close('all')


def save_metadata(dir_name: Path, **kwargs):
    with open(dir_name / 'metadata.txt', 'w') as f:
        for k, v in kwargs.items():
            f.write(f"{k}: {v}\n")


            
if __name__ == "__main__":
  
    datasets = sorted(list(Path('../data/filtered_data').glob('preprocessed_data_*.csv')))
    
    start = perf_counter()
    
    for dataset in datasets:
        name = dataset.name.replace('.csv', '')
        
        print(f"Starting modeling for dataset = {name}")
        
        data = pd.read_csv(dataset)
        
        if 'deprecatedID' in data.columns:
            data.drop(columns=['deprecatedID'], inplace=True)
        if 'data.key' in data.columns:
            data.drop(columns=['data.key'], inplace=True)
    
        print(f"# Samples found: {len(data)}, # unique users: {len(data.user_id.unique())}")

        print("Beginning sweeps.")

        # args = parse_args()
        sweep_number = 1

        root = Path('../outputs/benchmark_results')
        if not root.exists():
            root.mkdir()
            

        if 'section_mode_argmax' in data.columns and (data.section_mode_argmax.value_counts() < 2).any():
            # Find which mode.
            counts = data.section_mode_argmax.value_counts()
            modes = counts[counts < 2].index.tolist()
            print(f"Dropping {modes} because of sparsity (<2 samples)")
            
            data = data.loc[~data.section_mode_argmax.isin(modes), :].reset_index(drop=True)
            

        for split in [SPLIT_TYPE.INTER_USER, SPLIT_TYPE.INTRA_USER, SPLIT_TYPE.TARGET, SPLIT_TYPE.MODE, SPLIT_TYPE.HIDE_USER]:
            
            kwargs = {
                'dataset': name,
                'split': split
            }

            dir_name = root / f'benchmark_{name}_{sweep_number}'

            if not dir_name.exists():
                dir_name.mkdir()

            print(f"\t-> Running sweep #{sweep_number} with metadata={str(kwargs)}")
            save_metadata(dir_name, **kwargs)
            run_sampled_sweep(data.copy(), dir_name, **kwargs)
            print(f"Completed benchmarking for {sweep_number} experiment.")
            print(50*'-')
            sweep_number += 1
                
    elapsed = perf_counter() - start
    
    print(f"Completed sweeps in {elapsed/60.} minutes")