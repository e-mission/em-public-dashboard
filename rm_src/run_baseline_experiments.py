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

    
class SPLIT_TYPE(Enum):
    INTRA_USER = 0
    INTER_USER = 1
    TARGET = 2
    MODE = 3
    INTER_USER_STATIC = 4
    

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

        return X_tr, X_te
    
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
    
    elif how == SPLIT_TYPE.INTER_USER_STATIC:
        
        train_ids = ['810be63d084746e3b7da9d943dd88e8c', 'bf774cbe6c3040b0a022278d36a23f19', '8a8332a53a1b4cdd9f3680434e91a6ef', 
                     '5ad862e79a6341f69f28c0096fe884da', '7f89656bd4a94d12ad8e5ad9f0afecaf', 'fbaa338d7cd7457c8cad4d0e60a44d18', 
                     '3b25446778824941a4c70ae5774f4c68', '28cb1dde85514bbabfd42145bdaf7e0a', '3aeb5494088542fdaf798532951aebb0', 
                     '531732fee3c24366a286d76eb534aebc', '950f4287bab5444aa0527cc23fb082b2', '737ef8494f26407b8b2a6b1b1dc631a4', 
                     'e06cf95717f448ecb81c440b1b2fe1ab', '7347df5e0ac94a109790b31ba2e8a02a', 'bd9cffc8dbf1402da479f9f148ec9e60', 
                     '2f3b66a5f98546d4b7691fba57fa640f', 'f289f7001bd94db0b33a7d2e1cd28b19', '19a043d1f2414dbcafcca44ea2bd1f19', 
                     '68788082836e4762b26ad0877643fdcf', '4e8b1b7f026c4384827f157225da13fa', '703a9cee8315441faff7eb63f2bfa93f', 
                     'add706b73839413da13344c355dde0bb', '47b5d57bd4354276bb6d2dcd1438901d', 'e4cfb2a8f600426897569985e234636e', 
                     '0154d71439284c34b865e5a417cd48af', '234f4f2366244fe682dccded2fa7cc4e', '0d0ae3a556414d138c52a6040a203d24', 
                     '44c10f66dec244d6b8644231d4a8fecb', '30e9b141d7894fbfaacecd2fa18929f9', '0eb313ab00e6469da78cc2d2e94660fb', 
                     'fc51d1258e4649ecbfb0e6ecdaeca454', 'a1954793b1454b2f8cf95917d7547169', '6656c04c6cba4c189fed805eaa529741', 
                     '6a0f3653b80a4c949e127d6504debb55', 'dfe5ca1bb0854b67a6ffccad9565d669', '8b1f3ba43de945bea79de6a81716ad04', 
                     'cde34edb8e3a4278a18e0adb062999e5', '6d96909e5ca442ccb5679d9cdf3c8f5b', 'a60a64d82d1c439a901b683b73a74d73', 
                     '60e6a6f6ed2e4e838f2bbed6a427028d', '88041eddad7542ea8c92b30e5c64e198', '1635c003b1f94a399ebebe21640ffced', 
                     '1581993b404a4b9c9ca6b0e0b8212316', 'b1aed24c863949bfbfa3a844ecf60593', '4b89612d7f1f4b368635c2bc48bd7993', 
                     'eb2e2a5211564a9290fcb06032f9b4af', '26767f9f3da54e93b692f8be6acdac43', '8a98e383a2d143e798fc23869694934a', 
                     'b346b83b9f7c4536b809d5f92074fdae', 'd929e7f8b7624d76bdb0ec9ada6cc650', '863e9c6c8ec048c4b7653f73d839c85b', 
                     'f50537eb104e4213908f1862c8160a3e', '4a9db5a9bac046a59403b44b883cc0ba', 'cded005d5fd14c64a5bba3f5c4fe8385', 
                     'c7ce889c796f4e2a8859fa2d7d5068fe', '405b221abe9e43bc86a57ca7fccf2227', '0b3e78fa91d84aa6a3203440143c8c16', 
                     'fbff5e08b7f24a94ab4b2d7371999ef7', 'e35e65107a34496db49fa5a0b41a1e9e', 'd5137ebd4f034dc193d216128bb7fc9a', 
                     '3f7f2e536ba9481e92f8379b796ad1d0', 'dc75e0b776214e1b9888f6abd042fd95', 'b41dd7d7c6d94fe6afe2fd26fa4ac0bd', 
                     'eec6936e1ac347ef9365881845ec74df', '8c7d261fe8284a42a777ffa6f380ba3b', '4baf8c8af7b7445e9067854065e3e612', 
                     'c6e4db31c18b4355b02a7dd97deca70b', 'f0db3b1999c2410ba5933103eca9212f', '487e20ab774742378198f94f5b5b0b43', 
                     'dc1ed4d71e3645d0993885398d5628ca', '8c3c63abb3ec4fc3a61e7bf316ee4efd', '15eb78dd6e104966ba6112589c29dc41', 
                     'c23768ccb817416eaf08be487b2e3643', 'ecd2ae17d5184807abd87a287115c299', '71f21d53b655463784f3a3c63c56707b', 
                     '2931e0a34319495bbb5898201a54feb5', '92bde0d0662f45ac864629f486cffe77', '42b3ee0bc02a481ab1a94644a8cd7a0d', 
                     '15aa4ba144a34b8b8079ed7e049d84df', '509b909390934e988eb120b58ed9bd8c', '14103cda12c94642974129989d39e50d', 
                     '8b0876430c2641bcaea954ea00520e64', 'baa4ff1573ae411183e10aeb17c71c53', '14fe8002bbdc4f97acbd1a00de241bf6', 
                     '1b7d6dfea8464bcab9321018b10ec9c9', '487ad897ba93404a8cbe5de7d1922691', '5182d93d69754d7ba06200cd1ac5980a', 
                     '91f3ca1c278247f79a806e49e9cc236f', 'e66e63b206784a559d977d4cb5f1ec34', '840297ae39484e26bfebe83ee30c5b3e', 
                     'c6807997194c4c528a8fa8c1f6ee1595', '802667b6371f45b29c7abb051244836a', 'b2bbe715b6a14fd19f751cae8adf6b4e', 
                     'feb1d940cd3647d1a101580c2a3b3f8c', '1b9883393ab344a69bc1a0fab192a94c', 'ac604b44fdca482fb753034cb55d1351', 
                     'f446bf3102ff4bd99ea1c98f7d2f7af0', 'c2c5d4b9a607487ea405a99c721079d4', '85ddd3c34c58407392953c47a32f5428', 
                     'd51de709f95045f8bacf473574b96ba5', '6373dfb8cb9b47e88e8f76adcfadde20', '313d003df34b4bd9823b3474fc93f9f9', 
                     '53e78583db87421f8decb529ba859ca4', '8fdc9b926a674a9ea07d91df2c5e06f2', '90480ac60a3d475a88fbdab0a003dd5d', 
                     '7559c3f880f341e898a402eba96a855d', '19a4c2cf718d40588eb96ac25a566353', 'f4427cccaa9442b48b42bedab5ab648e', 
                     'e192b8a00b6c422296851c93785deaf7', '355e25bdfc244c5e85d358e39432bd44', 'a0c3a7b410b24e18995f63369a31d123', 
                     '03a395b4d8614757bb8432b4984559b0', 'a2d48b05d5454d428c0841432c7467b6', '3d981e617b304afab0f21ce8aa6c9786', 
                     '2cd5668ac9054e2eb2c88bb4ed94bc6d', 'd7a732f4a8644bcbb8dedfc8be242fb2', '367eb90b929d4f6e9470d15c700d2e3f', 
                     'e049a7b2a6cb44259f907abbb44c5abc', 'a231added8674bef95092b32bc254ac8', 'e88a8f520dde445484c0a9395e1a0599',
                     'cba570ae38f341faa6257342727377b7', '97953af1b97d4e268c52e1e54dcf421a', 'd200a61757d84b1dab8fbac35ff52c28', 
                     'fc68a5bb0a7b4b6386b3f08a69ead36f', '4a8210aec25e443391efb924cc0e5f23', '903742c353ce42c3ad9ab039fc418816', 
                     '2114e2a75304475fad06ad201948fbad', 'ac917eae407c4deb96625dd0dc2f2ba9', '3dddfb70e7cd40f18a63478654182e9a', 
                     'd3735ba212dd4c768e1675dca7bdcb6f', '7abe572148864412a33979592fa985fb', 'd3dff742d07942ca805c2f72e49e12c5' 
                     ]
        
        X_tr = data.loc[data.user_id.isin(train_ids), :]
        X_te = data.loc[~data.user_id.isin(train_ids), :]
        
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


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--max-iters', default=10000, type=int)
#     return parser.parse_args()

            
if __name__ == "__main__":
    data = pd.read_csv('../data/ReplacedMode_Fix_02142024.csv')
    data.drop_duplicates(inplace=True)
    
    print("Beginning sweeps.")
    
    # args = parse_args()
    
    start = perf_counter()
    sweep_number = 1
    
    root = Path('../benchmark_results')
    if not root.exists():
        root.mkdir()
    
    for split in [SPLIT_TYPE.INTER_USER, SPLIT_TYPE.INTRA_USER, SPLIT_TYPE.TARGET, SPLIT_TYPE.MODE]:
        for drop in [True, False]:
            for location_drop in [True, False]:
                kwargs = {
                    'drop_s_micro': drop,
                    'split': split,
                    'drop_location': location_drop
                }
                
                dir_name = root / f'benchmark_{sweep_number}'
                
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