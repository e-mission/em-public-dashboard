import warnings
warnings.simplefilter(action='ignore', category=Warning)

import os
import numpy as np
import pandas as pd
import pickle
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, log_loss, r2_score

SEED = 13210

class BayesianCV:
    def __init__(self, data):

        init_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        X = data.drop(columns=['target'])
        groups = data.user_id.values
        y = data.target.values
        
        for train_ix, test_ix in init_splitter.split(X, y, groups):
            train = data.iloc[train_ix, :]
            test = data.iloc[test_ix, :]
            
            break
        
        # Can't have split, so let it happen for two times.
        # train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data.target)
        
        print("Train-test split done.")
        
        # Estimate the test durations using the train data.
        params, train = self._get_duration_estimate(train, 'train', None)
        _, test = self._get_duration_estimate(test, 'test', params)

        # We drop the training duration estimates since we will be re-computing them during CV.
        train.drop(columns=[c for c in train.columns if 'tt_' in c], inplace=True)
        
        # This is out final train and test data.
        self.data = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        
        self._optimizer = self._setup_optimizer()
        

    def _drop_columns(self, df: pd.DataFrame):
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
            'available_modes', 'section_coordinates_argmax', 'section_mode_argmax',
            'start_lat', 'start_lng', 'end_lat', 'end_lng'
        ]

        # Drop section_mode_argmax and available_modes.
        return df.drop(
            columns=to_drop, 
            inplace=False
        )
    
    
    def _get_duration_estimate(self, df: pd.DataFrame, dset: str, model_dict: dict):
    
        X_features = ['section_distance_argmax', 'age']

        if 'mph' in df.columns:
            X_features += ['mph']

        if dset == 'train' and model_dict is None:
            model_dict = dict()

        if dset == 'test' and model_dict is None:
            raise AttributeError("Expected model dict for testing.")

        if dset == 'train':
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
                    # print(f"Train R2 for {section_mode}: {r2}")

                    model_dict[section_mode]['model'] = model

        elif dset == 'test':
            for section_mode in df.section_mode_argmax.unique():
                section_data = df.loc[df.section_mode_argmax == section_mode, :]
                X = section_data[
                    X_features
                ]
                Y = section_data[['section_duration_argmax']]

                y_pred = model_dict[section_mode]['model'].predict(X)
                r2 = r2_score(y_pred=y_pred, y_true=Y.values.ravel())
                # print(f"Test R2 for {section_mode}: {r2}")

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
    
    
    def _setup_optimizer(self):
        # Define search space.
        hparam_dict = {
            # 10-500
            'n_estimators': (0.25, 3),
            # 5-150
            'max_depth': (0.5, 15),
            # 2-20
            'min_samples_split': (0.2, 2.5),
            # 1-20
            'min_samples_leaf': (0.1, 2.5),
            # as-is.
            'ccp_alpha': (0., 0.5),
            # as-is.
            'max_features': (0.1, 0.99),
            # Use clip to establish mask.
            'class_weight': (0, 1),
        }
        
        return BayesianOptimization(
            self._surrogate,
            hparam_dict
        )

    
    def _surrogate(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, ccp_alpha, max_features, class_weight):

        cw = 'balanced_subsample' if class_weight < 0.5 else 'balanced'
        
        # Builds a surrogate model using the samples hparams.
        model = RandomForestClassifier(
            n_estimators=int(n_estimators * 100),
            max_depth=int(max_depth * 10),
            min_samples_split=int(min_samples_split * 10),
            min_samples_leaf=int(min_samples_leaf * 10),
            max_features=max(min(max_features, 0.999), 1e-3),
            ccp_alpha=ccp_alpha,
            bootstrap=True,
            class_weight=cw,
            n_jobs=os.cpu_count(),
            random_state=SEED
        )
        
        fold_crossentropy = list()
        
        # Use the train split and further split in train-val.
        X = self.data.drop(columns=['target'])
        y = self.data.target.values.ravel()
        users = X.user_id.values

        gkfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

        for train_ix, test_ix in gkfold.split(X, y, users):
            
            X_train = X.iloc[train_ix, :]
            X_test = X.iloc[test_ix, :]
            
            y_train = y[train_ix]
            y_test = y[test_ix]
            
            # Re-estimate durations.
            params, X_train = self._get_duration_estimate(X_train, 'train', None)
            _, X_test = self._get_duration_estimate(X_test, 'test', params)
            
            X_train = self._drop_columns(X_train)
            X_test = self._drop_columns(X_test)
            
            model.fit(
                X_train,
                y_train
            )
            
            # Measure performance on valid split.
            ce = log_loss(
                y_true=y_test,
                y_pred=model.predict_proba(X_test),
                labels=list(range(1, 10))
            )
            
            fold_crossentropy.append(ce)
        
        # Return the average negative crossentropy (since bayesian optimization aims to maximize an objective).
        return -np.mean(fold_crossentropy)
    
    
    def optimize(self):
        self._optimizer.maximize(n_iter=100, init_points=10)
        print("Done optimizing!")
        best_params = self._optimizer.max['params']
        best_loss = -self._optimizer.max['target']
        return best_loss, best_params


def train_final_model(params, cv_obj):
    # Construct the model using the params.
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators'] * 100),
        max_depth=int(params['max_depth'] * 10),
        min_samples_split=int(params['min_samples_split'] * 10),
        min_samples_leaf=int(params['min_samples_leaf'] * 10),
        max_features=params['max_features'],
        ccp_alpha=params['ccp_alpha'],
        bootstrap=True,
        class_weight='balanced_subsample',
        n_jobs=os.cpu_count()
    )
    
    
    X_tr = cv_obj.data.drop(columns=['target'])
    y_tr = cv_obj.data.target.values.ravel()
    
    X_te = cv_obj.test.drop(columns=['target'])
    y_te = cv_obj.test.target.values.ravel()
    
    params, X_tr = cv_obj._get_duration_estimate(X_tr, 'train', None)

    X_tr = cv_obj._drop_columns(X_tr)
    X_te = cv_obj._drop_columns(X_te)

    model.fit(
        X_tr,
        y_tr
    )
    
    model.fit(X_tr, y_tr)
    
    print(f"Train loss: {log_loss(y_true=y_tr, y_pred=model.predict_proba(X_tr))}")
    print(f"Train performance: {f1_score(y_true=y_tr, y_pred=model.predict(X_tr), average='weighted')}")
    print(f"Test loss: {log_loss(y_true=y_te, y_pred=model.predict_proba(X_te))}")
    print(f"Test performance: {f1_score(y_true=y_te, y_pred=model.predict(X_te), average='weighted')}")
    
    with open('./bayes_rf.pkl', 'wb') as f:
        f.write(pickle.dumps(model))
    
    
if __name__ == "__main__":
    data = pd.read_csv('../data/ReplacedMode_Fix_02142024.csv')
    bayes_cv = BayesianCV(data)
    best_loss, best_params = bayes_cv.optimize()
    print(f"Best loss: {best_loss}, best params: {str(best_params)}")
    train_final_model(best_params, bayes_cv)
    