import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, exp, PanelLikelihoodTrajectory, bioDraws, log, MonteCarlo
import biogeme.results as res
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier



def get_kf_indices(kf, train_data, test_data):
    # Train/test datasets can be different sizes making true k-fold impossible. Separate k-folds give virtually identical result.
    if kf is not None:
        train_indices = []
        test_indices = []
        for x in kf.split(train_data.values):
            train_indices.append(x[0])
        for x in kf.split(test_data.values):
            test_indices.append(x[1])
    else:
        # Use all data (single run) if kf not passed
        train_indices = [[x for x in range(0,len(train_data))]]
        test_indices = [[x for x in range(0,len(test_data))]]
    return train_indices, test_indices

def random_forest(train_data, train_col, test_data, test_col, feature_list, kf=None):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
    train_indices, test_indices = get_kf_indices(kf, train_data, test_data)

    for i in range(0,len(train_indices)):
        X_train, X_test = train_data[feature_list].values[train_indices[i]], test_data[feature_list].values[test_indices[i]]
        y_train, y_test = train_data[train_col].values[train_indices[i]], test_data[test_col].values[test_indices[i]]

        # Train random forest on training set
        rf = RandomForestClassifier(n_estimators=50)
        rf.fit(X_train, y_train)

        # Predict for test set
        y_pred = rf.predict(X_test)
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6], normalize='pred'))

    # Collect all model scores for comparison at the end
    return rf, accuracy, f1, confusion

def mxl(train_data, train_col, test_data, test_col, kf=None, estimate_new=True):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
    train_indices, test_indices = get_kf_indices(kf, train_data, test_data)

    for i in range(0, len(train_indices)):
        # y_train and y_test are also in the X dataframes as needed by Biogeme
        X_train, X_test = train_data.iloc[train_indices[i]], test_data.iloc[test_indices[i]]
        y_train, y_test = train_data[train_col].iloc[train_indices[i]], test_data[test_col].iloc[test_indices[i]]

        # Put the variables in global namespace to make Biogeme happy
        X_train = X_train.drop(columns=['date_time','_id','cleaned_trip','user_id'])
        database_train = db.Database('openpath_mxl_train', X_train)
        globals().update(database_train.variables)

        # Alternative specific constants
        ASC_CAR = Beta('ASC_CAR',0,None,None,1)
        ASC_S_CAR = Beta('ASC_S_CAR',0,None,None,0)
        ASC_RIDEHAIL = Beta('ASC_RIDEHAIL',0,None,None,0)
        ASC_TRANSIT = Beta('ASC_TRANSIT',0,None,None,0)
        ASC_P_MICRO = Beta('ASC_P_MICRO',0,None,None,0)
        ASC_S_MICRO = Beta('ASC_S_MICRO',0,None,None,0)
        ASC_WALK = Beta('ASC_WALK',0,None,None,0)

        # Define a random parameter, normally distributed, designed to be used
        # for Monte-Carlo simulation
        B_TIME_MOTOR = Beta('B_TIME_MOTOR', 0, None, None, 0)
        B_TIME_PHYS = Beta('B_TIME_PHYS', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)

        # It is advised not to use 0 as starting value for the following parameter.
        B_TIME_MOTOR_S = Beta('B_TIME_MOTOR_S', 1, None, None, 0)
        B_TIME_PHYS_S = Beta('B_TIME_PHYS_S', 1, None, None, 0)
        B_COST_S = Beta('B_COST_S', 1, None, None, 0)

        # Define random parameter, log normally distributed
        B_TIME_MOTOR_RND = -exp(B_TIME_MOTOR + B_TIME_MOTOR_S * bioDraws('B_TIME_MOTOR_RND', 'NORMAL'))
        B_TIME_PHYS_RND = -exp(B_TIME_PHYS + B_TIME_PHYS_S * bioDraws('B_TIME_PHYS_RND', 'NORMAL'))
        B_COST_RND = -exp(B_COST + B_COST_S * bioDraws('B_COST_RND', 'NORMAL'))

        # Utility functions
        V0 = ASC_CAR + \
        B_TIME_MOTOR_RND * tt_car + \
        B_COST * cost_car

        V1 = ASC_S_CAR + \
        B_TIME_MOTOR_RND * tt_s_car + \
        B_COST * cost_s_car

        V2 = ASC_RIDEHAIL + \
        B_TIME_MOTOR_RND * tt_ridehail + \
        B_COST * cost_ridehail

        V3 = ASC_TRANSIT + \
        B_TIME_MOTOR_RND * tt_transit + \
        B_COST * cost_transit

        V4 = ASC_P_MICRO + \
        B_TIME_PHYS_RND * tt_p_micro

        V5 = ASC_S_MICRO + \
        B_TIME_PHYS_RND * tt_s_micro + \
        B_COST * cost_s_micro

        V6 = ASC_WALK + \
        B_TIME_PHYS_RND * tt_walk

        # Map modes to utility functions
        V = {0: V0,
            1: V1,
            2: V2,
            3: V3,
            4: V4,
            5: V5,
            6: V6}

        # Mode availability
        av = {0: av_car,
            1: av_s_car,
            2: av_ridehail,
            3: av_transit,
            4: av_p_micro,
            5: av_s_micro,
            6: av_walk}

        # Train the model parameters
        prob = models.logit(V, av, Variable(train_col))
        logprob = log(MonteCarlo(prob))
        biogeme = bio.BIOGEME(database_train, logprob, numberOfDraws=100)
        biogeme.modelName = 'openpath_mxl_train'
        biogeme.generateHtml = True
        biogeme.generatePickle = True

        if estimate_new:
            results = biogeme.estimate()
        else:
            results = res.bioResults(pickleFile='openpath_mxl_train.pickle')

        # Put the variables in global namespace to make Biogeme happy
        X_test = X_test.drop(columns=['date_time','_id','cleaned_trip','user_id'])
        database_test = db.Database('openpath_mxl_test', X_test)
        globals().update(database_test.variables)

        # Assemble utility functions for testing modes
        prob_car = MonteCarlo(models.logit(V, av, 0))
        prob_s_car = MonteCarlo(models.logit(V, av, 1))
        prob_ridehail = MonteCarlo(models.logit(V, av, 2))
        prob_transit = MonteCarlo(models.logit(V, av, 3))
        prob_p_micro = MonteCarlo(models.logit(V, av, 4))
        prob_s_micro = MonteCarlo(models.logit(V, av, 5))
        prob_walk = MonteCarlo(models.logit(V, av, 6))

        simulate = {'Prob. car': prob_car,
                    'Prob. s_car': prob_s_car,
                    'Prob. ridehail': prob_ridehail,
                    'Prob. transit': prob_transit,
                    'Prob. p_micro': prob_p_micro,
                    'Prob. s_micro': prob_s_micro,
                    'Prob. walk': prob_walk}

        # Get results of last run (or loaded run)
        betas = results.getBetaValues()

        # Calculate utility values for each row in the test database
        biogeme = bio.BIOGEME(database_test, simulate, numberOfDraws=100)
        biogeme.modelName = 'openpath_mxl_test'
        simulatedValues = biogeme.simulate(betas)

        # Test predicting maximum mode utility as choice
        # Identify the column of highest probability, replace with number corresponding to the mode
        prob_max = simulatedValues.idxmax(axis=1)
        prob_max = prob_max.replace({'Prob. car': 0,
                                    'Prob. s_car': 1,
                                    'Prob. ridehail': 2,
                                    'Prob. transit': 3,
                                    'Prob. p_micro': 4,
                                    'Prob. s_micro': 5,
                                    'Prob. walk': 6})

        # Predict for test set
        accuracy.append(sklearn.metrics.accuracy_score(y_test, prob_max))
        f1.append(sklearn.metrics.f1_score(y_test, prob_max, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, prob_max, labels=[0,1,2,3,4,5,6], normalize='pred'))

        # Collect all model scores for comparison at the end
        return results, accuracy, f1, confusion

def mnl(train_data, train_col, test_data, test_col, kf=None, estimate_new=True):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []
    train_indices, test_indices = get_kf_indices(kf, train_data, test_data)

    for i in range(0, len(train_indices)):
        # y_train and y_test are also in the X dataframes as needed by Biogeme
        X_train, X_test = train_data.iloc[train_indices[i]], test_data.iloc[test_indices[i]]
        y_train, y_test = train_data[train_col].iloc[train_indices[i]], test_data[test_col].iloc[test_indices[i]]

        # Put the variables in global namespace to make Biogeme happy
        X_train = X_train.drop(columns=['date_time','_id','cleaned_trip','user_id'])
        database_train = db.Database('openpath_mnl_train', X_train)
        globals().update(database_train.variables)

        # Alternative specific constants
        ASC_CAR = Beta('ASC_CAR',0,None,None,1)
        ASC_S_CAR = Beta('ASC_S_CAR',0,None,None,0)
        ASC_RIDEHAIL = Beta('ASC_RIDEHAIL',0,None,None,0)
        ASC_TRANSIT = Beta('ASC_TRANSIT',0,None,None,0)
        ASC_P_MICRO = Beta('ASC_P_MICRO',0,None,None,0)
        ASC_S_MICRO = Beta('ASC_S_MICRO',0,None,None,0)
        ASC_WALK = Beta('ASC_WALK',0,None,None,0)
        ASC_EBIKE = Beta('ASC_EBIKE',0,None,None,0)

        # Trip parameters
        B_COST = Beta('B_COST',0,None,None,0)
        B_TT = Beta('B_TT',0,None,None,0)

        # Mode parameters
        B_ASV_TT_MOTOR = Beta('B_ASV_TT_MOTOR',0,None,None,0)
        B_ASV_TT_PHYS = Beta('B_ASV_TT_PHYS',0,None,None,0)

        # Utility functions
        V0 = ASC_CAR + \
        B_COST * cost_car + \
        B_ASV_TT_MOTOR * tt_car

        V1 = ASC_S_CAR + \
        B_COST * cost_s_car + \
        B_ASV_TT_MOTOR * tt_s_car

        V2 = ASC_RIDEHAIL + \
        B_COST * cost_ridehail + \
        B_ASV_TT_MOTOR * tt_ridehail

        V3 = ASC_TRANSIT + \
        B_COST * cost_transit + \
        B_ASV_TT_MOTOR * tt_transit

        V4 = ASC_P_MICRO + \
        B_ASV_TT_PHYS * tt_p_micro

        V5 = ASC_S_MICRO + \
        B_COST * cost_s_micro + \
        B_ASV_TT_PHYS * tt_s_micro

        V6 = ASC_WALK + \
        B_ASV_TT_PHYS * tt_walk

        # Map modes to utility functions
        V = {0: V0,
            1: V1,
            2: V2,
            3: V3,
            4: V4,
            5: V5,
            6: V6}

        # Mode availability
        av = {0: av_car,
            1: av_s_car,
            2: av_ridehail,
            3: av_transit,
            4: av_p_micro,
            5: av_s_micro,
            6: av_walk}
        
        # Train the model parameters
        logprob = models.loglogit(V, av, Variable(train_col))
        biogeme = bio.BIOGEME(database_train, logprob)
        biogeme.modelName = 'openpath_mnl_train'
        biogeme.generateHtml = True
        biogeme.generatePickle = True

        if estimate_new:
            results = biogeme.estimate()
        else:
            results = res.bioResults(pickleFile='openpath_mnl_train.pickle')

        # Put the variables in global namespace to make Biogeme happy
        X_test = X_test.drop(columns=['date_time','_id','cleaned_trip','user_id'])
        database_test = db.Database('openpath_mnl_test', X_test)
        globals().update(database_test.variables)
        
        # Assemble utility functions for testing modes
        prob_car = models.logit(V, av, 0)
        prob_s_car = models.logit(V, av, 1)
        prob_ridehail = models.logit(V, av, 2)
        prob_transit = models.logit(V, av, 3)
        prob_p_micro = models.logit(V, av, 4)
        prob_s_micro = models.logit(V, av, 5)
        prob_walk = models.logit(V, av, 6)

        simulate ={'Prob. car': prob_car,
                   'Prob. s_car': prob_s_car,
                   'Prob. ridehail': prob_ridehail,
                   'Prob. transit': prob_transit,
                   'Prob. p_micro': prob_p_micro,
                   'Prob. s_micro': prob_s_micro,
                   'Prob. walk': prob_walk}

        betas = results.getBetaValues()

        # Calculate utility values for each row in the test database
        biogeme = bio.BIOGEME(database_test, simulate)
        biogeme.modelName = 'openpath_mnl_test'
        simulatedValues = biogeme.simulate(betas)

        # Test predicting maximum mode utility as choice
        # Identify the column of highest probability, replace with number corresponding to the mode
        prob_max = simulatedValues.idxmax(axis=1)
        prob_max = prob_max.replace({'Prob. car': 0,
                                    'Prob. s_car': 1,
                                    'Prob. ridehail': 2,
                                    'Prob. transit': 3,
                                    'Prob. p_micro': 4,
                                    'Prob. s_micro': 5,
                                    'Prob. walk': 6})

        # Predict for test set
        accuracy.append(sklearn.metrics.accuracy_score(y_test, prob_max))
        f1.append(sklearn.metrics.f1_score(y_test, prob_max, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, prob_max, labels=[0,1,2,3,4,5,6], normalize='pred'))

        # Collect all model scores for comparison at the end
        return results, accuracy, f1, confusion
