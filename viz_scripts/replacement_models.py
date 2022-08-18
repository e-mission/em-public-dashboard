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
from sklearn.ensemble import GradientBoostingClassifier

def gbdt(data, choice_col, feature_list, kf):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []

    for train_indices, test_indices in kf.split(data.values):
        X_train, X_test = data[feature_list].values[train_indices], data[feature_list].values[test_indices]
        y_train, y_test = data[choice_col].values[train_indices], data[choice_col].values[test_indices]

        # Train random forest on training set
        rf = GradientBoostingClassifier(n_estimators=50)
        rf.fit(X_train, y_train)

        # Predict for test set
        y_pred = rf.predict(X_test)
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return rf, accuracy, f1, confusion

def rf(data, choice_col, feature_list, kf):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []

    for train_indices, test_indices in kf.split(data.values):
        X_train, X_test = data[feature_list].values[train_indices], data[feature_list].values[test_indices]
        y_train, y_test = data[choice_col].values[train_indices], data[choice_col].values[test_indices]

        # Train random forest on training set
        rf = RandomForestClassifier(n_estimators=50)
        rf.fit(X_train, y_train)

        # Predict for test set
        y_pred = rf.predict(X_test)
        accuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        f1.append(sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return rf, accuracy, f1, confusion

def mxl(data, choice_col, estimate_new=True):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []

    import biogeme.messaging as msg
    # Define level of verbosity
    logger = msg.bioMessage()
    # logger.setSilent()
    # logger.setWarning()
    logger.setGeneral()
    # logger.setDetailed()

    # y_train and y_test are also in the X dataframes as needed by Biogeme
    X_train, X_test = data, data
    y_train, y_test = data[choice_col], data[choice_col]

    # Put the variables in global namespace to make Biogeme happy
    X_train = X_train.drop(columns=['date_time','_id','cleaned_trip','user_id','is_sp'])
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
    ASC_EBIKE = Beta('ASC_EBIKE',0,None,None,0)

    # Define a random parameter, normally distributed, designed to be used
    # for Monte-Carlo simulation
    B_TIME_MOTOR = Beta('B_TIME_MOTOR',0,-1,1,0)
    B_TIME_PHYS = Beta('B_TIME_PHYS',0,-1,1,0)
    B_COST = Beta('B_COST',0,-1,-1,0)

    # It is advised not to use 0 as starting value for the following parameter.
    B_TIME_MOTOR_S = Beta('B_TIME_MOTOR_S',1,None,None,0)
    B_TIME_PHYS_S = Beta('B_TIME_PHYS_S',1,None,None,0)
    B_COST_S = Beta('B_COST_S',1,None,None,0)

    # Define random parameter, log normally distributed
    B_TIME_MOTOR_RND = -exp(B_TIME_MOTOR + B_TIME_MOTOR_S * bioDraws('B_TIME_MOTOR_RND','NORMAL'))
    B_TIME_PHYS_RND = -exp(B_TIME_PHYS + B_TIME_PHYS_S * bioDraws('B_TIME_PHYS_RND','NORMAL'))
    B_COST_RND = -exp(B_COST + B_COST_S * bioDraws('B_COST_RND','NORMAL'))

    # Utility functions
    V0 = ASC_CAR + \
    B_TIME_MOTOR_RND * tt_car + \
    B_COST_RND * cost_car

    V1 = ASC_S_CAR + \
    B_TIME_MOTOR_RND * tt_s_car + \
    B_COST_RND * cost_s_car

    V2 = ASC_RIDEHAIL + \
    B_TIME_MOTOR_RND * tt_ridehail + \
    B_COST_RND * cost_ridehail

    V3 = ASC_TRANSIT + \
    B_TIME_MOTOR_RND * tt_transit + \
    B_COST_RND * cost_transit

    V4 = ASC_P_MICRO + \
    B_TIME_PHYS_RND * tt_p_micro

    V5 = ASC_S_MICRO + \
    B_TIME_PHYS_RND * tt_s_micro + \
    B_COST_RND * cost_s_micro

    V6 = ASC_WALK + \
    B_TIME_PHYS_RND * tt_walk

    V7 = ASC_EBIKE + \
    B_TIME_PHYS_RND * tt_ebike

    # Map modes to utility functions
    V = {0: V0,
        1: V1,
        2: V2,
        3: V3,
        4: V4,
        5: V5,
        6: V6,
        7: V7}

    # Mode availability
    av = {0: av_car,
        1: av_s_car,
        2: av_ridehail,
        3: av_transit,
        4: av_p_micro,
        5: av_s_micro,
        6: av_walk,
        7: av_ebike}

    # Train the model parameters
    prob = models.logit(V, av, Variable(choice_col))
    logprob = log(MonteCarlo(prob))
    biogeme = bio.BIOGEME(database_train, logprob, numberOfDraws=100)
    biogeme.modelName = 'openpath_mxl_train'
    biogeme.generateHtml = True
    biogeme.generatePickle = True

    # Check that choices are always available
    diagnostic = database_train.checkAvailabilityOfChosenAlt(av, Variable(choice_col))
    if not diagnostic.all():
        row_indices = np.where(diagnostic == False)[0]
        print(f'Rows where the chosen alternative is not available: {row_indices}')

    # Fit model or load previous run
    if estimate_new:
        results = biogeme.estimate()
    else:
        results = res.bioResults(pickleFile='openpath_mxl_train.pickle')

    # Put the variables in global namespace to make Biogeme happy
    X_test = X_test.drop(columns=['date_time','_id','cleaned_trip','user_id','is_sp'])
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
    prob_ebike = MonteCarlo(models.logit(V, av, 7))

    simulate = {'Prob. car': prob_car,
                'Prob. s_car': prob_s_car,
                'Prob. ridehail': prob_ridehail,
                'Prob. transit': prob_transit,
                'Prob. p_micro': prob_p_micro,
                'Prob. s_micro': prob_s_micro,
                'Prob. walk': prob_walk,
                'Prob. ebike': prob_ebike}

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
                                    'Prob. walk': 6,
                                    'Prob. ebike': 7})

    # Predict for test set
    accuracy.append(sklearn.metrics.accuracy_score(y_test, prob_max))
    f1.append(sklearn.metrics.f1_score(y_test, prob_max, average='weighted'))
    confusion.append(sklearn.metrics.confusion_matrix(y_test, prob_max, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

    # Collect all model scores for comparison at the end
    return betas, accuracy, f1, confusion

def mnl(data, choice_col, kf, estimate_new=True):
    # Save metrics from each run
    accuracy = []
    f1 = []
    confusion = []

    import biogeme.messaging as msg
    # Define level of verbosity
    logger = msg.bioMessage()
    # logger.setSilent()
    # logger.setWarning()
    logger.setGeneral()
    # logger.setDetailed()

    for train_indices, test_indices in kf.split(data.values):
        # y_train and y_test are also in the X dataframes as needed by Biogeme
        X_train, X_test = data.iloc[train_indices], data.iloc[test_indices]
        y_train, y_test = data[choice_col].iloc[train_indices], data[choice_col].iloc[test_indices]

        # Put the variables in global namespace to make Biogeme happy
        X_train = X_train.drop(columns=['date_time','_id','cleaned_trip','user_id','is_sp'])
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

        V7 = ASC_EBIKE + \
        B_ASV_TT_PHYS * tt_ebike

        # Map modes to utility functions
        V = {0: V0,
            1: V1,
            2: V2,
            3: V3,
            4: V4,
            5: V5,
            6: V6,
            7: V7}

        # Mode availability
        av = {0: av_car,
            1: av_s_car,
            2: av_ridehail,
            3: av_transit,
            4: av_p_micro,
            5: av_s_micro,
            6: av_walk,
            7: av_ebike}
        
        # Train the model parameters
        logprob = models.loglogit(V, av, Variable(choice_col))
        biogeme = bio.BIOGEME(database_train, logprob)
        biogeme.modelName = 'openpath_mnl_train'
        biogeme.generateHtml = True
        biogeme.generatePickle = True

        # Fit model or load previous run
        if estimate_new:
            results = biogeme.estimate()
        else:
            results = res.bioResults(pickleFile='openpath_mnl_train.pickle')

        # Put the variables in global namespace to make Biogeme happy
        X_test = X_test.drop(columns=['date_time','_id','cleaned_trip','user_id','is_sp'])
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
        prob_ebike = models.logit(V, av, 7)

        simulate ={'Prob. car': prob_car,
                   'Prob. s_car': prob_s_car,
                   'Prob. ridehail': prob_ridehail,
                   'Prob. transit': prob_transit,
                   'Prob. p_micro': prob_p_micro,
                   'Prob. s_micro': prob_s_micro,
                   'Prob. walk': prob_walk,
                   'Prob. ebike': prob_ebike}

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
                                     'Prob. walk': 6,
                                     'Prob. ebike': 7})

        # Predict for test set
        accuracy.append(sklearn.metrics.accuracy_score(y_test, prob_max))
        f1.append(sklearn.metrics.f1_score(y_test, prob_max, average='weighted'))
        confusion.append(sklearn.metrics.confusion_matrix(y_test, prob_max, labels=[0,1,2,3,4,5,6,7], normalize='pred'))

        # Collect all model scores for comparison at the end
        return betas, accuracy, f1, confusion
