import pandas as pd
import numpy as np
import sys
from collections import defaultdict

import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.tcquery as esttc
import emission.core.wrapper.localdate as ecwl

# Module for pretty-printing outputs (e.g. head) to help users
# understand what is going on
# However, this means that this module can only be used in an ipython notebook

import IPython.display as disp

import emission.core.get_database as edb

def no_traceback_handler(exception_type, exception, traceback):
    print("%s: %s" % (exception_type.__name__, exception), file=sys.stderr)

def get_time_query(year, month):
    if year is None and month is None:
        return None

    if month is None:
        assert year is not None
        query_ld = ecwl.LocalDate({"year": year})
    else:
        assert year is not None and month is not None
        query_ld = ecwl.LocalDate({"year": year, "month": month})
    tq = esttc.TimeComponentQuery("data.start_local_dt", query_ld, query_ld)
    return tq

def get_participant_uuids(program, load_test_users):
    """
        Get the list of non-test users in the current program.
        Note that the "program" parameter is currently a NOP and should be removed in
        conjunction with modifying the notebooks.
    """
    all_users = pd.json_normalize(list(edb.get_uuid_db().find()))
    # CASE 1 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if len(all_users) == 0:
        return []
    if load_test_users:
        participant_list = all_users
    else:
        participant_list = all_users[np.logical_not(all_users.user_email.str.contains("_test_"))]
    participant_uuid_str = participant_list.uuid
    disp.display(participant_list.user_email)
    return participant_uuid_str

def load_all_confirmed_trips(tq):
    agg = esta.TimeSeries.get_aggregate_time_series()
    all_ct = agg.get_data_df("analysis/confirmed_trip", tq)
    print("Loaded all confirmed trips of length %s" % len(all_ct))
    disp.display(all_ct.head())
    return all_ct

def load_all_participant_trips(program, tq, load_test_users):
    participant_list = get_participant_uuids(program, load_test_users)
    all_ct = load_all_confirmed_trips(tq)
    # CASE 1 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if len(all_ct) == 0:
        return all_ct
    participant_ct_df = all_ct[all_ct.user_id.isin(participant_list)]
    print("After filtering, found %s participant trips " % len(participant_ct_df))
    disp.display(participant_ct_df.head())
    return participant_ct_df

def filter_labeled_trips(mixed_trip_df):
    # CASE 1 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if len(mixed_trip_df) == 0:
        return mixed_trip_df
    labeled_ct = mixed_trip_df[mixed_trip_df.user_input != {}]
    print("After filtering, found %s labeled trips" % len(labeled_ct))
    disp.display(labeled_ct.head())
    return labeled_ct

def expand_userinputs(labeled_ct):
    '''
    param: labeled_ct: a dataframe of confirmed trips, some of which have labels
    params: labels_per_trip: the number of labels for each trip.
        Currently, this is 2 for studies and 3 for programs, and should be 
        passed in by the notebook based on the input config.
        If used with a trip-level survey, it could be even larger.
    '''
    # CASE 1 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if len(labeled_ct) == 0:
        return labeled_ct
    label_only = pd.DataFrame(labeled_ct.user_input.to_list(), index=labeled_ct.index)
    disp.display(label_only.head())
    labels_per_trip = len(label_only.columns)
    print("Found %s columns of length %d" % (label_only.columns, labels_per_trip))
    expanded_ct = pd.concat([labeled_ct, label_only], axis=1)
    assert len(expanded_ct) == len(labeled_ct), \
        ("Mismatch after expanding labels, expanded_ct.rows = %s != labeled_ct.rows %s" %
            (len(expanded_ct), len(labeled_ct)))
    print("After expanding, columns went from %s -> %s" %
        (len(labeled_ct.columns), len(expanded_ct.columns)))
    assert len(expanded_ct.columns) == len(labeled_ct.columns) + labels_per_trip, \
        ("Mismatch after expanding labels, expanded_ct.columns = %s != labeled_ct.columns %s" %
            (len(expanded_ct.columns), len(labeled_ct.columns)))
    disp.display(expanded_ct.head())
    return expanded_ct

# CASE 2 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
unique_users = lambda df: len(df.user_id.unique()) if "user_id" in df.columns else 0
trip_label_count = lambda s, df: len(df[s].dropna()) if s in df.columns else 0

def load_viz_notebook_data(year, month, program, study_type, dynamic_labels, dic_re, dic_pur=None, include_test_users=False):
    """ Inputs:
    year/month/program/study_type = parameters from the visualization notebook
    dic_* = label mappings; if dic_pur is included it will be used to recode trip purpose

    Pipeline to load and process the data before use in visualization notebooks.
    """
    # Access database
    tq = get_time_query(year, month)
    participant_ct_df = load_all_participant_trips(program, tq, include_test_users)
    labeled_ct = filter_labeled_trips(participant_ct_df)
    expanded_ct = expand_userinputs(labeled_ct)
    expanded_ct = data_quality_check(expanded_ct)

    # Change meters to miles
    # CASE 2 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if "distance" in expanded_ct.columns:
        unit_conversions(expanded_ct)
    
    # Extract translations key
    dic_translations = dict()

    if "translations" in dynamic_labels and "en" in dynamic_labels["translations"]:
        dic_translations = dynamic_labels["translations"]["en"]
        dic_translations = defaultdict(lambda: 'Other', dic_translations)

    # Select the mapping based on availability of dynamic_labels
    if dic_translations:
        dic_mapping = dic_translations
    else:
        dic_mapping = dic_re
    
    # Map new mode labels with translations dictionary from dynamic_labels
    # CASE 2 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if "mode_confirm" in expanded_ct.columns:
        expanded_ct['Mode_confirm'] = expanded_ct['mode_confirm'].map(dic_mapping)
    if study_type == 'program':
        # CASE 2 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
        if 'replaced_mode' in expanded_ct.columns:
            expanded_ct['Replaced_mode'] = expanded_ct['replaced_mode'].map(dic_mapping)
        else:
            print("This is a program, but no replaced modes found. Likely cold start case. Ignoring replaced mode mapping")
    else:
            print("This is a study, not expecting any replaced modes.")

    # Trip purpose mapping
    # CASE 2 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if dic_pur is not None and "purpose_confirm" in expanded_ct.columns:
        expanded_ct['Trip_purpose'] = expanded_ct['purpose_confirm'].map(dic_pur)

    # Document data quality
    file_suffix = get_file_suffix(year, month, program)
    quality_text = get_quality_text(participant_ct_df, expanded_ct, None, include_test_users)

    debug_df = pd.DataFrame.from_dict({
            "year": year,
            "month": month,
            "Registered_participants": len(get_participant_uuids(program, include_test_users)),
            "Participants_with_at_least_one_trip": unique_users(participant_ct_df),
            "Participant_with_at_least_one_labeled_trip": unique_users(labeled_ct),
            "Trips_with_at_least_one_label": len(labeled_ct),
            "Trips_with_mode_confirm_label": trip_label_count("Mode_confirm", expanded_ct),
            "Trips_with_trip_purpose_label": trip_label_count("Trip_purpose", expanded_ct)
            },
        orient='index', columns=["value"])

    return expanded_ct, file_suffix, quality_text, debug_df

def load_viz_notebook_sensor_inference_data(year, month, program, include_test_users=False, sensed_algo_prefix="cleaned"):
    """ Inputs:
    year/month/program = parameters from the visualization notebook

    Pipeline to load and process the data before use in sensor-based visualization notebooks.
    """
    tq = get_time_query(year, month)
    participant_ct_df = load_all_participant_trips(program, tq, include_test_users)
    expanded_ct = participant_ct_df
    expanded_ct["primary_mode_non_other"] = participant_ct_df.cleaned_section_summary.apply(lambda md: max(md["distance"], key=md["distance"].get))
    expanded_ct.primary_mode_non_other.replace({"ON_FOOT": "WALKING"}, inplace=True)
    valid_sensed_modes = ["WALKING", "BICYCLING", "IN_VEHICLE", "AIR_OR_HSR", "UNKNOWN"]
    expanded_ct["primary_mode"] = expanded_ct.primary_mode_non_other.apply(lambda pm: "OTHER" if pm not in valid_sensed_modes else pm)

    # Change meters to miles
    # CASE 2 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    if "distance" in expanded_ct.columns:
        unit_conversions(expanded_ct)

    # Document data quality
    file_suffix = get_file_suffix(year, month, program)
    quality_text = get_quality_text_sensed(expanded_ct, include_test_users)

    debug_df = pd.DataFrame.from_dict({
            "year": year,
            "month": month,
            "Registered_participants": len(get_participant_uuids(program, include_test_users)),
            "Participants_with_at_least_one_trip": unique_users(participant_ct_df),
            "Number of trips": len(participant_ct_df),
            },
        orient='index', columns=["value"])

    return expanded_ct, file_suffix, quality_text, debug_df

def add_energy_labels(expanded_ct, df_ei, dic_fuel):
    """ Inputs:
    expanded_ct = dataframe of trips that has had Mode_confirm and Replaced_mode added
    dic/df_* = label mappings for energy impact and fuel
    """
    expanded_ct['Mode_confirm_fuel']= expanded_ct['Mode_confirm'].map(dic_fuel)
    expanded_ct = energy_intensity(expanded_ct, df_ei, 'Mode_confirm')
    expanded_ct = energy_footprint_kWH(expanded_ct, 'distance_miles', 'Mode_confirm')
    expanded_ct = CO2_footprint_lb(expanded_ct, 'distance_miles', 'Mode_confirm')
    return expanded_ct

def add_energy_impact(expanded_ct, df_ei, dic_fuel):
    # Let's first calculate everything for the mode confirm
    # And then calculate everything for the replaced mode
    expanded_ct = add_energy_labels(expanded_ct, df_ei, dic_fuel)
    expanded_ct['Replaced_mode_fuel']= expanded_ct['Replaced_mode'].map(dic_fuel)
    expanded_ct = energy_intensity(expanded_ct, df_ei, 'Replaced_mode')
    # and then compute the impacts
    expanded_ct = energy_impact_kWH(expanded_ct, 'distance_miles')
    expanded_ct = CO2_impact_lb(expanded_ct, 'distance_miles')
    return expanded_ct

def get_quality_text(before_df, after_df, mode_of_interest=None, include_test_users=False):
    """ Inputs:
    before_df = dataframe prior to filtering (usually participant_ct_df)
    after_df = dataframe after filtering (usually expanded_ct)
    mode_of_interest = optional detail to include in the text string
    """
    # CASE 1 of https://github.com/e-mission/em-public-dashboard/issues/69#issuecomment-1256835867
    after_pct = (len(after_df) * 100) / len(before_df) if len(before_df) != 0 else np.nan
    cq = (len(after_df), unique_users(after_df), len(before_df), unique_users(before_df),
        after_pct, )
    interest_str = mode_of_interest + ' ' if mode_of_interest is not None else ''
    total_str = 'confirmed' if mode_of_interest is not None else ''
    user_str = 'testers and participants' if include_test_users else 'users'
    quality_text = f"Based on %s confirmed {interest_str}trips from %d {user_str}\nof %s total {total_str} trips from %d users (%.2f%%)" % cq
    print(quality_text)
    return quality_text

def get_quality_text_sensed(df, include_test_users=False):
    cq = (len(df), unique_users(df))
    user_str = 'testers and participants' if include_test_users else 'users'
    quality_text = f"Based on %s trips from %d {user_str}" % cq
    print(quality_text)
    return quality_text

def get_file_suffix(year, month, program):
    suffix = "_%04d" % year if year is not None else ""
    suffix = suffix + "_%02d" % month if month is not None else ""
    suffix = suffix + "_%s" % program if program is not None else ""
    print(suffix)
    return suffix

def data_quality_check(expanded_ct):
    '''1. Delete rows where the mode_confirm was pilot_ebike and repalced_mode was pilot_ebike.
       2. Delete rows where the mode_confirm was pilot_ebike and repalced_mode was same_mode.
       3. Replace same_mode for the mode_confirm for Energy Impact Calcualtion.'''

    # TODO: This is only really required for the initial data collection around the minipilot
    # in subsequent deployes, we removed "same mode" and "pilot_ebike" from the options, so the
    # dataset did not contain of these data quality issues

    if 'replaced_mode' in expanded_ct.columns:
        expanded_ct.drop(expanded_ct[(expanded_ct['mode_confirm'] == 'pilot_ebike') & (expanded_ct['replaced_mode'] == 'pilot_ebike')].index, inplace=True)
        expanded_ct.drop(expanded_ct[(expanded_ct['mode_confirm'] == 'pilot_ebike') & (expanded_ct['replaced_mode'] == 'same_mode')].index, inplace=True)
        expanded_ct['replaced_mode'] = np.where(expanded_ct['replaced_mode'] == 'same_mode',expanded_ct['mode_confirm'], expanded_ct['replaced_mode'])
    
    return expanded_ct

def unit_conversions(df):
    df['distance_miles']= df["distance"]*0.00062 #meters to miles

def energy_intensity(trip_df,mode_intensity_df,col):
    """ Inputs:
    trip_df = dataframe with data
    mode_intensity_df = dataframe with energy/cost/time factors
    col = the column for which we want to map the intensity
    """

    mode_intensity_df = mode_intensity_df.copy()
    mode_intensity_df[col] = mode_intensity_df['mode']
    dic_ei_factor = dict(zip(mode_intensity_df[col],mode_intensity_df['energy_intensity_factor']))
    dic_CO2_factor = dict(zip(mode_intensity_df[col],mode_intensity_df['CO2_factor']))
    dic_ei_trip = dict(zip(mode_intensity_df[col],mode_intensity_df['(kWH)/trip']))

    trip_df['ei_'+col] = trip_df[col].map(dic_ei_factor)
    trip_df['CO2_'+col] = trip_df[col].map(dic_CO2_factor)
    trip_df['ei_trip_'+col] = trip_df[col].map(dic_ei_trip)
    return trip_df

def energy_footprint_kWH(df,distance,col):
    """ Inputs:
    df = dataframe with data
    distance = distance in miles
    col = Replaced_mode or Mode_confirm
    """
    conditions_col = [(df[col+'_fuel'] =='gasoline'),
                       (df[col+'_fuel'] == 'diesel'),
                       (df[col+'_fuel'] == 'electric')]
    gasoline_col = (df[distance]*df['ei_'+col]*0.000293071) # 1 BTU = 0.000293071 kWH
    diesel_col   = (df[distance]*df['ei_'+col]*0.000293071)
    electric_col = (df[distance]*df['ei_'+col])+ df['ei_trip_'+col]
    values_col = [gasoline_col,diesel_col,electric_col]
    df[col+'_EI(kWH)'] = np.select(conditions_col, values_col)
    return df

def energy_impact_kWH(df,distance):
    if 'Mode_confirm_EI(kWH)' not in df.columns:
        print("Mode confirm footprint not found, computing before impact")
        df = energy_footprint_kWH(df, distance, "Mode_confirm")
    df = energy_footprint_kWH(df, distance, "Replaced_mode")
    df['Energy_Impact(kWH)']  = round((df['Replaced_mode_EI(kWH)'] - df['Mode_confirm_EI(kWH)']),3)
    return df

def CO2_footprint_lb(df, distance, col):
    """ Inputs:
    df = dataframe with data
    distance = distance in miles
    col = Replaced_mode or Mode_confirm
    """
    conditions_col = [(df[col+'_fuel'] =='gasoline'),
                       (df[col+'_fuel'] == 'diesel'),
                       (df[col+'_fuel'] == 'electric')]
   
    gasoline_col = (df[distance]*df['ei_'+col]*0.000001)* df['CO2_'+col]
    diesel_col   = (df[distance]*df['ei_'+col]*0.000001)* df['CO2_'+col]
    electric_col = (((df[distance]*df['ei_'+col])+df['ei_trip_'+col])*0.001)*df['CO2_'+col]

    values_col = [gasoline_col,diesel_col,electric_col]
    df[col+'_lb_CO2'] = np.select(conditions_col, values_col)
    return df
    
  
def CO2_impact_lb(df,distance):
    if 'Mode_confirm_lb_CO2' not in df.columns:
        print("Mode confirm footprint not found, computing before impact")
        df = CO2_footprint_lb(df, distance, "Mode_confirm")
    df = CO2_footprint_lb(df, distance, "Replaced_mode")
    df['CO2_Impact(lb)']  = round((df['Replaced_mode_lb_CO2'] - df['Mode_confirm_lb_CO2']),3)
    return df
