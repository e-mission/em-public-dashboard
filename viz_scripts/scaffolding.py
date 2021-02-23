import pandas as pd
import numpy as np


import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.tcquery as esttc
import emission.core.wrapper.localdate as ecwl

# Module for pretty-printing outputs (e.g. head) to help users
# understand what is going on
# However, this means that this module can only be used in an ipython notebook

import IPython.display as disp

import emission.core.get_database as edb

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

def get_participant_uuids(program):
    """
        Get the list of participant UUIDs for the specified program.
        Note that the "program" parameter is currently a NOP but will be enabled
        once we have other programs start
    """
    participant_uuid_obj = list(edb.get_profile_db().find({"install_group": "participant"}, {"user_id": 1, "_id": 0}))
    participant_uuid_str = [u["user_id"] for u in participant_uuid_obj]
    disp.display(participant_uuid_str)
    return participant_uuid_str

def load_all_confirmed_trips(tq):
    agg = esta.TimeSeries.get_aggregate_time_series()
    all_ct = agg.get_data_df("analysis/confirmed_trip", tq)
    print("Loaded all confirmed trips of length %s" % len(all_ct))
    disp.display(all_ct.head())
    return all_ct

def load_all_participant_trips(program, tq):
    participant_list = get_participant_uuids(program)
    all_ct = load_all_confirmed_trips(tq)
    participant_ct_df = all_ct[all_ct.user_id.isin(participant_list)]
    print("After filtering, found %s participant trips " % len(participant_ct_df))
    disp.display(participant_ct_df.head())
    return participant_ct_df

def filter_labeled_trips(mixed_trip_df):
    labeled_ct = mixed_trip_df[mixed_trip_df.user_input != {}]
    print("After filtering, found %s labeled trips" % len(labeled_ct))
    disp.display(labeled_ct.head())
    return labeled_ct

def expand_userinputs(labeled_ct):
    label_only = pd.DataFrame(labeled_ct.user_input.to_list(), index=labeled_ct.index)
    disp.display(label_only.head())
    expanded_ct = pd.concat([labeled_ct, label_only], axis=1)
    assert len(expanded_ct) == len(labeled_ct), \
        ("Mismatch after expanding labels, expanded_ct.rows = %s != labeled_ct.rows" %
            (len(expanded_ct), len(labeled_ct)))
    print("After expanding, columns went from %s -> %s" %
        (len(labeled_ct.columns), len(expanded_ct.columns)))
    assert len(expanded_ct.columns) == len(labeled_ct.columns) + 3, \
        ("Mismatch after expanding labels, expanded_ct.columns = %s != labeled_ct.rows" %
            (len(expanded_ct.columns), len(labeled_ct.columns)))
    disp.display(expanded_ct.head())
    return expanded_ct

def get_quality_text(participant_ct_df, expanded_ct):
    cq = (len(expanded_ct), len(expanded_ct.user_id.unique()), len(participant_ct_df), len(participant_ct_df.user_id.unique()), (len(expanded_ct) * 100) / len(participant_ct_df), )
    quality_text = "Based on %s confirmed trips from %d users\nof %s total trips from %d users (%.2f%%)" % cq
    print(quality_text)
    return quality_text

def get_file_suffix(year, month, program):
    suffix = "%04d" % year if year is not None else ""
    suffix = suffix + "%02d" % month if month is not None else ""
    suffix = suffix + "%s" % program if program is not None else ""
    print(suffix)
    return suffix

def get_quality_text_ebike(all_confirmed_df, ebike_ct_df):
    cq = (len(ebike_ct_df), len(ebike_ct_df.user_id.unique()), len(all_confirmed_df), len(all_confirmed_df.user_id.unique()), (len(ebike_ct_df) * 100) / len(all_confirmed_df), )
    quality_text = "Based on %s eBike trips from %d users\nof %s confirmed trips (all modes) from %d users (%.2f%%)" % cq
    print(quality_text)
    return quality_text

def data_quality_check(expanded_ct):
    '''1. Delete rows where the mode_confirm was pilot_ebike and repalced_mode was pilot_ebike.
       2. Delete rows where the mode_confirm was pilot_ebike and repalced_mode was same_mode.
       3. Replace same_mode for the mode_confirm for Energy Impact Calcualtion.'''
    
    expanded_ct.drop(expanded_ct[(expanded_ct['mode_confirm'] == 'pilot_ebike') & (expanded_ct['replaced_mode'] == 'pilot_ebike')].index, inplace=True)
    expanded_ct.drop(expanded_ct[(expanded_ct['mode_confirm'] == 'pilot_ebike') & (expanded_ct['replaced_mode'] == 'same_mode')].index, inplace=True)
    expanded_ct['replaced_mode'] = np.where(expanded_ct['replaced_mode'] == 'same_mode',expanded_ct['mode_confirm'], expanded_ct['replaced_mode'])
    
    return expanded_ct

def energy_cal(expanded_ct,dic_ei):
    expanded_ct['ei_mode_confirm'] = expanded_ct['Mode_confirm'].map(dic_ei)
    expanded_ct['ei_replaced_mode'] = expanded_ct['Replaced_mode'].map(dic_ei)
    expanded_ct['distance_miles']= expanded_ct['distance']*0.00062
    expanded_ct['Energy_Impact(kWH)']= round((expanded_ct['ei_replaced_mode']-expanded_ct['ei_mode_confirm'])*expanded_ct['distance_miles'],3) 
    
    return expanded_ct

class Missing_dict(dict):
    '''Map the data from purpose_confirm column:
       if the key is in the dictionary,
       it will be replaced by its value, otherwise key)'''
    def __init__(self,*arg,**kw):
        super(Missing_dict, self).__init__(*arg, **kw)
    def __missing__(self, key) :
        return key
    
    