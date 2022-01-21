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
    suffix = "_%04d" % year if year is not None else ""
    suffix = suffix + "_%02d" % month if month is not None else ""
    suffix = suffix + "_%s" % program if program is not None else ""
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

def unit_conversions(df):
    df['distance_miles']= df["distance"]*0.00062 #meters to miles
    df['duration_h'] = df['duration'] / 60 / 60 #seconds to hours

def energy_intensity(df,df1,distance,col1,col2):
    """Inputs:
    df = dataframe with data from CanBikeCO
    df1 = dataframe with energy factors
    distance = distance in meters
    col1 = Replaced_mode
    col2= Mode_confirm
    """

    # Create a copy of the energy_factors dataframe
    df1 = df1.copy()

    # Create a replaced mode column in df1 same as mode
    df1[col1] = df1['mode']

    # Pair energy intensity with mode
    dic_ei_factor = dict(zip(df1[col1],df1['energy_intensity_factor']))
    
    # Pair CO2_factor with mode
    dic_CO2_factor = dict(zip(df1[col1],df1['CO2_factor']))

    # Pair (KWH)/trip with mode
    dic_ei_trip = dict(zip(df1[col1],df1['(kWH)/trip']))
    
    # Create new features in data for replaced mode
    df['ei_'+col1] = df[col1].map(dic_ei_factor)
    df['CO2_'+col1] = df[col1].map(dic_CO2_factor)
    df['ei_trip_'+col1] = df[col1].map(dic_ei_trip)
    
    # Create new features in data for confirmed mode
    df1[col2] = df1[col1]
    dic_ei_factor = dict(zip(df1[col2],df1['energy_intensity_factor']))
    dic_ei_trip = dict(zip(df1[col2],df1['(kWH)/trip']))
    dic_CO2_factor = dict(zip(df1[col2],df1['CO2_factor']))
    df['ei_'+col2] = df[col2].map(dic_ei_factor)
    df['CO2_'+col2] = df[col2].map(dic_CO2_factor)
    df['ei_trip_'+col2] = df[col2].map(dic_ei_trip)
           
    return df


def cost(data, cost, dist, repm, mode):
    """
    Calculates the cost of the CanBikeCO E-bike pilot program
    
    Parameters:
        data - CanBikeCO data input
        cost - dataframe defining cost ($/PMT) for each mode
        dist - feature name in data of feature with distance in miles
        repm - feature name in data of feature with replaced mode
        mode - feature name in data of feature with confirmed mode
        
    Returns:
        data with appended cost feature for each trip in $$$ for both mode and replaced mode (float)
    """

    # Create a copy of the cost dataframe
    cost = cost.copy()

    # Create a replaced mode column in cost same as mode
    cost[repm] = cost['mode']

    # Pair cost with mode
    dic_cost__trip = dict(zip(cost[repm],cost['C($/PMT)']))
    
    # Create new features in data for replaced mode
    data['cost__trip_'+repm] = data[repm].map(dic_cost__trip)
    
    # Create new features in data for confirmed mode
    cost[mode] = cost[repm]
    dic_cost__trip = dict(zip(cost[mode],cost['C($/PMT)']))
    data['cost__trip_'+mode] = data[mode].map(dic_cost__trip)
    
    return data


def time(data, dura, dist, repm, mode):
    """
    Calculates the cost of the CanBikeCO E-bike pilot program
    
    Parameters:
        data - CanBikeCO data input
        dura - dataframe defining duration ((1/speed)/PMT) for each mode
        dist - feature name in data of feature with distance in miles
        repm - feature name in data of feature with replaced mode
        mode - feature name in data of feature with confirmed mode
        
    Returns:
        data with appended cost feature for each trip in $$$ for both mode and replaced mode (float)
    """

    # Create a copy of the dura dataframe
    dura = dura.copy()

    # Create a replaced mode column in dura same as mode
    dura[repm] = dura['mode']

    # Pair dura with mode
    dic_dura__trip = dict(zip(dura[repm],dura['C($/PMT)']))
    
    # Create new features in data for replaced mode
    data['dura__trip_'+repm] = data[repm].map(dic_dura__trip)
    
    # Create new features in data for confirmed mode
    dura[mode] = dura[repm]
    dic_dura__trip = dict(zip(dura[mode],dura['D(hours/PMT)']))
    data['dura__trip_'+mode] = data[mode].map(dic_dura__trip)
           
    return data  
   
    
    
def energy_impact_kWH(df,distance,col1,col2):
    """ 
    Purpose:
        Calculates energy intensity for each mode
        by fuel type, then calculates the diference
        between the energy intensity of replaced and
        confirmed modes.
    
    Inputs:
    df = dataframe with data
    distance = distance in miles
    col1 = Replaced_mode
    col2= Mode_confirm
    """
    

    conditions_col1 = [(df['Replaced_mode_fuel'] =='gasoline'),
                       (df['Replaced_mode_fuel'] == 'diesel'),
                       (df['Replaced_mode_fuel'] == 'electric')]
   
    conditions_col2 = [(df['Mode_confirm_fuel'] =='gasoline'),
                       (df['Mode_confirm_fuel'] == 'diesel'),
                       (df['Mode_confirm_fuel'] == 'electric')]

    gasoline_col1 = (df[distance]*df['ei_'+col1]*0.000293071) # 1 BTU = 0.000293071 kWH
    diesel_col1   = (df[distance]*df['ei_'+col1]*0.000293071)
    electric_col1 = (df[distance]*df['ei_'+col1])+ df['ei_trip_'+col1]
    
    gasoline_col2 = (df[distance]*df['ei_'+col2]*0.000293071)
    diesel_col2   = (df[distance]*df['ei_'+col2]*0.000293071)
    electric_col2 = (df[distance]*df['ei_'+col2])+ df['ei_trip_'+col2]
  
    
    values_col1 = [gasoline_col1,diesel_col1,electric_col1]
    values_col2 = [gasoline_col2,diesel_col2,electric_col2]  
    
    df[col1+'_EI(kWH)'] = np.select(conditions_col1, values_col1)
    df[col2+'_EI(kWH)'] = np.select(conditions_col2, values_col2)
    
    df['Energy_Impact(kWH)']  = round((df[col1+'_EI(kWH)'] - df[col2+'_EI(kWH)']),3)
  
    return df


def CO2_impact_lb(df,distance,col1,col2):
    """ Inputs:
    df = dataframe with data
    distance = distance in miles
    col1 = Replaced_mode
    col2= Mode_confirm
    """
 
    conditions_col1 = [(df['Replaced_mode_fuel'] =='gasoline'),
                       (df['Replaced_mode_fuel'] == 'diesel'),
                       (df['Replaced_mode_fuel'] == 'electric')]
   
    conditions_col2 = [(df['Mode_confirm_fuel'] =='gasoline'),
                       (df['Mode_confirm_fuel'] == 'diesel'),
                       (df['Mode_confirm_fuel'] == 'electric')]

  
    gasoline_col1 = (df[distance]*df['ei_'+col1]*0.000001)* df['CO2_Replaced_mode']
    diesel_col1   = (df[distance]*df['ei_'+col1]*0.000001)* df['CO2_Replaced_mode']
    electric_col1 = (((df[distance]*df['ei_'+col1])+df['ei_trip_'+col1])*0.001)*df['CO2_'+col1]
    
    gasoline_col2 = (df[distance]*df['ei_'+col2]*0.000001)* df['CO2_Mode_confirm']
    diesel_col2   = (df[distance]*df['ei_'+col2]*0.000001)* df['CO2_Mode_confirm']
    electric_col2 = (((df[distance]*df['ei_'+col2])+df['ei_trip_'+col2])*0.001)*df['CO2_'+col2]
  
    
    values_col1 = [gasoline_col1,diesel_col1,electric_col1]
    values_col2 = [gasoline_col2,diesel_col2,electric_col2]  
    
    df[col1+'_lb_CO2'] = np.select(conditions_col1, values_col1)
    df[col2+'_lb_CO2'] = np.select(conditions_col2, values_col2)
    df['CO2_Impact(lb)']  = round((df[col1+'_lb_CO2'] - df[col2+'_lb_CO2']),3)
  
    return df


def cost_impact(data, dist, repm, mode):
    """
    Calculates the cost impact of the CanBikeCO E-bike program
    
     Parameters:
        data - CanBikeCO data input
        dist - feature name in df of feature with distance in miles
        repm - feature name in df of feature with replaced mode
        mode - feature name in df of feature with confirmed mode
        
    Returns:
        data with appended cost impact feature for each trip in $$$ (float)
    """
  
    data[mode+'_cost'] = data[dist] * data['cost__trip_'+mode]
    data[repm+'_cost'] = data[dist] * data['cost__trip_'+repm]
    data['Cost_Impact($)'] = round((data[mode+'_cost'] - data[repm+'_cost']),2)

    return data


def time_impact(data, dist, repm, mode):
    """
    Calculates the cost impact of the CanBikeCO E-bike program
    
     Parameters:
        data - CanBikeCO data input
        dist - feature name in df of feature with distance in miles
        repm - feature name in df of feature with replaced mode
        mode - feature name in df of feature with confirmed mode
        
    Returns:
        data with appended time impact feature for each trip in $$$ (float)
    """

    data[mode+'_dura'] = data[dist] * data['dura__trip_mode']
    data[repm+'_dura'] = data[dist] * data['dura__trip_repm']
    data['Cost_Impact($)'] = round((data[mode+'_dura'] - data[repm+'_dura']),3)

    return data


def calc_avg_speed(data, dist, time, mode, meth='average'):
    """
    Purpose:
        To determine average speed of modes in CanBikeCO data

    Parameters:
        data - CanBikeCO data input
        dist - feature name in df of feature with distance in miles
        time - feature name in df of feature with time information
        mode - feature name in df of feature with confirmed mode
        meth - string representing method for aggregation by group
                ['average', 'median']
    Process:
        Calculate and append speeds of each trip
        Aggregate average speed for each mode
        Save averages in auxiallary files
    
    Returns:
        data - data with speed feature for each trip
        df_T - a dataframe representing average speed by mode
    """
    
    data = data.copy()

    data['speed'] = data[dist] / data[time]

    grup = data.groupby(mode)

    mspd = None
    if(meth == 'average'):
        mspd = grup['speed'].mean()
    elif(meth == 'median'):
        mspd = grup['speed'].median()
    else:
        print(f'Method invalid: {meth}.')
        return data, None
    
    mspd.to_csv('auxiliary_files/time.csv')
    df_T = pd.read_csv('auxiliary_files/time.csv')

    return data, df_T