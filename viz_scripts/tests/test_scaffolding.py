import unittest.mock as mock
import emission.core.wrapper.localdate as ecwl
import emission.storage.timeseries.tcquery as esttc
import importlib
import pandas as pd
import numpy as np
import collections as colls
import pytest
import matplotlib.pyplot as plt

# Dynamically import saved-notebooks.plots
scaffolding = importlib.import_module('saved-notebooks.scaffolding')

def test_get_time_query():
    # Test with both year and month
    result = scaffolding.get_time_query(2022, 6)
    assert result is not None
    assert isinstance(result, esttc.TimeComponentQuery)

    # Test with year and no month
    result = scaffolding.get_time_query(2023, None)
    assert result is not None
    assert isinstance(result, esttc.TimeComponentQuery)

    # Test with month and no year
    with pytest.raises(Exception) as e_info:
        result = scaffolding.get_time_query(None, 12)

    # Test with no year or month
    result = scaffolding.get_time_query(None, None)
    assert result is None

def test_mapping_labels():
    dynamic_labels = {
        "MODE": [
            {"value":"gas_car", "base_mode": "CAR",
            "baseMode":"CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.22031},
            {"value":"motorcycle", "base_mode": "MOPED", "footprint": { "gasoline": { "wh_per_km": 473.17 }},
            "baseMode":"MOPED", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.113143309},
            {"value":"walk", "base_mode": "WALKING",
            "baseMode":"WALKING", "met_equivalent":"WALKING", "kgCo2PerKm": 0}
        ],
        "PURPOSE": [
            {"value":"home"},
            {"value":"shopping"},
            {"value":"meal"}
        ],
          "REPLACED_MODE": [
            {"value":"no_travel"},
            {"value":"bike"},
            {"value":"taxi"}
        ],
        "translations": {
            "en": {
            "walk": "Walk",
            "motorcycle":"Motorcycle",
            "bike": "Bicycle",
            "gas_car": "Car",
            "taxi": "Taxi",
            "no_travel": "No Travel",
            "home": "Home",
            "meal": "Meal",
            "shopping": "Shopping"
            }
        }
    }

    result_mode = scaffolding.mapping_labels(dynamic_labels, "MODE")
    result_purpose = scaffolding.mapping_labels(dynamic_labels, "PURPOSE")
    result_replaced = scaffolding.mapping_labels(dynamic_labels, "REPLACED_MODE")

    expected_result_mode = colls.defaultdict(lambda: 'Other', {
        "gas_car": "Car",
        "motorcycle": "Motorcycle",
        "walk": "Walk"
    })

    expected_result_purpose = colls.defaultdict(lambda: 'Other', {
        "home": "Home",
        "shopping": "Shopping",
        "meal": "Meal"
    })

    expected_result_replaced = colls.defaultdict(lambda: 'Other', {
        "no_travel": "No Travel",
        "bike": "Bicycle",
        "taxi": "Taxi"
    })
    assert result_mode == expected_result_mode
    assert result_purpose == expected_result_purpose
    assert result_replaced == expected_result_replaced

def test_mapping_color_surveys():
    dic_options = {
        'yes': 'Yes',
        'no': 'No',
        '1': 'Disagree (1)',
        '2': '2',
        '3': 'Neutral (3)',
        '4': '4',
        '5': 'Agree (5)',
        'unsure': 'Unsure'
    }

    result = scaffolding.mapping_color_surveys(dic_options)

    # Check that result is a dictionary
    assert isinstance(result, dict)

    # Check unique values have unique colors, with an Other
    unique_values = list(colls.OrderedDict.fromkeys(dic_options.values()))
    assert len(result) == len(unique_values) + 1

    # Check colors are from plt.cm.tab10
    for color in result.values():
        assert color in plt.cm.tab10.colors

    # Specific checks for this example
    assert result['Yes'] ==  plt.cm.tab10.colors[0]
    assert result['No'] ==  plt.cm.tab10.colors[1]
    assert result['Disagree (1)'] ==  plt.cm.tab10.colors[2]
    assert result['2'] == plt.cm.tab10.colors[3]
    assert result['Neutral (3)'] ==  plt.cm.tab10.colors[4]
    assert result['4'] == plt.cm.tab10.colors[5]
    assert result['Agree (5)'] ==  plt.cm.tab10.colors[6]
    assert result['Unsure'] == plt.cm.tab10.colors[7]
    assert result['Other'] ==  plt.cm.tab10.colors[8]

def test_mapping_color_surveys_empty():
    # Test with an empty dictionary
    with pytest.raises(Exception):
        mapping_color_surveys({})

@pytest.fixture
def before_df():
    return pd.DataFrame({
        "user_id":["user_1", "user_1", "user_1", "user_2", "user_2", "user_3", "user_4", "user_5"],
        "mode_confirm":["own_car", "own_car", "walk", "bus", "walk", "car", "motorcycle", "bike"],
        "Mode_confirm":["Other", "Other", "Walk", "Bus", "Walk", "Car", "Bike", "Bike"],
        "raw_trip":["trip_0", "trip_1", "trip_2", "trip_3", "trip_4", "trip_5", "trip_6", "trip_7"],
        "start_ts":[1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09],
        "duration": [1845.26, 1200.89, 1000.56, 564.54, 456.456, 156.45, 1564.456, 156.564]
    })

@pytest.fixture
def after_df():
    return pd.DataFrame({
        "user_id":["user_1", "user_1", "user_4", "user_5"],
        "mode_confirm":["own_car", "own_car",  "motorcycle", "bike"],
        "Mode_confirm":["Other", "Other",  "Bike", "Bike"],
        "raw_trip":["trip_0", "trip_1", "trip_6", "trip_7"],
        "start_ts":[1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09],
        "duration": [1845.26, 1200.89, 1564.456, 156.564]
    })

def test_get_quality_text(before_df, after_df):
    result = scaffolding.get_quality_text(before_df, after_df)
    assert result == "Based on 4 confirmed trips from 3 users\nof 8 total  trips from 5 users (50.00%)"

def test_get_quality_text_include_test_users(before_df, after_df):
    result = scaffolding.get_quality_text(before_df, after_df, include_test_users = True)
    assert result == "Based on 4 confirmed trips from 3 testers and participants\nof 8 total  trips from 5 users (50.00%)"

def test_get_quality_text_include_mode_of_interest(before_df, after_df):
    result = scaffolding.get_quality_text(before_df, after_df, mode_of_interest = "Motorcycle")
    assert result == "Based on 4 confirmed Motorcycle trips from 3 users\nof 8 total confirmed trips from 5 users (50.00%)"

@pytest.fixture
def sensed_df():
    return pd.DataFrame({
        "user_id":["user_1", "user_1", "user_1", "user_2", "user_2", "user_3", "user_4", "user_5"],
        "primary_mode":["IN_VEHICLE", "IN_VEHICLE", "IN_VEHICLE", "IN_VEHICLE", "IN_VEHICLE", "IN_VEHICLE", "IN_VEHICLE", "IN_VEHICLE"],
        "raw_trip":["trip_0", "trip_1", "trip_2", "trip_3", "trip_4", "trip_5", "trip_6", "trip_7"],
        "start_ts":[1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09],
        "duration": [1845.26, 1200.89, 1000.56, 564.54, 456.456, 156.45, 1564.456, 156.564]
    })

def test_get_quality_text_sensed(sensed_df):
    result = scaffolding.get_quality_text_sensed(sensed_df)
    assert result == "Based on 8 trips from 5 users"

def test_get_quality_text_sensed_include_test_users(sensed_df):
    result = scaffolding.get_quality_text_sensed(sensed_df, include_test_users=True)
    assert result == "Based on 8 trips from 5 testers and participants"

def test_get_quality_text_numerator(sensed_df):
    result = scaffolding.get_quality_text_sensed(sensed_df)
    assert result == "Based on 8 trips from 5 users"

def test_get_quality_text_numerator_include_test_users(sensed_df):
    result = scaffolding.get_quality_text_sensed(sensed_df, include_test_users=True)
    assert result == "Based on 8 trips from 5 testers and participants"

def test_get_file_suffix():
    year = 2024
    month = 12
    program = "default"
    result = scaffolding.get_file_suffix(year, month, program)
    assert result == "_2024_12_default"