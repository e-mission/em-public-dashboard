import unittest.mock as mock
import emission.core.wrapper.localdate as ecwl
import emission.storage.timeseries.tcquery as esttc
import importlib
import pandas as pd
import numpy as np
import collections as colls
import pytest
import asyncio
import matplotlib.pyplot as plt
import emcommon.util as emcu

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

@pytest.fixture
def dynamic_labels():
    return {
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
            },
            "es": {
                "walk": "Caminando",
                "motorcycle":"Motocicleta",
                "bike":"Bicicleta",
                "gas_car":"Coche de gasolina",
                "taxi":"Taxi",
                "no_travel":"No viajar",
                "home":"Casa",
                "meal":"Comida",
                "shopping":"Compras"
            }
        }
    }

def test_mapping_labels(dynamic_labels):
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
        "duration": [1845.26, 1200.89, 1000.56, 564.54, 456.456, 156.45, 1564.456, 156.564],
        "distance": [100, 150, 600, 500, 300, 200, 50, 20]
    })

@pytest.fixture
def after_df():
    return pd.DataFrame({
        "user_id":["user_1", "user_1", "user_4", "user_5"],
        "mode_confirm":["own_car", "own_car",  "motorcycle", "bike"],
        "Mode_confirm":["Other", "Other",  "Bike", "Bike"],
        "raw_trip":["trip_0", "trip_1", "trip_6", "trip_7"],
        "start_ts":[1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09],
        "duration": [1845.26, 1200.89, 1564.456, 156.564],
        "distance": [100, 150, 50, 20]
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

def test_unit_conversions(before_df):
    test_df = before_df.copy()
    scaffolding.unit_conversions(test_df)
    assert 'distance_miles' in test_df.columns
    assert 'distance_kms' in test_df.columns

    np.testing.assert_almost_equal(
        test_df['distance_miles'],
        [0.062, 0.093, 0.373, 0.311, 0.186, 0.124, 0.031, 0.012],
        decimal=2
    )

    np.testing.assert_almost_equal(
        test_df['distance_kms'],
        [0.1, 0.15, 0.6, 0.5, 0.3, 0.2, 0.05, 0.02],
        decimal=2
    )

def test_filter_labeled_trips_with_labeled_trips():
    mixed_trip_df = pd.DataFrame({
        'user_input':[
            {'purpose_confirm': 'work', 'mode_confirm':'own_car'},
            {'mode_confirm':'bus'},
            {'purpose_confirm': 'school'},
            {},
            {'purpose_confirm': 'shopping', 'mode_confirm':'car'},
            {},
            {}
        ],
        "distance": [100, 150, 50, 20, 50, 10, 60]
    })

    labeled_ct = scaffolding.filter_labeled_trips(mixed_trip_df)

    # Assert the length of the dataframe, which does not have user_input
    assert len(labeled_ct) == 4
    assert all(labeled_ct['user_input'].apply(bool))

def test_filter_labeled_trips_empty_dataframe():
    # Create an empty DataFrame
    mixed_trip_df = pd.DataFrame(columns=['user_input'])

    labeled_ct = scaffolding.filter_labeled_trips(mixed_trip_df)

    # Assert the returned DataFrame is empty
    assert len(labeled_ct) == 0

def test_filter_labeled_trips_no_labeled_trips():
    # Create a DataFrame with only unlabeled trips
    mixed_trip_df = pd.DataFrame({
        'user_input': [{}, {}, {}],
        "distance": [100, 150, 50]
    })

    labeled_ct = scaffolding.filter_labeled_trips(mixed_trip_df)

    # Assert the returned DataFrame is empty
    assert len(labeled_ct) == 0

@pytest.fixture
def labeled_ct():
    return pd.DataFrame({
        'user_input':[
            {'purpose_confirm': 'work', 'mode_confirm':'own_car'},
            {'mode_confirm':'bus'},
            {'purpose_confirm': 'school'},
            {'purpose_confirm': 'at_work', 'mode_confirm': 'own_car'},
            {'purpose_confirm': 'access_recreation', 'mode_confirm':'car'},
            {'mode_confirm':'bike', 'purpose_confirm':'pick_drop_person'},
            {'purpose_confirm':'work', 'mode_confirm':'bike'}
        ],
        "distance": [100, 150, 50, 20, 50, 10, 60],
        "user_id":["user_1", "user_1", "user_1", "user_2", "user_2", "user_3", "user_4"],
        "raw_trip":["trip_0", "trip_1", "trip_2", "trip_3", "trip_4", "trip_5", "trip_6"],
        "start_ts":[1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09],
        "duration": [1845.26, 1200.89, 1000.56, 564.54, 456.456, 156.45, 1564.456],
        "distance": [100, 150, 600, 500, 300, 200, 50]
    })

def test_expand_userinputs(labeled_ct):
    expanded_ct = scaffolding.expand_userinputs(labeled_ct)

    # Assert the length of the dataframe is not changed
    # Assert the columns have increased with labels_per_trip
    labels_per_trip = len(pd.DataFrame(labeled_ct.user_input.to_list()).columns)

    assert len(expanded_ct) == len(labeled_ct)
    assert labels_per_trip == 2
    assert len(expanded_ct.columns) == len(labeled_ct.columns) + labels_per_trip

    # Assert new columns and their values
    assert 'mode_confirm' in expanded_ct.columns
    assert 'purpose_confirm' in expanded_ct.columns
    assert expanded_ct['purpose_confirm'].fillna('NaN').tolist() == ['work', 'NaN', 'school', 'at_work', 'access_recreation', 'pick_drop_person', 'work']
    assert expanded_ct['mode_confirm'].fillna('NaN').tolist() == ['own_car', 'bus', 'NaN', 'own_car', 'car', 'bike', 'bike']

# Testing with just dynamic_labels since PR#164 Unify calls to read json resource from e-mission-common in generate_plots.py would make sure we have labels passed into this file, instead of fetching the label-options.json file here
def test_translate_values_to_labels_english(dynamic_labels):
    # Call the function with our predefined labels
    mode_translations, purpose_translations, replaced_translations = scaffolding.translate_values_to_labels(dynamic_labels)

    expected_mode_translations = colls.defaultdict(lambda: 'Other', {
        "gas_car": "Car",
        "motorcycle": "Motorcycle",
        "walk": "Walk",
        "other": "Other"
    })

    expected_purpose_translations = colls.defaultdict(lambda: 'Other', {
        "home": "Home",
        "shopping": "Shopping",
        "meal": "Meal",
        "other": "Other"
    })

    expected_replaced_translations = colls.defaultdict(lambda: 'Other', {
        "no_travel": "No Travel",
        "bike": "Bicycle",
        "taxi": "Taxi",
        "other": "Other"
    })
    assert mode_translations == expected_mode_translations
    assert purpose_translations == expected_purpose_translations
    assert replaced_translations == expected_replaced_translations

# TODO:: Implement language specific changes in mapping_translations
@pytest.mark.skip(reason="Implementation limited only for english translations")
def test_translate_values_to_labels_spanish(dynamic_labels, language="es"):
    # Call the function with our predefined labels
    mode_translations_es, purpose_translations_es, replaced_translations_es = scaffolding.translate_values_to_labels(dynamic_labels)

    expected_mode_translations_es = colls.defaultdict(lambda: 'Other', {
        "gas_car":"Coche de gasolina",
        "motorcycle":"Motocicleta",
        "walk": "Caminando"
    })

    expected_purpose_translations_es = colls.defaultdict(lambda: 'Other', {
        "home":"Casa",
        "shopping":"Compras",
        "meal":"Comida"
    })

    expected_result_replaced_translations_es = colls.defaultdict(lambda: 'Other', {
        "no_travel":"No viajar",
        "bike":"Bicicleta",
        "taxi": "Taxi"
    })
    assert mode_translations_es == expected_mode_translations_es
    assert purpose_translations_es == expected_purpose_translations_es
    assert replaced_translations_es == expected_result_replaced_translations_es

def test_translate_values_to_labels_empty_input():
    # Test with empty input
    with pytest.raises(AttributeError):
        mode_translations, purpose_translations, replaced_translations = scaffolding.translate_values_to_labels([])

@pytest.fixture
def inferred_ct():
    return pd.DataFrame({
        'user_input': [
            {},
            {},
            {},
            {},
            {},
            {'mode_confirm':'bike', 'purpose_confirm':'pick_drop_person'},
            {'purpose_confirm':'work', 'mode_confirm':'bike'}
        ],
        "distance": [100, 150, 50, 20, 50, 10, 60],
        "user_id": ["user_1", "user_1", "user_1", "user_2", "user_2", "user_3", "user_4"],
        "raw_trip": ["trip_0", "trip_1", "trip_2", "trip_3", "trip_4", "trip_5", "trip_6"],
        "start_ts": [1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09, 1.690e+09],
        "duration": [1845.26, 1200.89, 1000.56, 564.54, 456.456, 156.45, 1564.456],
        "inferred_trip": ["itrip_0", "itrip_1", "itrip_2", "itrip_3", "itrip_4", "itrip_5", "itrip_6"],
        "confidence_threshold": [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
        "inferred_labels": pd.Series([
            [{'labels': {'mode_confirm': 'shared_ride', 'purpose_confirm': 'school'}, 'p': 0.99}],
            [{'labels': {'mode_confirm': 'motorcycle', 'purpose_confirm': 'at_work'}, 'p': 0.9899999407250384}],
            [{'labels': {'mode_confirm': 'motorcycle', 'purpose_confirm': 'work'}, 'p': 0.9899999998030119}],
            [{'labels': {'mode_confirm': 'drove_alone', 'purpose_confirm': ''}, 'p': 0.99}],
            [{'labels': {'mode_confirm': 'drove_alone', 'purpose_confirm': 'work'}, 'p': 0.99}],
            [],
            []
        ])
    })

def test_expand_inferredlabels(inferred_ct):
    expanded_ct = scaffolding.expand_inferredlabels(inferred_ct)

    # Call the function to expand inferred labels
    assert len(expanded_ct) == len(inferred_ct)

    # Assert new columns have been added
    assert 'mode_confirm' in expanded_ct.columns
    assert 'purpose_confirm' in expanded_ct.columns

    # Check the values of the new columns
    expected_mode_confirms = [
        'shared_ride',
        'motorcycle',
        'motorcycle',
        'drove_alone',
        'drove_alone',
        'bike',
        'bike'
    ]

    expected_purpose_confirms = [
        'school',
        'at_work',
        'work',
        '',
        'work',
        'pick_drop_person',
        'work'
    ]

    # Compare the extracted mode and purpose confirm
    assert expanded_ct['mode_confirm'].tolist() == expected_mode_confirms
    assert expanded_ct['purpose_confirm'].tolist() == expected_purpose_confirms

def test_filter_inferredlabels():
    mixed_trip_df = pd.DataFrame({
        "user_input": [
            {},
            {},
            {},
            {},
            {},
            {'mode_confirm':'bike', 'purpose_confirm':'pick_drop_person'},
            {'purpose_confirm':'work', 'mode_confirm':'bike'},
            {},
            {}
        ],
        "inferred_labels": pd.Series([
            [{'labels': {'mode_confirm': 'shared_ride', 'purpose_confirm': 'school'}, 'p': 0.99}],
            [{'labels': {'mode_confirm': 'motorcycle', 'purpose_confirm': 'at_work'}, 'p': 0.9899999407250384}],
            [{'labels': {'mode_confirm': 'motorcycle', 'purpose_confirm': 'work'}, 'p': 0.9899999998030119}],
            [{'labels': {'mode_confirm': 'drove_alone', 'purpose_confirm': ''}, 'p': 0.99}],
            [{'labels': {'mode_confirm': 'drove_alone', 'purpose_confirm': 'work'}, 'p': 0.99}],
            [],
            [],
            [],
            []
        ])
    })

    inferred_ct = scaffolding.filter_inferred_trips(mixed_trip_df)
    assert len(inferred_ct) == 7

def test_filter_inferredlabels_emptydataframe():
    mixed_trip_df = pd.DataFrame({
        "user_input": [
            {},
            {}
        ],
        "inferred_labels": pd.Series([
            [],
            []
        ])
    })

    inferred_ct = scaffolding.filter_inferred_trips(mixed_trip_df)
    assert len(inferred_ct) == 0