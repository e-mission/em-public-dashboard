import unittest.mock as mock
import emission.core.wrapper.localdate as ecwl
import emission.storage.timeseries.tcquery as esttc
import importlib
import pandas as pd
import numpy as np
import collections as colls
import pytest

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