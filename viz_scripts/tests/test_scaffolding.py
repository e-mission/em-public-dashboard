import unittest.mock as mock
import emission.core.wrapper.localdate as ecwl
import emission.storage.timeseries.tcquery as esttc
import importlib
import pandas as pd
import numpy as np
import collections as colls

# Dynamically import saved-notebooks.plots
scaffolding = importlib.import_module('saved-notebooks.scaffolding')

@mock.patch('emission.core.wrapper.localdate.LocalDate')
@mock.patch('emission.storage.timeseries.tcquery.TimeComponentQuery')
def test_get_time_query(mock_tcquery, mock_localdate):
    mock_localdate.return_value = ecwl.LocalDate({"year": 2022, "month": 6})
    mock_tcquery.return_value = esttc.TimeComponentQuery("data.start_local_dt", mock_localdate.return_value, mock_localdate.return_value)

    result = scaffolding.get_time_query(2022, 6)

    assert result == mock_tcquery.return_value
    mock_localdate.assert_called_with({"year": 2022, "month": 6})
    mock_tcquery.assert_called_with("data.start_local_dt", mock_localdate.return_value, mock_localdate.return_value)

def test_mapping_labels():
    dynamic_labels = {
        "MODE": [
            {"value":"gas_car", "base_mode": "CAR",
            "baseMode":"CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.22031},
            {"value":"motorcycle", "base_mode": "MOPED", "footprint": { "gasoline": { "wh_per_km": 473.17 }},
            "baseMode":"MOPED", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.113143309},
            {"value":"walk", "base_mode": "WALKING",
            "baseMode":"WALKING", "met_equivalent":"WALKING", "kgCo2PerKm": 0},
            {"value":"e_car", "base_mode": "E_CAR",
            "baseMode":"E_CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.08216},
            {"value":"taxi", "base_mode": "TAXI",
            "baseMode":"TAXI", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.30741},
            {"value":"bike", "base_mode": "BICYCLING",
            "baseMode":"BICYCLING", "met_equivalent":"BICYCLING", "kgCo2PerKm": 0},
            {"value":"air", "base_mode": "AIR",
            "baseMode":"AIR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.09975}
        ],
        "translations": {
            "en": {
            "walk": "Walk",
            "motorcycle":"Motorcycle",
            "bike": "Bicycle",
            "gas_car": "Car",
            "e_car": "Electric Car",
            "taxi": "Taxi",
            "air": "Airplane"
            }
        }
    }

    result_mode = scaffolding.mapping_labels(dynamic_labels, "MODE")

    expected_result_mode = colls.defaultdict(lambda: 'Other', {
        "gas_car": "Car",
        "motorcycle": "Motorcycle",
        "walk": "Walk",
        "e_car": "Electric Car",
        "taxi": "Taxi",
        "bike": "Bicycle",
        "air": "Airplane"
    })
    assert result_mode == expected_result_mode