# One-off: load each server real_examples dataset as a separate user and run the intake pipeline.
# Run via docker-compose.recreate-test-exp.yml, which also dumps the result and regenerates the expected values.
import json
import logging
import os
import types

import numpy as np

import emission.core.get_database as edb
import emission.storage.json_wrappers as esj
import emission.tests.common as etc

DATA_DIR = "emission/tests/data/real_examples"
INPUT_SUFFIXES = [".user_inputs", ".mode_confirm", ".purpose_confirm"]
NON_DATA_MARKERS = [".ground_truth", ".expected_"] + INPUT_SUFFIXES
# stage-program uses MULTILABEL, so survey-style inputs would never appear in its data
SURVEY_INPUT_KEYS = {"manual/trip_user_input", "manual/trip_addition_input", "manual/place_addition_input"}
# The real_examples labels have no e-bike, replaced mode or work purpose; stage-program studies e-bike
# Label changes that need transit footprints may make emcommon call the US Census geocoder,
# which test_generate_plots.py does not mock
REPLACED_MODE_CYCLE = ["drove_alone", "bus", "walk", "no_travel"]


def load_entries(path):
    with open(path) as fp:
        return json.load(fp, object_hook=esj.wrapped_object_hook)


def save_for_user(user_id, entries):
    tsdb = edb.get_timeseries_db()
    for entry in entries:
        # Several files share entries; drop _id so users don't overwrite each other
        entry.pop("_id", None)
        entry["user_id"] = user_id
        edb.save(tsdb, entry)


def add_program_labels(user_id):
    tsdb = edb.get_timeseries_db()
    tsdb.update_many({"user_id": user_id, "metadata.key": "manual/mode_confirm", "data.label": "bike"},
                     {"$set": {"data.label": "e-bike"}})
    tsdb.update_many({"user_id": user_id, "metadata.key": "manual/purpose_confirm", "data.label": "library"},
                     {"$set": {"data.label": "work"}})
    mode_confirms = tsdb.find({"user_id": user_id, "metadata.key": "manual/mode_confirm"},
                              {"_id": 0}).sort("data.start_ts", 1)
    for i, entry in enumerate(mode_confirms):
        entry["metadata"]["key"] = "manual/replaced_mode"
        entry["data"]["label"] = REPLACED_MODE_CYCLE[i % len(REPLACED_MODE_CYCLE)]
        tsdb.insert_one(entry)


def main():
    etc.configLogging()
    logging.getLogger().setLevel(logging.INFO)
    data_files = sorted(f for f in os.listdir(DATA_DIR)
                        if not any(m in f for m in NON_DATA_MARKERS))
    for name in data_files:
        user = types.SimpleNamespace(testEmail=name)
        etc.createAndFillUUID(user)
        save_for_user(user.testUUID, load_entries(os.path.join(DATA_DIR, name)))
        for suffix in INPUT_SUFFIXES:
            input_path = os.path.join(DATA_DIR, name + suffix)
            if os.path.exists(input_path):
                save_for_user(user.testUUID, [e for e in load_entries(input_path)
                                              if e["metadata"]["key"] not in SURVEY_INPUT_KEYS])
        add_program_labels(user.testUUID)
        print(f"Loaded {name} as {user.testUUID}, running pipeline", flush=True)
        # location_smoothing fills gaps with random points; same seed as TestPipelineRealData
        np.random.seed(61297777)
        etc.runIntakePipeline(user.testUUID)


if __name__ == "__main__":
    main()
