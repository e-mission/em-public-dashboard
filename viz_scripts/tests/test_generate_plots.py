import collections
import json
import os
import pathlib
import subprocess
import sys
import time

import pymongo
import pytest
from pymongo import uri_parser

TESTS_DIR = pathlib.Path(__file__).parent
NOTEBOOK_DIR = TESTS_DIR.parent
EXPECTED_DIR = TESTS_DIR / "data" / "expected_plots"
EXPECTED_STATS_FILE = TESTS_DIR / "data" / "expected_summary_statistics.json"
STUDY_CONFIG_FILE = TESTS_DIR / "data" / "configs" / "stage-program.nrel-op.json"
PLOTS_DIR = pathlib.Path("/plots")
# Restored from tests/data/openpath_prod_stage_program.tar.gz by the db-restore service in docker-compose.tests.yml
TEST_DB = "openpath_prod_stage_program"
# Survey notebooks only apply to ENKETO deployments; precompute_metrics refuses the overall (None, None) run
NOTEBOOKS = ["generic_metrics", "generic_metrics_sensed", "generic_timeseries",
             "mode_specific_metrics", "mode_specific_timeseries", "energy_calculations"]
# PNG bytes differ between runs, so only check that they are written
# .html outputs repeat the .txt results as tables, so only .txt is compared
EXPECTED_TEXT_FILES = sorted(p.name for p in EXPECTED_DIR.glob("*.txt"))
EXPECTED_PNGS = sorted(p.stem + ".png" for p in EXPECTED_DIR.glob("*.txt"))
# generate_plots.py runs in a subprocess, so the config download is mocked there
# Not mocked: emcommon may call the US Census geocoder for transit footprints from the notebook kernels;
# the current test data does not trigger it, but data changes could make this test network-dependent
RUN_WITH_LOCAL_CONFIG = """
import runpy, sys, types
from unittest import mock
import emission.core.deployment_config as ecdc
config_file, script = sys.argv[1], sys.argv[2]
with open(config_file) as f:
    response = types.SimpleNamespace(status_code=200, text=f.read())
sys.argv = sys.argv[2:]
with mock.patch.object(ecdc.requests, "get", return_value=response) as get:
    runpy.run_path(script, run_name="__main__")
get.assert_called_once_with(ecdc.CONFIGS_URL + "stage-program.nrel-op.json")
"""


def get_test_db_url():
    host = os.environ.get("DB_HOST", "localhost")
    if "://" in host:
        host = "%s:%s" % uri_parser.parse_uri(host)["nodelist"][0]
    return f"mongodb://{host}/{TEST_DB}"


def compute_summary_statistics(db):
    ats = db.Stage_analysis_timeseries
    composite_query = {"metadata.key": "analysis/composite_trip"}
    confirmed = list(ats.find({"metadata.key": "analysis/confirmed_trip"}, {"data.user_input": 1}))
    labels = [c["data"]["user_input"] for c in confirmed if c["data"].get("user_input")]
    return {
        "collection_counts": {c: db[c].count_documents({}) for c in sorted(db.list_collection_names())},
        "composite_trips": ats.count_documents(composite_query),
        "users_with_composite_trips": len(ats.distinct("user_id", composite_query)),
        "confirmed_trips": len(confirmed),
        "labeled_confirmed_trips": len(labels),
        "label_counts": {field: dict(sorted(collections.Counter(l[field] for l in labels if field in l).items()))
                         for field in ["mode_confirm", "purpose_confirm", "replaced_mode"]},
    }


def run_generate_plots(url):
    """Returns the start time, so callers can tell regenerated outputs from stale ones."""
    assert not (NOTEBOOK_DIR / "conf" / "storage" / "db.conf").exists(), \
        "conf/storage/db.conf overrides DB_HOST, so the notebooks would not read the test database"
    env = {**os.environ, "DB_HOST": url, "STUDY_CONFIG": "stage-program"}
    start = time.time()
    for nb in NOTEBOOKS:
        result = subprocess.run(
            [sys.executable, "-c", RUN_WITH_LOCAL_CONFIG, str(STUDY_CONFIG_FILE),
             "bin/generate_plots.py", f"{nb}.ipynb", "default", "-d", "2016", "8"],
            cwd=NOTEBOOK_DIR, env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"{nb} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return start


@pytest.fixture(scope="module")
def restored_db():
    with pymongo.MongoClient(get_test_db_url()) as client:
        yield client[TEST_DB]


@pytest.fixture(scope="module")
def generated_plots():
    return run_generate_plots(get_test_db_url())


def test_summary_statistics(restored_db):
    assert restored_db.Stage_pipeline_state.count_documents({"curr_run_ts": {"$ne": None}}) == 0
    with open(EXPECTED_STATS_FILE) as f:
        assert compute_summary_statistics(restored_db) == json.load(f)


@pytest.mark.parametrize("name", EXPECTED_TEXT_FILES)
def test_text_output_matches(generated_plots, name):
    actual = PLOTS_DIR / name
    assert actual.stat().st_mtime >= generated_plots - 1, f"{name} was not regenerated"
    assert actual.read_text() == (EXPECTED_DIR / name).read_text()


@pytest.mark.parametrize("name", EXPECTED_PNGS)
def test_png_generated(generated_plots, name):
    assert (PLOTS_DIR / name).stat().st_mtime >= generated_plots - 1, f"{name} was not regenerated"
