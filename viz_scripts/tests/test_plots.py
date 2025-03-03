import pytest as pytest
import pandas as pd
import numpy as np
# Using import_module, as we have saved-notebooks as the directory
import importlib
import pathlib
plots = importlib.import_module('saved-notebooks.plots')

# Test Data Fixtures
@pytest.fixture
def sample_labels():
    return ['Car', 'Bus', 'Train', 'Walk']

@pytest.fixture
def sample_values():
    return [100, 50, 3, 1]

@pytest.fixture
def sample_labels_no_small():
    return ['Car', 'Bus']


@pytest.fixture
def sample_values_no_small():
    return [100, 100]

class TestCalculatePct:
    def test_calculate_pct_basic(self, sample_labels, sample_values):
        labels, values, pcts = plots.calculate_pct(sample_labels, sample_values)
        assert len(labels) == len(sample_labels)
        assert len(values) == len(sample_values)
        assert sum(pcts) == pytest.approx(100.0, abs=0.1)

    def test_calculate_pct_empty(self):
        labels, values, pcts = plots.calculate_pct([],[])
        assert len(labels) == 0
        assert len(values) == 0
        assert len(pcts) == 0

    def test_calculate_pct_single(self):
        labels, values, pcts = plots.calculate_pct(['Car'], [100])
        assert pcts == [100.0]

class TestMergeSmallEntries:
    def test_merge_small_entries_basic(self, sample_labels, sample_values):
        labels, values, pcts = plots.merge_small_entries(sample_labels, sample_values)
        assert all(pct > 2.0 for pct in pcts)

    def test_merge_small_entries_no_small(self, sample_labels_no_small, sample_values_no_small):
        result_labels, result_values, result_pcts = plots.merge_small_entries(sample_labels_no_small, sample_values_no_small)
        assert len(result_labels) == 2
        assert 'other' not in result_labels
        assert 'OTHER' not in result_labels

    def test_merge_small_entries_some_small(self, sample_labels, sample_values):
        result_labels, result_values, result_pcts = plots.merge_small_entries(sample_labels, sample_values)
        print(result_labels)
        assert len(result_labels) == 3
        assert result_labels[0] in ['Car', 'Bus','other', 'OTHER']

@pytest.fixture
def sample_alt_text_error():
    return '''\\Error while generating chart:Traceback (most recent call last):
                        File "/tmp/ipykernel_320/1178426981.py", line 9, in <module>
                            plot_and_text_stacked_bar_chart(expanded_ctd, lambda df: (df.groupby("mode_confirm_w_other").agg({distance_col: 'count'}).sort_values(by=distance_col, ascending=False)),
                        NameError: name 'expanded_ctd' is not define'''

@pytest.fixture
def sample_alt_text():
    return '''Number of trips for each mode
                    Stacked Bar of: Labeled by user
                    9389 trips (46.87%)
                    from 24 users
                    Gas Car Shared Ride is 2233(23.8%).
                    E-bike is 2010(21.4%).
                    Walk is 1626(17.3%).
                    Gas Car Drove Alone is 1586(16.9%).
                    Other is 716(7.6%).
                    Hybrid Drove Alone is 455(4.8%).
                    Hybrid Shared Ride is 350(3.7%).
                    Taxi / Uber / Lyft is 117(1.2%).
                    Bus is 85(0.9%).
                    Regular Bike is 80(0.9%).
                    Not a trip is 59(0.6%).
                    E-Car Shared Ride is 34(0.4%).
                    E-Car Drove Alone is 11(0.1%).
                    Free Shuttle is 10(0.1%).
                    Train is 8(0.1%).
                    Air is 4(0.0%).
                    Scooter share is 3(0.0%).
                    Bikeshare is 2(0.0%).

                    Stacked Bar of: Inferred from prior labels
                    14087 trips (70.33%)
                    from 24 users
                    E-bike is 4428(31.4%).
                    Gas Car Shared Ride is 2980(21.2%).
                    Gas Car Drove Alone is 2453(17.4%).
                    Walk is 2142(15.2%).
                    Other is 726(5.2%).
                    Hybrid Drove Alone is 455(3.2%).
                    Hybrid Shared Ride is 351(2.5%).
                    Taxi / Uber / Lyft is 172(1.2%).
                    Bus is 117(0.8%).
                    Regular Bike is 114(0.8%).
                    Not a trip is 69(0.5%).
                    E-Car Shared Ride is 41(0.3%).
                    E-Car Drove Alone is 11(0.1%).
                    Free Shuttle is 11(0.1%).
                    Train is 8(0.1%).
                    Air is 4(0.0%).
                    Scooter share is 3(0.0%).
                    Bikeshare is 2(0.0%).

                    Stacked Bar of: Sensed by OpenPATH
                    20030 trips (100%)
                    from 27 users
                    IN_VEHICLE is 12269(61.3%).
                    WALKING is 3312(16.5%).
                    UNKNOWN is 2682(13.4%).
                    BICYCLING is 1735(8.7%).
                    AIR_OR_HSR is 32(0.2%).
            '''

class BaseAltTextTest:
    def setup_method(self):
        pathlib.Path(plots.SAVE_DIR).mkdir(exist_ok=True)

    def teardown_method(self):
        for file in pathlib.Path(plots.SAVE_DIR).glob('test_*.txt'):
            file.unlink()

class TestAccessAltText(BaseAltTextTest):

    def test_access_alt_text_creates_file(self, sample_alt_text):
        chart_name = "ntrips_total_default"
        result = plots.access_alt_text(sample_alt_text, chart_name)

        assert result == sample_alt_text
        file_path = pathlib.Path(plots.SAVE_DIR) / f"{chart_name}.txt"
        assert file_path.exists()

    def test_access_alt_text_writes_content(self, sample_alt_text):
        chart_name = "ntrips_total_default"
        plots.access_alt_text(sample_alt_text, chart_name)
        file_path = pathlib.Path(plots.SAVE_DIR) / f"{chart_name}.txt"
        content = file_path.read_text()
        assert content == sample_alt_text

    def test_access_alt_text_overwrites_existing_file(self, sample_alt_text, sample_alt_text_error):
        chart_name = "ntrips_total_default"
        plots.access_alt_text(sample_alt_text_error, chart_name)
        result =plots.access_alt_text(sample_alt_text, chart_name)
        assert result == sample_alt_text

        file_path = pathlib.Path(plots.SAVE_DIR) / f"{chart_name}.txt"
        content = file_path.read_text()
        assert content == sample_alt_text
