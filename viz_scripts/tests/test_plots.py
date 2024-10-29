import pytest as pytest
import pandas as pd
import numpy as np
import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import plots as plots

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
