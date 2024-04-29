
# Efforts towards predicting the replaced mode without user labels

## Prerequisites:
- These experiments were conducted on top of the `emission` anaconda environment. Please ensure that this environment is available to you before re-running the code.
- In addition, some notebooks use `seaborn` for plotting and `pandarallel` for parallel pandas processing. The packages can be installed in the following manner:

```
(After activating emission conda env)
pip3 install pandarallel==1.6.5
pip3 install seaborn==0.12.2
```

- Ensure you have the following data sources loaded in your MongoDB Docker container:
	- Stage_database (All CEO)
	- Durham
	- Masscec
	- Ride2own
	- UPRM NICR
- Once these data sources are procured and loaded in your Mongo container, you will need to add the inferred sections to the data. To do this, please run the [add_sections_and_summaries_to_trips.py](https://github.com/e-mission/e-mission-server/blob/master/bin/historical/migrations/add_sections_and_summaries_to_trips.py) script. **NOTE**: If you see a lot of errors in the log, try to re-run the script by modifying the following line from:

```language=python
# Before
eps.dispatch(split_lists, skip_if_no_new_data=False, target_fn=add_sections_to_trips)

# After
eps.dispatch(split_lists, skip_if_no_new_data=False, target_fn=None)
```

This will trigger the intake pipeline for the current db and add the inferred section.

- Note 2: The script above did not work for the All CEO data for me. Therefore, I obtained the section durations using the `get_section_durations` method I've written in `scaffolding.py` (you do not have to call this method, it is already handled in the notebooks). Please note that running this script takes a long time and it is advised to cache the generated output.

## Running the experiments
The order in which the experiments are to be run are denoted by the preceding number. The following is a brief summary about each notebook:
1. `01_extract_db_data.ipynb`: This notebook extracts the data, performs the necessary preprocessing, updates availability indicators, computes cost estimates, and stores the preprocessed data in `data/filtered_trips`.
2. `02_run_trip_level_models.py`: This script reads all the preprocessed data, fits trip-level models with different stratitifications, generates the outputs, and stores them in `outputs/benchmark_results/`.
3. `03_user_level_models.ipynb`: This notebook explores user fingerprints, similarity searching, and naive user-level models.
4. `04_FeatureClustering.ipynb`: This notebook performs two functions: (a) Cluster users based on demographics/trip feature summaries and check for target distributions across clusters, and (b) Cluster users by grouping w.r.t. the target and checking for feature homogeneity within clusters
