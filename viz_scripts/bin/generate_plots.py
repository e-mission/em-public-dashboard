import nbclient
import nbformat
import nbparameterise as nbp
import argparse
import arrow
import requests
import json
import os
import sys


# Configuration settings to use for all generated plots by this instance
# This could also be specified as a parser argument, if we want to generate plots for all programs from one instance
# Full list is at
# https://github.com/e-mission/nrel-openpath-deploy-configs/tree/main/configs
STUDY_CONFIG = os.getenv('STUDY_CONFIG', "stage-program")

parser = argparse.ArgumentParser(prog="generate_metrics")
parser.add_argument("plot_notebook", help="the notebook the generates the plot")
parser.add_argument("program", help="the program for the plot")
parser.add_argument("-d", "--date", nargs=2, type=int,
    help="the year and month for the plot. Default: all previous days and months since program start + one combined for the program as a whole")

args = parser.parse_args()

# Read and use parameters from the unified config file on the e-mission Github page
download_url = "https://raw.githubusercontent.com/e-mission/nrel-openpath-deploy-configs/main/configs/" + STUDY_CONFIG + ".nrel-op.json"
print("About to download config from %s" % download_url)
r = requests.get(download_url)
if r.status_code is not 200:
    print(f"Unable to download study config, status code: {r.status_code}")
    sys.exit(1)
else:
    dynamic_config = json.loads(r.text)
    print(f"Successfully downloaded config with version {dynamic_config['version']} "\
        f"for {dynamic_config['intro']['translated_text']['en']['deployment_name']} ")

if dynamic_config['intro']['program_or_study'] == 'program':
    mode_studied = dynamic_config['intro']['mode_studied']
else:
    mode_studied = None

# Check if the dynamic config contains dynamic labels 'label_options'
# Passing the boolean flag is not enough, bcos we need to trace through the dynamic_config
if 'label_options' in dynamic_config:
    has_dynamic_labels = True
    dynamic_labels_url = dynamic_config['label_options']
else:
    has_dynamic_labels = False

if args.date is None:
    start_date = arrow.get(int(dynamic_config['intro']['start_year']),
        int(dynamic_config['intro']['start_month']), 1)
    end_date = arrow.get()
else:
    start_date = arrow.get()
    end_date = start_date

compute_range = list(arrow.Arrow.range('month', start_date, end_date))

print(f"Running at {arrow.get()} with args {args} for range {compute_range[0], compute_range[-1]}")

with open(args.plot_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Get a list of Parameter objects
orig_parameters = nbp.extract_parameters(nb)

# We will be recomputing values for multiple months
# So let's make a common function to invoke
def compute_for_date(month, year):
    params = nbp.parameter_values(
        orig_parameters,
        year=year,
        month=month,
        program=args.program,
        study_type=dynamic_config['intro']['program_or_study'],
        mode_of_interest=mode_studied,
        include_test_users=dynamic_config.get('metrics', {}).get('include_test_users', False),
        has_dynamic_labels = has_dynamic_labels,
        dynamic_labels_url = dynamic_labels_url)
    print(f"Running at {arrow.get()} with params {params}")

    # Make a notebook object with these definitions
    new_nb = nbp.replace_definitions(nb, params, execute=False)

    # Execute the notebook with the new parameters
    nbclient.execute(new_nb)

# Compute the overall metrics
compute_for_date(None, None)

# Compute for every month until now
for month_year in compute_range:
    compute_for_date(month_year.month, month_year.year)
