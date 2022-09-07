import nbclient
import nbformat
import nbparameterise as nbp
import argparse
import arrow
import requests
import json


# Configuration settings to use for all generated plots by this instance
# This could also be specified as a parser argument, if we want to generate plots for all programs from one instance
STUDY_CONFIG = "stage-program"

parser = argparse.ArgumentParser(prog="generate_metrics")
parser.add_argument("plot_notebook", help="the notebook the generates the plot")
parser.add_argument("program", help="the program for the plot")
parser.add_argument("-d", "--date", nargs=2, type=int,
    help="the year and month for the plot. Default: yesterday's year and month.")

args = parser.parse_args()

if args.date is None:
    # TODO: Figure out some fuzziness around the edges of the month
    # e.g. when the task runs in UTC, will we still not recompute on the last day
    yesterday = arrow.get()
    args.date = [yesterday.year, yesterday.month]

# Read and use parameters from the unified config file on the e-mission Github page
r = requests.get("https://raw.githubusercontent.com/e-mission/nrel-openpath-deploy-configs/main/configs/" + STUDY_CONFIG + ".nrel-op.json")
if r.status_code is not 200:
    print(f"Unable to download study config, status code: {r.status_code}")
else:
    dynamic_config = json.loads(r.text)
    print(f"Successfully downloaded config with version {dynamic_config['version']} "\
        f"for {dynamic_config['intro']['translated_text']['en']['deployment_name']} "\
        f"and data collection URL {dynamic_config['server']['connectUrl']}")
if dynamic_config['intro']['program_or_study'] == 'program':
    mode_studied = dynamic_config['intro']['mode_studied']
else:
    mode_studied = None

print(f"Running at {arrow.get()} with args {args}")

with open(args.plot_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Get a list of Parameter objects
orig_parameters = nbp.extract_parameters(nb)

# Update one or more parameters
params = nbp.parameter_values(
    orig_parameters,
    year=args.date[0],
    month=args.date[1],
    program=args.program,
    study_type=dynamic_config['intro']['program_or_study'],
    mode_of_interest=mode_studied)

# Make a notebook object with these definitions
new_nb = nbp.replace_definitions(nb, params, execute=False)

# Execute the notebook with the new parameters
nbclient.execute(new_nb)
