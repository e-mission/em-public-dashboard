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
        f"for {dynamic_config['intro']['translated_text']['en']['deployment_name']} "\
        f"and data collection URL {dynamic_config['server']['connectUrl'] if 'server' in dynamic_config else 'default'}")

if dynamic_config['intro']['program_or_study'] == 'program':
    mode_studied = dynamic_config['intro']['mode_studied']
else:
    mode_studied = None

# Default value of dynamic_labels referenced from 
# https://github.com/e-mission/nrel-openpath-deploy-configs/blob/main/label_options/example-study-label-options.json
dynamic_labels = {
  "MODE": [
    {"value":"walk", "baseMode":"WALKING", "met_equivalent":"WALKING", "kgCo2PerKm": 0},
    {"value":"e-bike", "baseMode":"E_BIKE", "met": {"ALL": {"range": [0, -1], "mets": 4.9}}, "kgCo2PerKm": 0.00728},
    {"value":"bike", "baseMode":"BICYCLING", "met_equivalent":"BICYCLING", "kgCo2PerKm": 0},
    {"value":"bikeshare", "baseMode":"BICYCLING", "met_equivalent":"BICYCLING", "kgCo2PerKm": 0},
    {"value":"scootershare", "baseMode":"E_SCOOTER", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.00894},
    {"value":"drove_alone", "baseMode":"CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.22031},
    {"value":"shared_ride", "baseMode":"CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.11015},
    {"value":"e_car_drove_alone", "baseMode":"E_CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.08216},
    {"value":"e_car_shared_ride", "baseMode":"E_CAR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.04108},
    {"value":"moped", "baseMode":"MOPED", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.05555},
    {"value":"taxi", "baseMode":"TAXI", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.30741},
    {"value":"bus", "baseMode":"BUS", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.20727},
    {"value":"train", "baseMode":"TRAIN", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.12256},
    {"value":"free_shuttle", "baseMode":"BUS", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.20727},
    {"value":"air", "baseMode":"AIR", "met_equivalent":"IN_VEHICLE", "kgCo2PerKm": 0.09975},
    {"value":"not_a_trip", "baseMode":"UNKNOWN", "met_equivalent":"UNKNOWN", "kgCo2PerKm": 0},
    {"value":"other", "baseMode":"OTHER", "met_equivalent":"UNKNOWN", "kgCo2PerKm": 0}
  ],
  "PURPOSE": [
    {"value":"home"},
    {"value":"work"},
    {"value":"at_work"},
    {"value":"school"},
    {"value":"transit_transfer"},
    {"value":"shopping"},
    {"value":"meal"},
    {"value":"pick_drop_person"},
    {"value":"pick_drop_item"},
    {"value":"personal_med"},
    {"value":"access_recreation"},
    {"value":"exercise"},
    {"value":"entertainment"},
    {"value":"religious"},
    {"value":"other"}
  ],
  "translations": {
    "en": {
      "walk": "Walk",
      "e-bike": "E-bike",
      "bike": "Regular Bike",
      "bikeshare": "Bikeshare",
      "scootershare": "Scooter share",
      "drove_alone": "Gas Car Drove Alone",
      "shared_ride": "Gas Car Shared Ride",
      "e_car_drove_alone": "E-Car Drove Alone",
      "e_car_shared_ride": "E-Car Shared Ride",
      "moped": "Moped",
      "taxi": "Taxi/Uber/Lyft",
      "bus": "Bus",
      "train": "Train",
      "free_shuttle": "Free Shuttle",
      "air": "Air",
      "not_a_trip": "Not a trip",
      "home": "Home",
      "work": "To Work",
      "at_work": "At Work",
      "school": "School",
      "transit_transfer": "Transit transfer",
      "shopping": "Shopping",
      "meal": "Meal",
      "pick_drop_person": "Pick-up/ Drop off Person",
      "pick_drop_item": "Pick-up/ Drop off Item",
      "personal_med": "Personal/ Medical",
      "access_recreation": "Access Recreation",
      "exercise": "Recreation/ Exercise",
      "entertainment": "Entertainment/ Social",
      "religious": "Religious",
      "other": "Other"
    },
    "es": {
      "walk": "Caminando",
      "e-bike": "e-bicicleta",
      "bike": "Bicicleta",
      "bikeshare": "Bicicleta compartida",
      "scootershare": "Motoneta compartida",
      "drove_alone": "Coche de Gas, Condujo solo",
      "shared_ride": "Coche de Gas, Condujo con otros",
      "e_car_drove_alone": "e-coche, Condujo solo",
      "e_car_shared_ride": "e-coche, Condujo con ontras",
      "moped": "Ciclomotor",
      "taxi": "Taxi/Uber/Lyft",
      "bus": "Autobús",
      "train": "Tren",
      "free_shuttle": "Colectivo gratuito",
      "air": "Avión",
      "not_a_trip": "No es un viaje",
      "home": "Inicio",
      "work": "Trabajo",
      "at_work": "En el trabajo",
      "school": "Escuela",
      "transit_transfer": "Transbordo",
      "shopping": "Compras",
      "meal": "Comida",
      "pick_drop_person": "Recoger/ Entregar Individuo",
      "pick_drop_item": "Recoger/ Entregar Objeto",
      "personal_med": "Personal/ Médico",
      "access_recreation": "Acceder a Recreación",
      "exercise": "Recreación/ Ejercicio",
      "entertainment": "Entretenimiento/ Social",
      "religious": "Religioso",
      "other": "Otros"
    }
  }
}

# Check if the dynamic config contains dynamic labels 'label_options'
# Parse through the dynamic_labels_url:
if 'label_options' in dynamic_config:
    dynamic_labels_url = dynamic_config['label_options']

    if  dynamic_labels_url:
        req = requests.get(dynamic_labels_url)
        if req.status_code != 200:
            print(f"Unable to download dynamic_labels, status code: {req.status_code}")
        else:
            print("Dynamic labels download was successful.")
            dynamic_labels = json.loads(req.text)
    else:
        print("Dynamic labels URL is unavailable.")
else:
    print("Dynamic labels are not available.")

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
        dynamic_labels = dynamic_labels,
        sensed_algo_prefix=dynamic_config.get('metrics', {}).get('sensed_algo_prefix', "cleaned"))
        

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
