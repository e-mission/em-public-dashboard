import nbclient
import nbformat
import nbparameterise as nbp
import argparse
import arrow

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

print(f"Running at {arrow.get()} with args {args}")

with open(args.plot_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Get a list of Parameter objects
orig_parameters = nbp.extract_parameters(nb)

# Update one or more parameters
params = nbp.parameter_values(orig_parameters,
    year=args.date[0], month=args.date[1], program=args.program)

# Make a notebook object with these definitions
new_nb = nbp.replace_definitions(nb, params, execute=False)

# Execute the notebook with the new parameters
nbclient.execute(new_nb)
