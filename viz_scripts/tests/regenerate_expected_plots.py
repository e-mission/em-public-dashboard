# Test-only: regenerate the expected summary statistics and tests/data/expected_plots from the test database.
# Runs as the last step of docker-compose.recreate-test-exp.yml; to regenerate only the expected values from
# the existing dump, run it in the docker-compose.tests.yml stack from saved-notebooks/tests:
#   PYTHONPATH=/usr/src/app python regenerate_expected_plots.py
import json
import re
import shutil

import pymongo

import test_generate_plots as tgp

MONTHLY_OUTPUT = re.compile(r"_\d{4}_\d{2}_")


def main():
    url = tgp.get_test_db_url()
    with pymongo.MongoClient(url) as client:
        stats = tgp.compute_summary_statistics(client[tgp.TEST_DB])
    with open(tgp.EXPECTED_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {tgp.EXPECTED_STATS_FILE}")

    start = tgp.run_generate_plots(url)

    overall = sorted(p for p in tgp.PLOTS_DIR.iterdir()
                     if p.suffix == ".txt" and not MONTHLY_OUTPUT.search(p.name)
                     and p.stat().st_mtime >= start - 1)
    tgp.EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    for old in tgp.EXPECTED_DIR.iterdir():
        old.unlink()
    for p in overall:
        shutil.copy2(p, tgp.EXPECTED_DIR)
    print(f"Wrote {len(overall)} expected outputs to {tgp.EXPECTED_DIR}; review them with git diff")


if __name__ == "__main__":
    main()
