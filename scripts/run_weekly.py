import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

sys.path.insert(0, str(ROOT / "src"))

from paper_scoring.dates import last_completed_week


def run(script, year, week, *extra):
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / script),
        str(year),
        str(week),
        *extra,
    ]

    print(f"\n{'='*60}")
    print(f"Running {script}")
    print(f"{'='*60}")

    subprocess.run(cmd, cwd=ROOT, check=True)


def main():

    year, week = last_completed_week()

    print(f"\nProcessing Week {year}-W{week:02d}\n")

    # Step 1
    run("fetch_week.py", year, week)

    # Step 2
    run("score_week.py", year, week, "--limit", "1")

    # Step 3
    run("publish_week.py", year, week, "--push")

    print("\nPipeline Finished Successfully.")


if __name__ == "__main__":
    main()