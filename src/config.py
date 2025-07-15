from pathlib import Path
from typing import List

# File paths
RESULTS_FILE_PATH = Path("results/fits/")
DATA_DIR = Path("data/with_event_properties")

# List of vague adverbials
VAGUE_ADVERBIALS: List[str] = ["recently", "just", "some time ago", "long time ago"]

# Duration and frequency order
DURATION_ORDER = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years', 'Decades']
FREQUENCY_ORDER = ['Daily', 'Monthly', 'Yearly', 'Decadal', 'Once in Life']
