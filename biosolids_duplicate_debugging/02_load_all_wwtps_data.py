# Setup

import pandas as pd
import pathlib
from baseline_utilities import load_all_wwtps_data


all_wwtps = load_all_wwtps_data()
all_wwtps.to_csv(pathlib.PurePath('02_clean_data', 'all_wwtps_relevant_cols.csv'))

