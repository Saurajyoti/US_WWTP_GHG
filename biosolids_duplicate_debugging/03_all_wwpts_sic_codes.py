# This code takes the all_wwtps dataset and identifies facilities that do NOT have an SIC wastewater permit code
# File is saved as all_wwtps_relevant_cols.csv

# This code adds a column to all_wwtps called check_sewer_permits, which can have the following entries:
# > 'no_permit' if NPDES == 0
# > 'sewer_system' if there is at least one sewer permit in the facility-associated SIC codes
# > 'other_system' if there is no sewer permit in any of the SIC codes

# Output is saved: 04_results > facilities_with_ww_permits.csv
# Code is used in 05_check_icis_facilities.py to identify non-sewer facilities sourced from ICIS2022

# Setup

import pandas as pd
import pathlib
from baseline_utilities import check_for_ww_permits
from tqdm import tqdm

tqdm.pandas()

# Data imports

facility_sic_data = pd.read_csv(pathlib.PurePath('02_clean_data', 'facility_sic_water.csv'))
all_wwtp = pd.read_csv(pathlib.PurePath('02_clean_data', 'all_wwtps_relevant_cols.csv'))

# For now, I'm starting by looking for a direct match with NPDES_ID. Christina says that sometimes you can only
# find a match for the last entries in the numerical sequence
all_npdes_matches = []
# test_wwtp = all_wwtp.head(10).copy()

# all_wwtp = all_wwtp.head(10)

all_wwtp['check_sewer_permits'] = all_wwtp['NPDES_ID'].progress_apply(check_for_ww_permits)
all_wwtp.to_csv(pathlib.PurePath('04_results', 'facilities_with_ww_permits.csv'))

# check if any facilities do not have category "sewer permits"

all_wwtp_non_sewer = all_wwtp[all_wwtp['check_sewer_permits'] == 'other_system']
print(f'Number of facilities with no sewer permit: {len(all_wwtp_non_sewer)}')