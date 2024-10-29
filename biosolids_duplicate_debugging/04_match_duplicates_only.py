# Setup

import pandas as pd
import pathlib
from baseline_utilities import check_for_dw_permits

# Load Christina's duplicate data
duplicate_facilities = pd.read_excel(pathlib.PurePath('03_christina_data', 'biosolids_dups_with_info_source_sic_code_notes.xlsx'))

duplicate_facilities['sic_code_type'] = duplicate_facilities['NPDES ID - Biosolids'].apply(check_for_dw_permits)
duplicate_facilities.to_csv(pathlib.PurePath('04_results', 'biosolids_duplicates_sic_code_assignment.csv'))