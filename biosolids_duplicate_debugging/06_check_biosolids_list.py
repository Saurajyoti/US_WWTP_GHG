# Check which NPDES permits are not for wastewater facilities

# Use the same filtering criteria as applied to all_wwtps:

# > 'sewer_system' if there is at least one sewer permit in the facility-associated SIC codes
# > 'other_system' if there is no sewer permit in any of the SIC codes

#%%
# Imports
import pandas as pd
import pathlib
from tqdm import tqdm
from baseline_utilities import check_for_ww_permits

tqdm.pandas()

# Load data
biosolids_data = pd.read_csv(pathlib.PurePath('01_raw_data', 'Data_Download_1699657092121.csv'))
biosolids_data['check_sewer_permits'] = biosolids_data['NPDES ID'].progress_apply(check_for_ww_permits)

# Save dataset that has all entries in biosolids dataset, with new column indicating if it has a sewer permit
biosolids_data.to_csv(pathlib.PurePath('04_results', 'biosolids_with_ww_permits.csv'))

# Filter by facilities that do not have
biosolids_not_sewer = biosolids_data[biosolids_data['check_sewer_permits'] == 'other_system'].copy()
biosolids_not_sewer.to_csv(pathlib.PurePath('04_results', 'bioslids_not_sewer.csv'))
print('Number of NPDES permits in the bioslids dataset without associated sewer facilities:')
print(f'{len(biosolids_not_sewer)}')

