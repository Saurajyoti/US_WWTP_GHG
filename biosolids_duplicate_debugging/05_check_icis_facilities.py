import pandas as pd
import pathlib
from baseline_utilities import check_for_dw_permits, check_for_ww_permits, flag_not_ww
from tqdm import tqdm

all_wwtp_sewer_codes = pd.read_csv(pathlib.PurePath('04_results', 'facilities_with_ww_permits.csv'))

# Check facilities that do not have WW permits associated with them
print('All datasest:')
print(all_wwtp_sewer_codes.head())

print(all_wwtp_sewer_codes['check_sewer_permits'].value_counts())

no_sewer_mask = (all_wwtp_sewer_codes['SOURCE'] == 'ICIS2022') & (all_wwtp_sewer_codes['check_sewer_permits'] == 'other_system' )
# Apply a filter based on source
all_wwtps_icis_no_sewer_code = all_wwtp_sewer_codes[no_sewer_mask]

print('Facilities in all_wwtps that do not have a sewer code, and are from ICIS2022')
print(len(all_wwtps_icis_no_sewer_code))
all_wwtps_icis_no_sewer_code.to_csv(pathlib.PurePath('04_results', 'all_wwtps_icis_no_sewer_code.csv'), index=False)

#%%


# # filter for datasets for values where SOURCE column is only ICIS
# print(all_wwtp['SOURCE'].unique())
# icis_wwtp = all_wwtp.loc[all_wwtp['SOURCE'] == 'ICIS2022'].copy()
#
# icis_wwtp['sic_code_type'] = icis_wwtp['NPDES_ID'].apply(check_for_dw_permits)
# icis_wwtp.to_csv(pathlib.PurePath('04_results', 'wwtp_icis_sic_code_assignment.csv'))
#
# icis_dw_plants = icis_wwtp.loc[icis_wwtp['sic_code_type'] == 'drinking_water'].copy()
# icis_dw_plants.to_csv(pathlib.PurePath('04_results', 'wwtp_icis_dw_code.csv'))
#
# #%%
#
# # NOTE - I've now moved this code to the explore_all_wwtps.ipynb dataset
# print('Run code on all_wwtps')
# # Flag rows for review that have an NPDES permit that is not wastewater
# all_wwtp['ww_sic_code'] = all_wwtp['NPDES_ID'].apply(flag_not_ww)
#
# all_wwtp_for_review = all_wwtp[all_wwtp['ww_sic_code'] == 'REVIEW'].copy()
# print(all_wwtp_for_review.shape)
# all_wwtp_for_review.to_csv(pathlib.PurePath('04_results', 'flag_all_wwtp.csv'))
