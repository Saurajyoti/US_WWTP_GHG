# Setup
import pandas as pd
import numpy as np
import pathlib


def load_all_wwtps_data():
    # I can add any data cleaning steps later
    relevant_wwtps_cols = ['FACILITY_CODE', 'SOURCE', 'FACILITY', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE',
                           'FLOW_2012_MGD',
                           '2012_TOT_ANNUAL_MM3', 'CWNS_NUM', 'NPDES_ID']
    all_wwtps = pd.read_csv(pathlib.PurePath('01_raw_data', 'all_wwtps_data.csv'), usecols=relevant_wwtps_cols,
                            low_memory=False)

    # Remove any rows that have 0 as NPDES_ID
    # all_wwtps = all_wwtps[all_wwtps['NPDES_ID'] != '0'] #TODO figure out why I did this

    return all_wwtps


#
def lookup_water_sic_code(npdes_id):
    """Look up the SIC codes associated with facilities that have the NPDES ID of interest.
    Input: string npdes_id to look up any associated facilities that have this permit;
    Output: the SIC codes associated with the facilities that have the NPDES ID.
    """

    # Dataframe for storing
    all_npdes_matches = []

    # Load SIC water / wastewater database
    facility_sic_data = pd.read_csv(pathlib.PurePath('02_clean_data', 'facility_sic_water.csv'))

    # Create a mask for the NPDES of interest
    npdes_mask = facility_sic_data['PGM_SYS_ACRNMS'].str.contains(npdes_id)

    # Filter the matches using the mask
    matched_npdes = facility_sic_data[npdes_mask]

    return matched_npdes['SIC_CODE'].tolist()


def check_all_sic_code(npdes_id):
    """Look up the SIC codes associated with facilities that have the NPDES ID of interest.
    Input: string npdes_id to look up any associated facilities that have this permit;
    Output: the SIC codes associated with the facilities that have the NPDES ID.
    """

    # Load SIC water / wastewater database
    # print('Loading facility data')

    relevant_facility_cols = ['REGISTRY_ID', 'PRIMARY_NAME', 'SIC_CODE', 'PGM_SYS_ACRNMS', 'LATITUDE83', 'LONGITUDE83',
                              'CITY_NAME',
                              'PRIMARY_NAME', 'LOCATION_ADDRESS']

    facility_sic_data = pd.read_csv(pathlib.PurePath('02_clean_data', 'facility_sic_all.csv'), low_memory=False,
                                    usecols=relevant_facility_cols)

    # print('Facility data loaded')

    # Drop NAN
    facility_sic_data.dropna(subset=['PGM_SYS_ACRNMS'], inplace=True)

    # print('Removed nan entries in permit column')

    # Create a mask for the NPDES of interest
    npdes_mask = facility_sic_data['PGM_SYS_ACRNMS'].str.contains(npdes_id)

    # Filter the matches using the mask
    matched_npdes = facility_sic_data[npdes_mask]

    sic_codes = matched_npdes['SIC_CODE'].tolist()

    if not sic_codes:
        return ['NO_SIC_MATCH']
    else:
        return sic_codes


def check_sic_facility_type(sic_code):
    """Look up if an SIC code is for drinking water treatment or wastewater treatment """

    # Water supply = 4941
    # Sewerage Systems = 4952
    # Refuse System s= 4953
    # Sanitary Services, Not Classified Elsewhere = 4959

    sic_dictionary = {
        4941: 'drinking_water',
        4952: 'sewer_system',
        4953: 'sewer_system',
        4959: 'sewer_system',
        '4941': 'drinking_water',
        '4952': 'sewer_system',
        '4953': 'sewer_system',
        '4959': 'sewer_system',
    }

    if sic_code in sic_dictionary:
        return sic_dictionary[sic_code]
    else:
        return 'other_system'


def check_for_dw_permits(npdes_id):
    """Check to see if any facilities associated with a given NPDES ID have a permit with a drinking water SIC code"""
    sic_list = lookup_water_sic_code(npdes_id)
    facility_types = []

    # Add each relevant facility type
    for sic_code in sic_list:
        facility_types.append(check_sic_facility_type(sic_code))

    if 'drinking_water' in facility_types:
        return 'drinking_water'
    else:
        return 'no_dw_permit'


def check_for_ww_permits(npdes_id):
    """Check to see if any facilities associated with a given NPDES ID have a permit with a drinking water SIC code"""

    if npdes_id == '0':
        return 'no_permit'

    sic_list = lookup_water_sic_code(npdes_id)
    facility_types = []

    # Add each relevant facility type
    for sic_code in sic_list:
        facility_types.append(check_sic_facility_type(sic_code))

    if 'sewer_system' in facility_types:
        return 'sewer_system'
    else:
        return 'other_system'


# def check_for_ww_permits(npdes_id):
#     """Check to see if any facilities associated with a given NPDES ID have a permit with a wastewater or sewer SIC code"""
#
#     sic_list = lookup_sic_code(npdes_id)
#     facility_types = []
#
#     # Add each relevant facility type
#     for sic_code in sic_list:
#         facility_types.append(check_sic_facility_type(sic_code))
#
#     if ('drinking_water' in facility_types) or ('other_system' in facility_types):
#         return facility_types
#     else:
#         return 'sewer_system'


def flag_not_ww(npdes_id):
    permits = check_for_ww_permits(npdes_id)

    if permits == 'sewer_system':
        return 'sewer_system'
    else:
        return 'REVIEW'


def view_all_sic_codes(npdes_id, review='multiple_water_permits'):
    """Make a dataframe with all the permits associated with a given NPDES.
    set review to either be "any_water_permit" to flag facilities for review that
    have at least 1 water permit; OR set to "multiple_water_permits" to require
    more than one water permit for a facility that """
    sic_list = lookup_water_sic_code(npdes_id)
    facility_types = []
    no_of_codes = len(sic_list)
    flag = []

    # make an array for npdes code tracking
    npdes_list = np.full(no_of_codes, npdes_id)

    permit_no = [i for i in range(1, no_of_codes + 1)]

    for sic_code in sic_list:
        facility_types.append(check_sic_facility_type(sic_code))

    # flags
    # Check if 'drinking_water' appears more than once in a list
    count_dw = facility_types.count('drinking_water')

    # Check all instances of sic codes that are other facilities
    if 'other_system' in facility_types:
        flag = np.full(no_of_codes, 'REVIEW')

    if review == 'any_water_permit':
        if 'drinking_water' in facility_types:
            flag = np.full(no_of_codes, 'REVIEW')
    elif count_dw > 1 or (len(facility_types) == 1 and 'drinking_water' in facility_types):
        flag = np.full(no_of_codes, 'REVIEW')
    else:
        flag = np.full(no_of_codes, 'OK')

    # combine the npdes list nad facility types into a df

    df = pd.DataFrame({
        'NPDES_ID': npdes_list,
        'sic_cod_no': permit_no,
        'sic_facility_type': facility_types,
        'flag': flag,
    })

    return df


def summarize_sic_codes(permit_list, col_name):
    unique_entries_count = permit_list[col_name].apply(tuple).nunique()

    # Flatten lists and convert them to tuples for consistency
    flat_column = permit_list[col_name].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Count unique entries and print them along with their counts
    unique_counts = flat_column.value_counts()
    print("Unique entries with counts:")
    print(unique_counts)
    return unique_counts