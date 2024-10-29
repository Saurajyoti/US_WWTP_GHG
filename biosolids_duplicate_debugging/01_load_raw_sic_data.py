# Setup
import pandas as pd
import numpy as np
import pathlib

# Import raw data
# Set low_memory to false to avoid a warning caused by mixed data types in each column
national_facility_raw = pd.read_csv(pathlib.PurePath('01_raw_data', 'NATIONAL_FACILITY_FILE.CSV'), low_memory=False)
national_sic_raw = pd.read_csv(pathlib.PurePath('01_raw_data', 'NATIONAL_SIC_FILE.CSV'), low_memory=False)

#%% View data

# View data
national_facility_raw.info()
national_sic_raw.info()

#%%
# national_facility_raw has the following columns:

"""
#   Column                          Dtype  
---  ------                          -----  
 0   FRS_FACILITY_DETAIL_REPORT_URL  object 
 1   REGISTRY_ID                     int64  
 2   PRIMARY_NAME                    object 
 3   LOCATION_ADDRESS                object 
 4   SUPPLEMENTAL_LOCATION           object 
 5   CITY_NAME                       object 
 6   COUNTY_NAME                     object 
 7   FIPS_CODE                       object 
 8   STATE_CODE                      object 
 9   STATE_NAME                      object 
 10  COUNTRY_NAME                    object 
 11  POSTAL_CODE                     object 
 12  FEDERAL_FACILITY_CODE           object 
 13  FEDERAL_AGENCY_NAME             object 
 14  TRIBAL_LAND_CODE                object 
 15  TRIBAL_LAND_NAME                object 
 16  CONGRESSIONAL_DIST_NUM          object 
 17  CENSUS_BLOCK_CODE               float64
 18  HUC_CODE                        float64
 19  EPA_REGION_CODE                 float64
 20  SITE_TYPE_NAME                  object 
 21  LOCATION_DESCRIPTION            object 
 22  CREATE_DATE                     object 
 23  UPDATE_DATE                     object 
 24  US_MEXICO_BORDER_IND            object 
 25  PGM_SYS_ACRNMS                  object 
 26  LATITUDE83                      float64
 27  LONGITUDE83                     float64
 28  CONVEYOR                        object 
 29  COLLECT_DESC                    object 
 30  ACCURACY_VALUE                  float64
 31  REF_POINT_DESC                  object 
 32  HDATUM_DESC                     object 
 33  SOURCE_DESC                     float64
 """
# Relevant columns are: REGISTRY_ID, PRIMARY_NAME, PGM_SYS_ACRNMS (this column contains NPDES number)
# Also useful for cross checking: LATITUDE83, LONGITUDE83, CITY_NAME, PRIMARY_NAME, LOCATION_ADDRESS


# national_sic_raw dataframe has the following columns:

"""
 #   Column             Dtype 
---  ------             ----- 
 0   REGISTRY_ID        int64 
 1   PGM_SYS_ACRNM      object
 2   PGM_SYS_ID         object
 3   INTEREST_TYPE      object
 4   SIC_CODE           int64 
 5   PRIMARY_INDICATOR  object
 6   CODE_DESCRIPTION   object
"""

# Relevant columns are REGISTRY_ID and SIC_CODE

#%%
# FILTER SIC CODES

# We need to filter for the relevant SIC codes to make the dataframe more manageable
# EPA PDF of SIC codes (not searchable): https://www3.epa.gov/npdes/pubs/app-c.pdf
# Look  up SIC codes here: https://www.naics.com/standard-industrial-code-divisions/?code=49

# Water supply = 4941

# Sewerage Systems = 4952
# Refuse System s= 4953
# Sanitary Services, Not Classified Elsewhere = 4959

relevant_sic_codes = [4941, 4952, 4953, 4959]
national_sic_filtered = national_sic_raw[national_sic_raw['SIC_CODE'].isin(relevant_sic_codes)]

# Only keep relevant columns
relevant_sic_cols = ['REGISTRY_ID', 'SIC_CODE']
national_sic_filtered = national_sic_filtered[relevant_sic_cols]

#%%
# FILTER FACILITIES DATABASE

relevant_facility_cols = ['REGISTRY_ID', 'PRIMARY_NAME', 'PGM_SYS_ACRNMS', 'LATITUDE83', 'LONGITUDE83', 'CITY_NAME', 'PRIMARY_NAME', 'LOCATION_ADDRESS' ]
national_facility_filtered = national_facility_raw[relevant_facility_cols]

#%%

# Merge the two datasets on REGISTRY_ID
# Conduct an inner join to only keep rows that appear in both REGISTRY_ID columns
facility_sic_filtered = pd.merge(national_facility_filtered, national_sic_filtered, on='REGISTRY_ID', how='inner')

# Save filtered dataframe
facility_sic_filtered.to_csv(pathlib.PurePath('02_clean_data', 'facility_sic_water.csv'))


#%%

# Save a clean dataframe of all SIC facilities

# national_facility with relevant columns only
national_facility = national_facility_raw[relevant_facility_cols]
national_sic = national_sic_raw[relevant_sic_cols]
facility_all_sic = pd.merge(national_facility, national_sic, on='REGISTRY_ID', how='inner')
facility_all_sic.to_csv(pathlib.PurePath('02_clean_data', 'facility_sic_all.csv'))

