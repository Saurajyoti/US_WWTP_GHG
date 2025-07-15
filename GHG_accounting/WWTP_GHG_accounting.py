#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Inventory of wastewater plants in the United States and their associated energy
usage and greenhouse gas emissions

The code is developed by:
    Jianan Feng <jiananf2@illinois.edu>
    
Revelant publications:
    [1] El Abbadi, S. H.; Feng, J.; Hodson, A. R.; Amouamouha, M.; Busse, M. M.;
        Polcuch, C.; Zhou, P.; Macknick, J.; Guest, J. S.; Stokes-Draut, J. R.;
        Dunn, J. B. Benchmarking Greenhouse Gas Emissions from U.S. Wastewater 
        reatment for Targeted Reduction. 2024. https://doi.org/10.31223/X5VQ59.
    [2] Feng, J.; Strathmann, T.; Guest, J. Financially Driven Hydrothermal-Based
        Wastewater Solids Management for Targeted Resource Recovery and
        Decarbonization in the Contiguous U.S. ChemRxiv May 28, 2025.
        https://doi.org/10.26434/chemrxiv-2025-qfxwd.

References and data sources can be found in [1].
'''

#%% initialization

import numpy as np, pandas as pd, scipy.stats as st, chaospy as cp, geopandas as gpd, matplotlib.pyplot as plt, seaborn as sns
import matplotlib.colors as colors
from colorpalette import Color
from chaospy import distributions as shape
from scipy import stats
from matplotlib.mathtext import _mathtext as mathtext
from matplotlib.patches import Rectangle
from math import pi

# palette
b = Color('blue', (96, 193, 207)).HEX
g = Color('green', (121, 191, 130)).HEX
r = Color('red', (237, 88, 111)).HEX
o = Color('orange', (249, 143, 96)).HEX
y = Color('yellow', (243, 195, 84)).HEX
a = Color('gray', (144, 145, 142)).HEX
p = Color('purple', (162, 128, 185)).HEX

db = Color('dark_blue', (53, 118, 127)).HEX
dg = Color('dark_green', (77, 126, 83)).HEX
dr = Color('dark_red', (156, 75, 80)).HEX
do = Color('dark_orange', (167, 95, 62)).HEX
dy = Color('dark_yellow', (171, 137, 55)).HEX
da = Color('dark_gray', (78, 78, 78)).HEX
dp = Color('dark_purple', (76, 56, 90)).HEX

# for contour plots
color_map = colors.LinearSegmentedColormap.from_list('color_map', [b, g, y, o, r])

# constants
# MG: million gallon
MG_2_m3 = 3785.4
kWh_2_MJ = 3.6
# convert N2O-N to N2O
N_2_N2O = 44/28
# 2023 Biosolids Annual Reports, EPA
ratio_LF_LF_LA = 24/(24+60)
ratio_LA_LF_LA = 60/(24+60)
ratio_LF_LF_IN = 24/(24+14)
ratio_IN_LF_IN = 14/(24+14)
ratio_LA_LA_IN = 60/(60+14)
ratio_IN_LA_IN = 14/(60+14)
ratio_LF_all = 24/(24+60+14)
ratio_LA_all = 60/(24+60+14)
ratio_IN_all = 14/(24+60+14)

crosswalk = {'B1':'*A1',
             'B1E':'*A1e',
             'B2':'*A3',
             'B3':'*A4',
             'B4':'*A2',
             'B5':'*A5',
             'B6':'*A6',
             'C1':'A1',
             'C1E':'A1e',
             'C2':'A3',
             'C3':'A4',
             'C5':'A5',
             'C6':'A6',
             'D1':'*C1',
             'D1E':'*C1e',
             'D2':'*C3',
             'D3':'*C4',
             'D5':'*C5',
             'D6':'*C6',
             'E2':'E3',
             'E2P':'*E3',
             'F1':'*E1',
             'F1E':'*E1e',
             'G1':'*G1',
             'G1E':'*G1e',
             'G2':'*G3',
             'G3':'*G4',
             'G5':'*G5',
             'G6':'*G6',
             'H1':'*G1-p',
             'H1E':'*G1e-p',
             'I1':'F1',
             'I1E':'F1e',
             'I2':'F3',
             'I3':'F4',
             'I5':'F5',
             'I6':'F6',
             'LAGOON_AER':'L-a',
             'LAGOON_ANAER':'L-n',
             'LAGOON_FAC':'L-f',
             'LAGOON_UNCATEGORIZED':'L-u',
             'N1':'*D1',
             'N1E':'*D1e',
             'N2':'*D3',
             'O1':'*B1',
             'O1E':'*B1e',
             'O2':'*B3',
             'O3':'*B4',
             'O5':'*B5',
             'O6':'*B6'}

#%% load treatment train information for each WWTP

WWTP_TT = pd.read_csv('tt_assignments_2022.csv')

WWTP_info =  pd.read_csv('all_wwtps_data_070124.csv')
WWTP_info = WWTP_info[['CWNS_ID','FACILITY_NAME','CITY',
                       'STATE_CODE','LATITUDE','LONGITUDE']]

WWTP_info.rename(columns={'CWNS_ID':'CWNS_NUM',
                          'FACILITY_NAME':'FACILITY',
                          'STATE_CODE':'STATE'},
                 inplace=True)

assert WWTP_TT.duplicated(subset='CWNS_NUM').sum() == 0
assert WWTP_info.duplicated(subset='CWNS_NUM').sum() == 0

WWTP_test = WWTP_TT.merge(WWTP_info, on='CWNS_NUM', how='inner')

assert len(WWTP_TT) == len(WWTP_test)

WWTP_TT = WWTP_test

WWTP_TT['LAGOON_UNCATEGORIZED'] = WWTP_TT[['LAGOON_OTHER','STBL_POND']].max(axis=1)

WWTP_TT['TT_IDENTIFIED'] = WWTP_TT[crosswalk.keys()].sum(axis=1)

assert (WWTP_TT['TT_IDENTIFIED'] >= 1).sum() == len(WWTP_TT)

non_continental = ['HI','VI','MP','GU','AK','AS','PR']

WWTP_TT = WWTP_TT[~WWTP_TT['STATE'].isin(non_continental)]

#%% calculate LAGOON_UNCATEGORIZED composition

# based on the flow weighted average of aerobic and (anaerobic+facultative)
# lagoons of the rest of the WRRFs in the contiguous U.S.
L_AE_flow = (WWTP_TT['LAGOON_AER']/WWTP_TT['TT_IDENTIFIED']*WWTP_TT['FLOW_2022_MGD_FINAL']).sum()
L_AN_FA_flow = ((WWTP_TT['LAGOON_ANAER']+WWTP_TT['LAGOON_FAC'])/\
                WWTP_TT['TT_IDENTIFIED']*WWTP_TT['FLOW_2022_MGD_FINAL']).sum()

L_AE_flow_ratio = L_AE_flow/(L_AE_flow+L_AN_FA_flow)
L_AN_FA_flow_ratio = L_AN_FA_flow/(L_AE_flow+L_AN_FA_flow)

#%% find TT codes used in WWTPs

final_code = []

# remove TTs that no WWTP is assined to
for TT in list(crosswalk.keys()):
    if WWTP_TT[TT].sum() != 0:
        final_code.append(TT)

# confirm each TT occur up to once for every WWTP
assert WWTP_TT[final_code].max().max() == 1
assert WWTP_TT[final_code].min().min() == 0

#%% data preparation - TT electricity and NG

energy_typical = pd.read_excel('Treatment Train Energy Spreadsheet (Typical).xlsx','All Trains (For Code)')
energy_typical.fillna(0, inplace=True)
energy_typical.rename(columns={'Unnamed: 0':'item'}, inplace=True)
energy_typical.set_index('item', inplace=True)
energy_typical = energy_typical.loc[['Total Electricity Usage [kWh/d] (including chemical production)',
                                     'Total Electricity Usage [kWh/d] (excluding chemical production)',
                                     'CHP Electricity Generation [kWh/d]',
                                     'Total Natural Gas Usage [MJ/d] (including chemical production)',
                                     'Total Natural Gas Usage [MJ/d] (excluding chemical production)']]

energy_typical.loc['typical_elec_usage_chemical'] = energy_typical.loc['Total Electricity Usage [kWh/d] (including chemical production)'] -\
                                                    energy_typical.loc['Total Electricity Usage [kWh/d] (excluding chemical production)']
energy_typical.loc['typical_NG_usage_chemical'] = energy_typical.loc['Total Natural Gas Usage [MJ/d] (including chemical production)'] -\
                                                    energy_typical.loc['Total Natural Gas Usage [MJ/d] (excluding chemical production)']

energy_typical.drop(['Total Electricity Usage [kWh/d] (excluding chemical production)',
                     'Total Natural Gas Usage [MJ/d] (excluding chemical production)'], inplace=True)

energy_typical.rename(index={'Total Electricity Usage [kWh/d] (including chemical production)':'typical_elec_usage',
                             'CHP Electricity Generation [kWh/d]':'typical_elec_CHP',
                             'Total Natural Gas Usage [MJ/d] (including chemical production)':'typical_NG_usage'}, inplace=True)

energy_best = pd.read_excel('Treatment Train Energy Spreadsheet (Best Practice).xlsx','All Trains (For Code)')
energy_best.fillna(0, inplace=True)
energy_best.rename(columns={'Unnamed: 0':'item'}, inplace=True)
energy_best.set_index('item', inplace=True)
energy_best.rename(index={'Total Electricity Usage [kWh/d]':'best_elec_usage',
                          'CHP Electricity Generation [kWh/d]':'best_elec_CHP',
                          'Total Natural Gas Usage [MJ/d]':'best_NG_usage'}, inplace=True)

elec_best_WERF = pd.read_excel('Treatment Train Energy Spreadsheet (Best Practice).xlsx','WERF Trains (Electricity)')
elec_best_WERF.set_index('All values in kWh/day', inplace=True)
elec_best_WERF = elec_best_WERF.loc[['Hypchlorite Production','Acetic Acid Production','Lime Production']].sum()

elec_best_Franken = pd.read_excel('Treatment Train Energy Spreadsheet (Best Practice).xlsx','Frankentrains (Electricity)')
elec_best_Franken.set_index('All values in kWh/day', inplace=True)
elec_best_Franken = elec_best_Franken.loc[['Hypochlorite Production','Acetic Acid Production','Lime Production']].sum()

elec_best_chemical = pd.concat([elec_best_WERF, elec_best_Franken])

NG_best_WERF = pd.read_excel('Treatment Train Energy Spreadsheet (Best Practice).xlsx','WERF Trains (Natural Gas)')
NG_best_WERF.set_index('Unnamed: 0', inplace=True)
NG_best_WERF = NG_best_WERF.loc[['Fuel Used for Chemical Production (Acetic Acid)',
                                 'Fuel Used for Chemical Production (Methanol)',
                                 'Fuel Used for Chemical Production (Lime)']].sum()

NG_best_Franken = pd.read_excel('Treatment Train Energy Spreadsheet (Best Practice).xlsx','Frankentrains (Natural Gas)')
NG_best_Franken.set_index('Unnamed: 0', inplace=True)
NG_best_Franken = NG_best_Franken.loc[['Fuel Used for Chemical Production (Acetic Acid)',
                                       'Fuel Used for Chemical Production (Methanol)',
                                       'Fuel Used for Chemical Production (Lime)']].sum()

NG_best_chemical = pd.concat([NG_best_WERF, NG_best_Franken])

assert (energy_best.columns != elec_best_chemical.index).sum() == 0
energy_best.loc['best_elec_usage_chemical'] = elec_best_chemical

assert (energy_best.columns != NG_best_chemical.index).sum() == 0
energy_best.loc['best_NG_usage_chemical'] = NG_best_chemical

energy_uncertainty = pd.concat([energy_typical, energy_best])

#%% data preparation - WWTP electricity

# electricity
# balancing area
balnc_area = pd.read_excel('WWTP Baseline Trains_8.xlsx','Balance_Area')

upstream_elec_GHG = UEG = {# item          :   kg CO2e/MWh
                            'natural_gas'  :   24/1000*kWh_2_MJ*1000,
                            'coal'         :   18/1000*kWh_2_MJ*1000,
                            'nuclear'      :   1.9/1000*kWh_2_MJ*1000,
                            'wind'         :   2.86/1000*kWh_2_MJ*1000,
                            'solar'        :   10.48/1000*kWh_2_MJ*1000,
                            'biomass'      :   19.02/1000*kWh_2_MJ*1000,
                            'geothermal'   :   1.35/1000*kWh_2_MJ*1000,
                            'hydro'        :   2.08/1000*kWh_2_MJ*1000}

balnc_area['CO2_kg_total'] = balnc_area['co2_gen_mmt']*1000000000 +\
                             balnc_area['gas-ct_MWh']*UEG['natural_gas'] +\
                             balnc_area['gas-cc_MWh']*UEG['natural_gas'] +\
                             balnc_area['coal_MWh']*UEG['coal'] +\
                             balnc_area['nuclear_MWh']*UEG['nuclear'] +\
                             balnc_area['wind-ons_MWh']*UEG['wind'] +\
                             balnc_area['wind-ofs_MWh']*UEG['wind'] +\
                             balnc_area['csp_MWh']*UEG['solar'] +\
                             balnc_area['upv_MWh']*UEG['solar'] +\
                             balnc_area['distpv_MWh']*UEG['solar'] +\
                             balnc_area['o-g-s_MWh']*UEG['solar'] +\
                             balnc_area['biomass_MWh']*UEG['biomass'] +\
                             balnc_area['geothermal_MWh']*UEG['geothermal'] +\
                             balnc_area['phs_MWh']*UEG['hydro'] +\
                             balnc_area['hydro_MWh']*UEG['hydro']

# balancing_area['generation'] in MWh
balnc_area['kg_CO2_kWh'] = balnc_area['CO2_kg_total']/balnc_area['generation']/1000

# just 2020
balnc_area = balnc_area.loc[balnc_area['t'] == 2020]
# remove 'p'
balnc_area['r'] = balnc_area['r'].str[1:].astype(int)
balnc_area = balnc_area[['r','state','kg_CO2_kWh']]

assert balnc_area.duplicated(subset='r').sum() == 0

balnc_area_WWTP = pd.read_excel('WWTP_balancing_area.xlsx')

balnc_area_WWTP = balnc_area_WWTP.merge(balnc_area,
                                        how='inner',
                                        left_on='balancing_area',
                                        right_on='r')

assert WWTP_TT.duplicated(subset='CWNS_NUM').sum() == 0
assert balnc_area_WWTP.duplicated(subset='CWNS_NUM').sum() == 0

WWTP_TT = WWTP_TT.merge(balnc_area_WWTP, how='left', on='CWNS_NUM')

assert WWTP_TT['kg_CO2_kWh'].isna().sum() == 0

#%% data preparation - biosolids

# biosolids in tonne/year
biosolids = pd.read_csv('biosolids_cwns_match_update_032524.csv')

biosolids.fillna({'Amount of Biosolids Managed - Land Applied': 0}, inplace=True)
biosolids.fillna({'Amount of Biosolids Managed - Surface Disposal': 0}, inplace=True)
biosolids.fillna({'Amount of Biosolids Managed - Incinerated': 0}, inplace=True)
biosolids.fillna({'Amount of Biosolids Managed - Other Management Practice': 0}, inplace=True)

biosolids['total_biosolids'] = biosolids['Amount of Biosolids Managed - Land Applied'] +\
                               biosolids['Amount of Biosolids Managed - Surface Disposal']+\
                               biosolids['Amount of Biosolids Managed - Incinerated']+\
                               biosolids['Amount of Biosolids Managed - Other Management Practice']

biosolids = biosolids[['CWNS',
                       'total_biosolids',
                       'Management Practice Type(s)',
                       'Amount of Biosolids Managed - Land Applied',
                       'Amount of Biosolids Managed - Surface Disposal',
                       'Amount of Biosolids Managed - Incinerated',
                       'Amount of Biosolids Managed - Other Management Practice']]

assert biosolids.duplicated(subset=['CWNS']).sum() == 0
assert WWTP_TT.duplicated(subset=['CWNS_NUM']).sum() == 0

WWTP_TT = WWTP_TT.merge(biosolids, how='left', left_on='CWNS_NUM', right_on='CWNS')

# calculate the ratio between biosolids amount and plant size
WWTP_TT['biosolids_MGD_ratio'] = WWTP_TT['total_biosolids']/WWTP_TT['FLOW_2022_MGD_FINAL']

# remove outliers from biosolids data (set 'Management Practice Type(s)' as np.nan)
# use 10th and 90th
quantile_10 = WWTP_TT['biosolids_MGD_ratio'].dropna().quantile(0.1)
quantile_90 = WWTP_TT['biosolids_MGD_ratio'].dropna().quantile(0.9)

WWTP_TT.loc[((WWTP_TT['biosolids_MGD_ratio'] < quantile_10) |\
             (WWTP_TT['biosolids_MGD_ratio'] > quantile_90)),\
            'Management Practice Type(s)'] = np.nan

# calculate the biosolids amount in kg/year
# step 1: if having biosolids data, use biosolids data

# surface disposal = landfilling
# other_management = landfilling + land application
WWTP_TT.loc[WWTP_TT['Management Practice Type(s)'].notna(), 'landfill'] =\
    (WWTP_TT['Amount of Biosolids Managed - Surface Disposal'] +\
     WWTP_TT['Amount of Biosolids Managed - Other Management Practice']*ratio_LF_LF_LA)*1000
        
WWTP_TT.loc[WWTP_TT['Management Practice Type(s)'].notna(), 'land_application'] =\
    (WWTP_TT['Amount of Biosolids Managed - Land Applied'] +\
     WWTP_TT['Amount of Biosolids Managed - Other Management Practice']*ratio_LA_LF_LA)*1000
        
WWTP_TT.loc[WWTP_TT['Management Practice Type(s)'].notna(), 'incineration'] =\
    WWTP_TT['Amount of Biosolids Managed - Incinerated']*1000

# step 2: if no biosolids data, calculate the total amount of biosolids

TT_w_IN = ['B5','B6','C5','C6','D5','D6','G5','G6','I5','I6','O5','O6']
TT_w_primary_IN = ['B5','B6','D5','D6','G5','G6','O5','O6']
TT_wo_primary_IN = ['C5','C6','I5','I6']
TT_w_primary_AD = ['B1','B1E','B4','D1','D1E','F1','F1E','G1','G1E','H1','H1E','N1','N1E','O1','O1E']
TT_w_primary_AeD = ['B2','D2','E2P','G2','N2','O2']
TT_w_primary_none = ['B3','B5','B6','D3','D5','D6','G3','G5','G6','O3','O5','O6']
TT_w_primary_lime = ['B3','D3','G3','O3']
TT_wo_primary_AD = ['C1','C1E','I1','I1E','LAGOON_ANAER','LAGOON_FAC']
TT_wo_primary_AeD = ['C2','E2','I2','LAGOON_AER']
TT_wo_primary_L_u = ['LAGOON_UNCATEGORIZED']
TT_wo_primary_none = ['C3','C5','C6','I3','I5','I6']
TT_wo_primary_lime = ['C3','I3']

# kg/m3
sludge_w_primary = 0.2636
sludge_wo_primary = 0.131

# VS removal: AD (35-50%), AeD (40-55%) (Metcalf and Eddy, 2003)
# VS content of biosolids: 60-80% (Agricultural Recycling of Sewage Sludge and
# the Environment, National Research Council (US) Committee on Land Application
# of Sewage Sludge, 1996)
# VS content of uncomposted biosolids: 40-60% (Tchobanoglous textbook)
# assume 60% VS, 42.5% VS reduction for AD and 47.5% for AeD
TS_2_VS = 0.6
reduction_AD = 0.425
reduction_AeD = 0.475
# use the flow weighted average for uncategorized lagoon
reduction_uncategorized = reduction_AeD*L_AE_flow_ratio + reduction_AD*L_AN_FA_flow_ratio

coefficient = WWTP_TT['FLOW_2022_MGD_FINAL']/WWTP_TT['TT_IDENTIFIED']

# calculate the theoretical biosolids amount in kg/year
WWTP_TT['theoretical_biosolids'] = (WWTP_TT[TT_w_primary_AD].sum(axis=1)*\
                                        coefficient*sludge_w_primary*\
                                            (1-TS_2_VS*reduction_AD) +\
                                    WWTP_TT[TT_w_primary_AeD].sum(axis=1)*\
                                        coefficient*sludge_w_primary*\
                                            (1-TS_2_VS*reduction_AeD) +\
                                    WWTP_TT[TT_w_primary_none].sum(axis=1)*\
                                        coefficient*sludge_w_primary +\
                                    WWTP_TT[TT_wo_primary_AD].sum(axis=1)*\
                                        coefficient*sludge_wo_primary*\
                                            (1-TS_2_VS*reduction_AD) +\
                                    WWTP_TT[TT_wo_primary_AeD].sum(axis=1)*\
                                        coefficient*sludge_wo_primary*\
                                            (1-TS_2_VS*reduction_AeD) +\
                                    WWTP_TT[TT_wo_primary_L_u].sum(axis=1)*\
                                        coefficient*sludge_wo_primary*\
                                            (1-TS_2_VS*reduction_uncategorized) +\
                                    WWTP_TT[TT_wo_primary_none].sum(axis=1)*\
                                        coefficient*sludge_wo_primary)*MG_2_m3*365

# step 3: for plants w/o biosolids data (WWTP_TT['Management Practice Type(s)'].isna())
# step 3.1: calculate the biosolids amount based on disposal_2022.csv data

disposal = pd.read_csv('disposal_2022.csv')

assert disposal.duplicated(subset=['CWNS_NUM']).sum() == 0
assert WWTP_TT.duplicated(subset=['CWNS_NUM']).sum() == 0

WWTP_TT = WWTP_TT.merge(disposal, how='left', on='CWNS_NUM')

# LF = landfill, LA = land application, IN = incineration
data_22_LF = (WWTP_TT['Management Practice Type(s)'].isna()) &\
             (WWTP_TT['LANDFILL'].notna()) &\
             (WWTP_TT['LAND_APP'].notna()) &\
             (WWTP_TT['FBI_y'].notna()) &\
             (WWTP_TT['MHI_y'].notna()) &\
             (WWTP_TT['LANDFILL'] != 0) &\
             (WWTP_TT['LAND_APP'] == 0) &\
             (WWTP_TT['FBI_y'] == 0) &\
             (WWTP_TT['MHI_y'] == 0)

WWTP_TT.loc[data_22_LF, 'landfill'] = WWTP_TT['theoretical_biosolids']

data_22_LA = (WWTP_TT['Management Practice Type(s)'].isna()) &\
             (WWTP_TT['LANDFILL'].notna()) &\
             (WWTP_TT['LAND_APP'].notna()) &\
             (WWTP_TT['FBI_y'].notna()) &\
             (WWTP_TT['MHI_y'].notna()) &\
             (WWTP_TT['LANDFILL'] == 0) &\
             (WWTP_TT['LAND_APP'] != 0) &\
             (WWTP_TT['FBI_y'] == 0) &\
             (WWTP_TT['MHI_y'] == 0)

WWTP_TT.loc[data_22_LA, 'land_application'] = WWTP_TT['theoretical_biosolids']

data_22_IN = (WWTP_TT['Management Practice Type(s)'].isna()) &\
             (WWTP_TT['LANDFILL'].notna()) &\
             (WWTP_TT['LAND_APP'].notna()) &\
             (WWTP_TT['FBI_y'].notna()) &\
             (WWTP_TT['MHI_y'].notna()) &\
             (WWTP_TT['LANDFILL'] == 0) &\
             (WWTP_TT['LAND_APP'] == 0) &\
             ((WWTP_TT['FBI_y'] != 0) |\
              (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_IN, 'incineration'] = WWTP_TT['theoretical_biosolids']

data_22_LF_LA = (WWTP_TT['Management Practice Type(s)'].isna()) &\
                (WWTP_TT['LANDFILL'].notna()) &\
                (WWTP_TT['LAND_APP'].notna()) &\
                (WWTP_TT['FBI_y'].notna()) &\
                (WWTP_TT['MHI_y'].notna()) &\
                (WWTP_TT['LANDFILL'] != 0) &\
                (WWTP_TT['LAND_APP'] != 0) &\
                (WWTP_TT['FBI_y'] == 0) &\
                (WWTP_TT['MHI_y'] == 0)

WWTP_TT.loc[data_22_LF_LA, 'landfill'] = WWTP_TT['theoretical_biosolids']*ratio_LF_LF_LA
WWTP_TT.loc[data_22_LF_LA, 'land_application'] = WWTP_TT['theoretical_biosolids']*ratio_LA_LF_LA

data_22_LF_IN = (WWTP_TT['Management Practice Type(s)'].isna()) &\
                (WWTP_TT['LANDFILL'].notna()) &\
                (WWTP_TT['LAND_APP'].notna()) &\
                (WWTP_TT['FBI_y'].notna()) &\
                (WWTP_TT['MHI_y'].notna()) &\
                (WWTP_TT['LANDFILL'] != 0) &\
                (WWTP_TT['LAND_APP'] == 0) &\
                ((WWTP_TT['FBI_y'] != 0) |\
                 (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_LF_IN, 'landfill'] = WWTP_TT['theoretical_biosolids']*ratio_LF_LF_IN
WWTP_TT.loc[data_22_LF_IN, 'incineration'] = WWTP_TT['theoretical_biosolids']*ratio_IN_LF_IN

data_22_LA_IN = (WWTP_TT['Management Practice Type(s)'].isna()) &\
                (WWTP_TT['LANDFILL'].notna()) &\
                (WWTP_TT['LAND_APP'].notna()) &\
                (WWTP_TT['FBI_y'].notna()) &\
                (WWTP_TT['MHI_y'].notna()) &\
                (WWTP_TT['LANDFILL'] == 0) &\
                (WWTP_TT['LAND_APP'] != 0) &\
                ((WWTP_TT['FBI_y'] != 0) |\
                 (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_LA_IN, 'land_application'] = WWTP_TT['theoretical_biosolids']*ratio_LA_LA_IN
WWTP_TT.loc[data_22_LA_IN, 'incineration'] = WWTP_TT['theoretical_biosolids']*ratio_IN_LA_IN

data_22_all = (WWTP_TT['Management Practice Type(s)'].isna()) &\
              (WWTP_TT['LANDFILL'].notna()) &\
              (WWTP_TT['LAND_APP'].notna()) &\
              (WWTP_TT['FBI_y'].notna()) &\
              (WWTP_TT['MHI_y'].notna()) &\
              (WWTP_TT['LANDFILL'] != 0) &\
              (WWTP_TT['LAND_APP'] != 0) &\
              ((WWTP_TT['FBI_y'] != 0) |\
               (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_all, 'landfill'] = WWTP_TT['theoretical_biosolids']*ratio_LF_all
WWTP_TT.loc[data_22_all, 'land_application'] = WWTP_TT['theoretical_biosolids']*ratio_LA_all
WWTP_TT.loc[data_22_all, 'incineration'] = WWTP_TT['theoretical_biosolids']*ratio_IN_all

# step 3.2: calculate the biosolids amount for the rest of WWTPs based on TTs

TT_IN = (WWTP_TT['Management Practice Type(s)'].isna()) &\
        (WWTP_TT['LANDFILL'].isna()) &\
        (WWTP_TT['LAND_APP'].isna()) &\
        (WWTP_TT['FBI_y'].isna()) &\
        (WWTP_TT['MHI_y'].isna()) &\
        (WWTP_TT['TT_IDENTIFIED'] == 1) &\
        ((WWTP_TT['B5'] == 1) |\
         (WWTP_TT['B6'] == 1) |\
         (WWTP_TT['C5'] == 1) |\
         (WWTP_TT['C6'] == 1) |\
         (WWTP_TT['D5'] == 1) |\
         (WWTP_TT['D6'] == 1) |\
         (WWTP_TT['G5'] == 1) |\
         (WWTP_TT['G6'] == 1) |\
         (WWTP_TT['I5'] == 1) |\
         (WWTP_TT['I6'] == 1) |\
         (WWTP_TT['O5'] == 1) |\
         (WWTP_TT['O6'] == 1))

if TT_IN.sum() > 0:
    WWTP_TT.loc[TT_IN, 'incineration'] = WWTP_TT['theoretical_biosolids']

TT_disposal = (WWTP_TT['Management Practice Type(s)'].isna()) &\
              (WWTP_TT['LANDFILL'].isna()) &\
              (WWTP_TT['LAND_APP'].isna()) &\
              (WWTP_TT['FBI_y'].isna()) &\
              (WWTP_TT['MHI_y'].isna()) &\
              (WWTP_TT['B5'] == 0) &\
              (WWTP_TT['B6'] == 0) &\
              (WWTP_TT['C5'] == 0) &\
              (WWTP_TT['C6'] == 0) &\
              (WWTP_TT['D5'] == 0) &\
              (WWTP_TT['D6'] == 0) &\
              (WWTP_TT['G5'] == 0) &\
              (WWTP_TT['G6'] == 0) &\
              (WWTP_TT['I5'] == 0) &\
              (WWTP_TT['I6'] == 0) &\
              (WWTP_TT['O5'] == 0) &\
              (WWTP_TT['O6'] == 0)

if TT_disposal.sum() > 0:
    WWTP_TT.loc[TT_disposal, 'landfill'] = WWTP_TT['theoretical_biosolids']*ratio_LF_LF_LA
    WWTP_TT.loc[TT_disposal, 'land_application'] = WWTP_TT['theoretical_biosolids']*ratio_LA_LF_LA

TT_IN_disposal = (WWTP_TT['Management Practice Type(s)'].isna()) &\
                 (WWTP_TT['LANDFILL'].isna()) &\
                 (WWTP_TT['LAND_APP'].isna()) &\
                 (WWTP_TT['FBI_y'].isna()) &\
                 (WWTP_TT['MHI_y'].isna()) &\
                 (WWTP_TT['TT_IDENTIFIED'] > 1) &\
                 ((WWTP_TT['B5'] == 1) |\
                  (WWTP_TT['B6'] == 1) |\
                  (WWTP_TT['C5'] == 1) |\
                  (WWTP_TT['C6'] == 1) |\
                  (WWTP_TT['D5'] == 1) |\
                  (WWTP_TT['D6'] == 1) |\
                  (WWTP_TT['G5'] == 1) |\
                  (WWTP_TT['G6'] == 1) |\
                  (WWTP_TT['I5'] == 1) |\
                  (WWTP_TT['I6'] == 1) |\
                  (WWTP_TT['O5'] == 1) |\
                  (WWTP_TT['O6'] == 1))

if TT_IN_disposal.sum() > 0:
    WWTP_TT.loc[TT_IN_disposal, 'incineration'] =\
        (WWTP_TT[TT_w_primary_IN].sum(axis=1)*\
             coefficient*sludge_w_primary +\
                 WWTP_TT[TT_wo_primary_IN].sum(axis=1)*\
                     coefficient*sludge_wo_primary)*MG_2_m3*365

    assert (WWTP_TT.loc[TT_IN_disposal,'theoretical_biosolids']-\
            WWTP_TT.loc[TT_IN_disposal,'incineration']).min() >= 0
    
    WWTP_TT.loc[TT_IN_disposal, 'landfill'] = (WWTP_TT['theoretical_biosolids']-\
                                               WWTP_TT['incineration'])*ratio_LF_LF_LA
    WWTP_TT.loc[TT_IN_disposal, 'land_application'] = (WWTP_TT['theoretical_biosolids']-\
                                                       WWTP_TT['incineration'])*ratio_LA_LF_LA

WWTP_TT.fillna({'landfill': 0}, inplace=True)
WWTP_TT.fillna({'land_application': 0}, inplace=True)
WWTP_TT.fillna({'incineration': 0}, inplace=True)

assert WWTP_TT[['landfill','land_application','incineration']].sum(axis=1).min() > 0

#%% TT Monte Carlo

for TT in final_code:
    if TT[0] != 'L':
        elec_lower = min(energy_uncertainty[TT]['best_elec_usage'] - energy_uncertainty[TT]['best_elec_CHP'],
                         energy_uncertainty[TT]['typical_elec_usage'] - energy_uncertainty[TT]['typical_elec_CHP'])/10
        elec_baseline = max(energy_uncertainty[TT]['best_elec_usage'] - energy_uncertainty[TT]['best_elec_CHP'],
                            energy_uncertainty[TT]['typical_elec_usage'] - energy_uncertainty[TT]['typical_elec_CHP'])/10
        elec_upper = 2*elec_baseline - elec_lower
        elec_MC = shape.Uniform(elec_lower, elec_upper)
        if energy_uncertainty[TT]['typical_NG_usage'] == energy_uncertainty[TT]['best_NG_usage']:
            NG_MC = energy_uncertainty[TT]['typical_NG_usage']/10
        else:
            NG_lower = min(energy_uncertainty[TT]['best_NG_usage'], energy_uncertainty[TT]['typical_NG_usage'])/10
            NG_baseline = max(energy_uncertainty[TT]['best_NG_usage'], energy_uncertainty[TT]['typical_NG_usage'])/10
            NG_upper = 2*NG_baseline - NG_lower
            NG_MC = shape.Uniform(NG_lower, NG_upper)
    elif TT == 'LAGOON_AER':
        elec_MC = shape.Uniform(1386*0.8, 1386*1.2)
    elif TT in ['LAGOON_ANAER','LAGOON_FAC']:
        elec_MC = shape.Uniform(660*0.8, 660*1.2)
    elif TT == 'LAGOON_UNCATEGORIZED':
        elec_MC = shape.Uniform(1386*L_AE_flow_ratio*0.8 + 660*L_AN_FA_flow_ratio*0.8,
                                1386*L_AE_flow_ratio*1.2 + 660*L_AN_FA_flow_ratio*1.2)
    
    elec_CI_MC = shape.Triangle(balnc_area['kg_CO2_kWh'].quantile(0.05),
                                balnc_area['kg_CO2_kWh'].quantile(0.5),
                                balnc_area['kg_CO2_kWh'].quantile(0.95))
    
    # from GREET: onsite NG combustion emission, kg CO2-eq/MJ
    NG_combustion_CI_MC = shape.Uniform(56.3/1000*0.9, 56.3/1000*1.1)
    # from GREET: upstream NG emission, kg CO2-eq/MJ
    NG_upstream_CI_MC = shape.Uniform(12.7/1000*0.9, 12.7/1000*1.1)
    
    COD_MC = shape.Uniform(339/1000, 1016/1000)
    
    # TTs with anaerobic digestion
    w_AD = ['B1','B1E','B4',
            'C1','C1E',
            'D1','D1E',
            'F1','F1E',
            'G1','G1E',
            'H1','H1E',
            'I1','I1E',
            'L1',
            'M1',
            'N1','N1E',
            'O1','O1E',
            'P1','P1E']
    
    # TTs without anaerobic digestion and without lagoon
    wo_AD_wo_lagoon = ['B2','B3','B5','B6',
                       'C2','C3','C5','C6',
                       'D2','D3','D5','D6',
                       'E2','E2P',
                       'G2','G3','G5','G6',
                       'I2','I3','I5','I6',
                       'N2',
                       'O2','O3','O5','O6']
    
    if TT in w_AD:
        data = np.array([5.62, 7.25, 7.95, 8.68, 9.63, 12.17, 12.45, 17.30, 21.19, 25.48])/1000
        sh, loc, scale = st.lognorm.fit(data, floc=0)
        mu = np.log(scale)
        CH4_EF_MC = shape.LogNormal(mu, sh, loc)
    elif TT in wo_AD_wo_lagoon:
        data = np.array([0.22, 0.40, 0.99, 1.56, 2.17, 2.44, 18.38, 0.01, 0.03,
                         0.34, 0.36, 1.01, 1.15, 1.32, 6.40, 11.12, 17.29])/1000
        sh, loc, scale = st.weibull_min.fit(data, floc=0)
        CH4_EF_MC = shape.Weibull(sh, scale, loc)
    elif TT == 'LAGOON_AER':
        CH4_EF_MC = 0
    elif TT in ['LAGOON_ANAER','LAGOON_FAC']:
        CH4_COD_EF_MC = shape.Uniform(0, 0.0975)
    elif TT == 'LAGOON_UNCATEGORIZED':
        CH4_COD_EF_MC = shape.Uniform(0, 0.0975*L_AN_FA_flow_ratio)
    else:
        raise ValueError(f'{TT} does not exist.')
    
    CH4_CF_MC = shape.Uniform(29.8*0.9, 29.8*1.1)

    TN_MC = shape.Uniform(23/1000, 69/1000)
    
    # organics removal
    if TT[0] in ['B','C','D','O']:
        N2O_EF_MC = shape.Triangle(0.000035502958579881655, 0.0014563706563706566, 0.0062)
    # nitrification
    elif TT[0] in ['E','F']:
        N2O_EF_MC = shape.Triangle(0.0003008785529715762, 0.0021505376344086022, 0.04639999999999992)
    # BNR
    elif TT[0] in ['G','H','I','N']:
        N2O_EF_MC = shape.Triangle(0.000044063593004769475, 0.0043858612, 0.03861727897938609)
    elif TT == 'LAGOON_AER':
        N2O_EF_MC = shape.Uniform(0.00016, 0.045)
    elif TT in ['LAGOON_ANAER','LAGOON_FAC']:
        N2O_EF_MC = 0
    elif TT == 'LAGOON_UNCATEGORIZED':
        N2O_EF_MC = shape.Uniform(0.00016*L_AE_flow_ratio, 0.045*L_AE_flow_ratio)
    else:
        raise ValueError(f'{TT} does not exist.')
    
    N2O_CF_MC = shape.Uniform(273*0.9, 273*1.1)
    
    fossil_COD_MC = shape.Triangle(0.001, 0.119, 0.279)
    assimilated_COD_MC = shape.Uniform(0.428, 0.642)
    
    # with primary treatment
    if TT in ['B1','B1E','B2','B3','B4','B5','B6',
              'D1','D1E','D2','D3','D5','D6','E2P',
              'F1','F1E','G1','G1E','G2','G3','G5',
              'G6','H1','H1E','N1','N1E','N2','O1',
              'O1E','O2','O3','O5','O6']:
        sludge_yield_MC = shape.Uniform(0.2636*0.9, 0.2636*1.1)
    # without primary treatment
    else: 
        sludge_yield_MC = shape.Uniform(0.131*0.9, 0.131*1.1)
    
    TS_2_VS_MC = shape.Uniform(0.4, 0.8)
    
    # with AD
    if TT in ['B1','B1E','B4','C1','C1E','D1','D1E',
              'F1','F1E','G1','G1E','H1','H1E','I1',
              'I1E','LAGOON_ANAER','LAGOON_FAC','N1',
              'N1E','O1','O1E']:
        biosolids_reduction_MC = shape.Uniform(0.35, 0.5)
    # with AeD
    elif TT in ['B2','C2','D2','E2','E2P','G2','I2',
                'LAGOON_AER','N2','O2']:
        biosolids_reduction_MC = shape.Uniform(0.4, 0.55)
    elif TT == 'LAGOON_UNCATEGORIZED':
        biosolids_reduction_MC = shape.Uniform(0.4*L_AE_flow_ratio + 0.35*L_AN_FA_flow_ratio,
                                               0.55*L_AE_flow_ratio + 0.5*L_AN_FA_flow_ratio)
    else:
        biosolids_reduction_MC = 0
    
    # 1st year CH4 emission from landfill using EPA's LandGEM model
    # baseline: 5.65 kg CH4 per tonne biosolids converted to per kg biosolids
    LF_CH4_EF_MC = shape.Uniform(5.65*0.9/1000, 5.65*1.1/1000)
    # N2O from land application
    # baseline: 0.049 kg N per kg biosolids land applied
    LA_solids_N_MC = shape.Triangle(0.0122, 0.049, 0.062)
    # baseline: 0.01 kg N2O-N/kg N
    LA_N2O_N_EF_MC = shape.Uniform(0.002, 0.018)
    
    result_MC = pd.DataFrame()
    
    # TTs that are not lagoons and use incineration
    if TT[0] != 'L' and TT[1] in ['5','6']:
        try:
            joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                       NG_MC, NG_combustion_CI_MC, NG_upstream_CI_MC,
                                       COD_MC, CH4_EF_MC, CH4_CF_MC,
                                       TN_MC, N2O_EF_MC, N2O_CF_MC,
                                       fossil_COD_MC, assimilated_COD_MC)
            sample = joint.sample(10000)
            
            result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
            result_MC['NG'] = sample[2]*(sample[3] + sample[4])/MG_2_m3
            result_MC['NG_combustion'] = sample[2]*sample[3]/MG_2_m3
            result_MC['NG_upstream'] = sample[2]*sample[4]/MG_2_m3
            result_MC['CH4'] = sample[6]*sample[7]
            result_MC['N2O'] = sample[8]*sample[9]*N_2_N2O*sample[10]
            result_MC['NC_CO2'] = sample[5]*sample[11]*(1 - sample[12])
            result_MC['solids'] = 0
            result_MC['solids_LF'] = 0
            result_MC['solids_LA'] = 0
            result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
            
            i = 0
            spearman_p_result = pd.DataFrame()
            spearman_rho_result = pd.DataFrame()
            all_distributions = ['elec_MC','elec_CI_MC',
                                 'NG_MC','NG_combustion_CI_MC','NG_upstream_CI_MC',
                                 'COD_MC','CH4_EF_MC','CH4_CF_MC',
                                 'TN_MC','N2O_EF_MC','N2O_CF_MC',
                                 'fossil_COD_MC','assimilated_COD_MC']
            assert len(sample) == len(all_distributions)
            for item in all_distributions:
                result_MC[item] = sample[i]
                spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
                spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
                i += 1
        except AssertionError:
            joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                       NG_combustion_CI_MC, NG_upstream_CI_MC,
                                       COD_MC, CH4_EF_MC, CH4_CF_MC,
                                       TN_MC, N2O_EF_MC, N2O_CF_MC,
                                       fossil_COD_MC, assimilated_COD_MC)
            sample = joint.sample(10000)
            
            result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
            result_MC['NG'] = NG_MC*(sample[2] + sample[3])/MG_2_m3
            result_MC['NG_combustion'] = NG_MC*sample[2]/MG_2_m3
            result_MC['NG_upstream'] = NG_MC*sample[3]/MG_2_m3
            result_MC['CH4'] = sample[5]*sample[6]
            result_MC['N2O'] = sample[7]*sample[8]*N_2_N2O*sample[9]
            result_MC['NC_CO2'] = sample[4]*sample[10]*(1 - sample[11])
            result_MC['solids'] = 0
            result_MC['solids_LF'] = 0
            result_MC['solids_LA'] = 0
            result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
            
            i = 0
            spearman_p_result = pd.DataFrame()
            spearman_rho_result = pd.DataFrame()
            all_distributions = ['elec_MC','elec_CI_MC',
                                 'NG_combustion_CI_MC','NG_upstream_CI_MC',
                                 'COD_MC','CH4_EF_MC','CH4_CF_MC',
                                 'TN_MC','N2O_EF_MC','N2O_CF_MC',
                                 'fossil_COD_MC','assimilated_COD_MC']
            assert len(sample) == len(all_distributions)
            for item in all_distributions:
                result_MC[item] = sample[i]
                spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
                spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
                i += 1
    elif TT in ['B3','C3','D3','G3','I3','O3']:
        try:
            joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                       NG_MC, NG_combustion_CI_MC, NG_upstream_CI_MC,
                                       COD_MC, CH4_EF_MC, CH4_CF_MC,
                                       TN_MC, N2O_EF_MC, N2O_CF_MC,
                                       fossil_COD_MC, assimilated_COD_MC,
                                       sludge_yield_MC,
                                       LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)
            sample = joint.sample(10000)
            
            result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
            result_MC['NG'] = sample[2]*(sample[3] + sample[4])/MG_2_m3
            result_MC['NG_combustion'] = sample[2]*sample[3]/MG_2_m3
            result_MC['NG_upstream'] = sample[2]*sample[4]/MG_2_m3
            result_MC['CH4'] = sample[6]*sample[7]
            result_MC['N2O'] = sample[8]*sample[9]*N_2_N2O*sample[10]
            result_MC['NC_CO2'] = sample[5]*sample[11]*(1 - sample[12])
            result_MC['solids'] = sample[13]*ratio_LF_LF_LA*sample[14]*sample[7] +\
                                  sample[13]*ratio_LA_LF_LA*sample[15]*sample[16]*N_2_N2O*sample[10]
            result_MC['solids_LF'] = sample[13]*ratio_LF_LF_LA*sample[14]*sample[7]
            result_MC['solids_LA'] = sample[13]*ratio_LA_LF_LA*sample[15]*sample[16]*N_2_N2O*sample[10]
            result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
            
            i = 0
            spearman_p_result = pd.DataFrame()
            spearman_rho_result = pd.DataFrame()
            all_distributions = ['elec_MC','elec_CI_MC',
                                 'NG_MC','NG_combustion_CI_MC','NG_upstream_CI_MC',
                                 'COD_MC','CH4_EF_MC','CH4_CF_MC',
                                 'TN_MC','N2O_EF_MC','N2O_CF_MC',
                                 'fossil_COD_MC','assimilated_COD_MC',
                                 'sludge_yield_MC',
                                 'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
            assert len(sample) == len(all_distributions)
            for item in all_distributions:
                result_MC[item] = sample[i]
                spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
                spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
                i += 1
        except AssertionError:
            joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                       NG_combustion_CI_MC, NG_upstream_CI_MC,
                                       COD_MC, CH4_EF_MC, CH4_CF_MC,
                                       TN_MC, N2O_EF_MC, N2O_CF_MC,
                                       fossil_COD_MC, assimilated_COD_MC,
                                       sludge_yield_MC,
                                       LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)
            sample = joint.sample(10000)
            
            result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
            result_MC['NG'] = NG_MC*(sample[2] + sample[3])/MG_2_m3
            result_MC['NG_combustion'] = NG_MC*sample[2]/MG_2_m3
            result_MC['NG_upstream'] = NG_MC*sample[3]/MG_2_m3
            result_MC['CH4'] = sample[5]*sample[6]
            result_MC['N2O'] = sample[7]*sample[8]*N_2_N2O*sample[9]
            result_MC['NC_CO2'] = sample[4]*sample[10]*(1 - sample[11])
            result_MC['solids'] = sample[12]*ratio_LF_LF_LA*sample[13]*sample[6] +\
                                  sample[12]*ratio_LA_LF_LA*sample[14]*sample[15]*N_2_N2O*sample[9]
            result_MC['solids_LF'] = sample[12]*ratio_LF_LF_LA*sample[13]*sample[6]
            result_MC['solids_LA'] = sample[12]*ratio_LA_LF_LA*sample[14]*sample[15]*N_2_N2O*sample[9]
            result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
            
            i = 0
            spearman_p_result = pd.DataFrame()
            spearman_rho_result = pd.DataFrame()
            all_distributions = ['elec_MC','elec_CI_MC',
                                 'NG_combustion_CI_MC','NG_upstream_CI_MC',
                                 'COD_MC','CH4_EF_MC','CH4_CF_MC',
                                 'TN_MC','N2O_EF_MC','N2O_CF_MC',
                                 'fossil_COD_MC','assimilated_COD_MC',
                                 'sludge_yield_MC',
                                 'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
            assert len(sample) == len(all_distributions)
            for item in all_distributions:
                result_MC[item] = sample[i]
                spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
                spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
                i += 1
    # TTs that are not lagoons and do not use incineration
    elif TT[0] != 'L' and TT[1] not in ['5','6']:
        try:
            joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                       NG_MC, NG_combustion_CI_MC, NG_upstream_CI_MC,
                                       COD_MC, CH4_EF_MC, CH4_CF_MC,
                                       TN_MC, N2O_EF_MC, N2O_CF_MC,
                                       fossil_COD_MC, assimilated_COD_MC,
                                       sludge_yield_MC, TS_2_VS_MC, biosolids_reduction_MC,
                                       LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)
            sample = joint.sample(10000)
            
            result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
            result_MC['NG'] = sample[2]*(sample[3] + sample[4])/MG_2_m3
            result_MC['NG_combustion'] = sample[2]*sample[3]/MG_2_m3
            result_MC['NG_upstream'] = sample[2]*sample[4]/MG_2_m3
            result_MC['CH4'] = sample[6]*sample[7]
            result_MC['N2O'] = sample[8]*sample[9]*N_2_N2O*sample[10]
            result_MC['NC_CO2'] = sample[5]*sample[11]*(1 - sample[12])
            result_MC['solids'] = sample[13]*(1-sample[14]*sample[15])*ratio_LF_LF_LA*sample[16]*sample[7] +\
                                  sample[13]*(1-sample[14]*sample[15])*ratio_LA_LF_LA*sample[17]*sample[18]*N_2_N2O*sample[10]
            result_MC['solids_LF'] = sample[13]*(1-sample[14]*sample[15])*ratio_LF_LF_LA*sample[16]*sample[7]
            result_MC['solids_LA'] = sample[13]*(1-sample[14]*sample[15])*ratio_LA_LF_LA*sample[17]*sample[18]*N_2_N2O*sample[10]
            result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
            
            i = 0
            spearman_p_result = pd.DataFrame()
            spearman_rho_result = pd.DataFrame()
            all_distributions = ['elec_MC','elec_CI_MC',
                                 'NG_MC','NG_combustion_CI_MC','NG_upstream_CI_MC',
                                 'COD_MC','CH4_EF_MC','CH4_CF_MC',
                                 'TN_MC','N2O_EF_MC','N2O_CF_MC',
                                 'fossil_COD_MC','assimilated_COD_MC',
                                 'sludge_yield_MC','TS_2_VS_MC','biosolids_reduction_MC',
                                 'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
            assert len(sample) == len(all_distributions)
            for item in all_distributions:
                result_MC[item] = sample[i]
                spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
                spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
                i += 1
        except AssertionError:
            joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                       NG_combustion_CI_MC, NG_upstream_CI_MC,
                                       COD_MC, CH4_EF_MC, CH4_CF_MC,
                                       TN_MC, N2O_EF_MC, N2O_CF_MC,
                                       fossil_COD_MC, assimilated_COD_MC,
                                       sludge_yield_MC, TS_2_VS_MC, biosolids_reduction_MC,
                                       LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)
            sample = joint.sample(10000)
            
            result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
            result_MC['NG'] = NG_MC*(sample[2] + sample[3])/MG_2_m3
            result_MC['NG_combustion'] = NG_MC*sample[2]/MG_2_m3
            result_MC['NG_upstream'] = NG_MC*sample[3]/MG_2_m3
            result_MC['CH4'] = sample[5]*sample[6]
            result_MC['N2O'] = sample[7]*sample[8]*N_2_N2O*sample[9]
            result_MC['NC_CO2'] = sample[4]*sample[10]*(1 - sample[11])
            result_MC['solids'] = sample[12]*(1-sample[13]*sample[14])*ratio_LF_LF_LA*sample[15]*sample[6] +\
                                  sample[12]*(1-sample[13]*sample[14])*ratio_LA_LF_LA*sample[16]*sample[17]*N_2_N2O*sample[9]
            result_MC['solids_LF'] = sample[12]*(1-sample[13]*sample[14])*ratio_LF_LF_LA*sample[15]*sample[6]
            result_MC['solids_LA'] = sample[12]*(1-sample[13]*sample[14])*ratio_LA_LF_LA*sample[16]*sample[17]*N_2_N2O*sample[9]
            result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
            
            i = 0
            spearman_p_result = pd.DataFrame()
            spearman_rho_result = pd.DataFrame()
            all_distributions = ['elec_MC','elec_CI_MC',
                                 'NG_combustion_CI_MC','NG_upstream_CI_MC',
                                 'COD_MC','CH4_EF_MC','CH4_CF_MC',
                                 'TN_MC','N2O_EF_MC','N2O_CF_MC',
                                 'fossil_COD_MC','assimilated_COD_MC',
                                 'sludge_yield_MC','TS_2_VS_MC','biosolids_reduction_MC',
                                 'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
            assert len(sample) == len(all_distributions)
            for item in all_distributions:
                result_MC[item] = sample[i]
                spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
                spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
                i += 1
    # no NG is used in lagoons
    elif TT == 'LAGOON_AER':
        joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                   COD_MC, CH4_CF_MC,
                                   TN_MC, N2O_EF_MC, N2O_CF_MC,
                                   fossil_COD_MC, assimilated_COD_MC,
                                   sludge_yield_MC, TS_2_VS_MC, biosolids_reduction_MC,
                                   LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)
        sample = joint.sample(10000)
        
        result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
        result_MC['NG'] = 0
        result_MC['NG_combustion'] = 0
        result_MC['NG_upstream'] = 0
        result_MC['CH4'] = 0
        result_MC['N2O'] = sample[4]*sample[5]*N_2_N2O*sample[6]
        result_MC['NC_CO2'] = sample[2]*sample[7]*(1 - sample[8])
        result_MC['solids'] = sample[9]*(1-sample[10]*sample[11])*ratio_LF_LF_LA*sample[12]*sample[3] +\
                              sample[9]*(1-sample[10]*sample[11])*ratio_LA_LF_LA*sample[13]*sample[14]*N_2_N2O*sample[6]
        result_MC['solids_LF'] = sample[9]*(1-sample[10]*sample[11])*ratio_LF_LF_LA*sample[12]*sample[3]
        result_MC['solids_LA'] = sample[9]*(1-sample[10]*sample[11])*ratio_LA_LF_LA*sample[13]*sample[14]*N_2_N2O*sample[6]
        result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
        
        i = 0
        spearman_p_result = pd.DataFrame()
        spearman_rho_result = pd.DataFrame()
        all_distributions = ['elec_MC','elec_CI_MC',
                             'COD_MC','CH4_CF_MC',
                             'TN_MC','N2O_EF_MC','N2O_CF_MC',
                             'fossil_COD_MC','assimilated_COD_MC',
                             'sludge_yield_MC','TS_2_VS_MC','biosolids_reduction_MC',
                             'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
        assert len(sample) == len(all_distributions)
        for item in all_distributions:
            result_MC[item] = sample[i]
            spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
            spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
            i += 1
    elif TT in ['LAGOON_ANAER','LAGOON_FAC']:
        joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                   COD_MC, CH4_COD_EF_MC, CH4_CF_MC,
                                   N2O_CF_MC,
                                   fossil_COD_MC, assimilated_COD_MC,
                                   sludge_yield_MC, TS_2_VS_MC, biosolids_reduction_MC,
                                   LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)  
        sample = joint.sample(10000)
        
        result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
        result_MC['NG'] = 0
        result_MC['NG_combustion'] = 0
        result_MC['NG_upstream'] = 0
        result_MC['CH4'] = sample[2]*sample[3]*sample[4]
        result_MC['N2O'] = 0
        result_MC['NC_CO2'] = sample[2]*sample[6]*(1 - sample[7])
        result_MC['solids'] = sample[8]*(1-sample[9]*sample[10])*ratio_LF_LF_LA*sample[11]*sample[4] +\
                              sample[8]*(1-sample[9]*sample[10])*ratio_LA_LF_LA*sample[12]*sample[13]*N_2_N2O*sample[5]
        result_MC['solids_LF'] = sample[8]*(1-sample[9]*sample[10])*ratio_LF_LF_LA*sample[11]*sample[4]
        result_MC['solids_LA'] = sample[8]*(1-sample[9]*sample[10])*ratio_LA_LF_LA*sample[12]*sample[13]*N_2_N2O*sample[5]
        result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)
        
        i = 0
        spearman_p_result = pd.DataFrame()
        spearman_rho_result = pd.DataFrame()
        all_distributions = ['elec_MC','elec_CI_MC',
                             'COD_MC','CH4_EF_MC','CH4_CF_MC',
                             'N2O_CF_MC',
                             'fossil_COD_MC','assimilated_COD_MC',
                             'sludge_yield_MC','TS_2_VS_MC','biosolids_reduction_MC',
                             'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
        assert len(sample) == len(all_distributions)
        for item in all_distributions:
            result_MC[item] = sample[i]
            spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
            spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
            i += 1
    elif TT == 'LAGOON_UNCATEGORIZED':
        joint = cp.distributions.J(elec_MC, elec_CI_MC,
                                   COD_MC, CH4_COD_EF_MC, CH4_CF_MC,
                                   TN_MC, N2O_EF_MC, N2O_CF_MC,
                                   fossil_COD_MC, assimilated_COD_MC,
                                   sludge_yield_MC, TS_2_VS_MC, biosolids_reduction_MC,
                                   LF_CH4_EF_MC, LA_solids_N_MC, LA_N2O_N_EF_MC)  
        sample = joint.sample(10000)

        result_MC['elec'] = sample[0]*sample[1]/MG_2_m3
        result_MC['NG'] = 0
        result_MC['NG_combustion'] = 0
        result_MC['NG_upstream'] = 0
        result_MC['CH4'] = sample[2]*sample[3]*sample[4]
        result_MC['N2O'] = sample[5]*sample[6]*N_2_N2O*sample[7]
        result_MC['NC_CO2'] = sample[2]*sample[8]*(1 - sample[9])
        result_MC['solids'] = sample[10]*(1-sample[11]*sample[12])*ratio_LF_LF_LA*sample[13]*sample[4] +\
                              sample[10]*(1-sample[11]*sample[12])*ratio_LA_LF_LA*sample[14]*sample[15]*N_2_N2O*sample[7]
        result_MC['solids_LF'] = sample[10]*(1-sample[11]*sample[12])*ratio_LF_LF_LA*sample[13]*sample[4]
        result_MC['solids_LA'] = sample[10]*(1-sample[11]*sample[12])*ratio_LA_LF_LA*sample[14]*sample[15]*N_2_N2O*sample[7]
        result_MC['total'] = result_MC[['elec','NG','CH4','N2O','NC_CO2','solids']].sum(axis=1)

        i = 0
        spearman_p_result = pd.DataFrame()
        spearman_rho_result = pd.DataFrame()
        all_distributions = ['elec_MC','elec_CI_MC',
                             'COD_MC','CH4_EF_MC','CH4_CF_MC',
                             'TN_MC','N2O_EF_MC','N2O_CF_MC',
                             'fossil_COD_MC','assimilated_COD_MC',
                             'sludge_yield_MC','TS_2_VS_MC','biosolids_reduction_MC',
                             'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']
        assert len(sample) == len(all_distributions)
        for item in all_distributions:
            result_MC[item] = sample[i]
            spearman_p_result[item] = [stats.spearmanr(sample[i], result_MC['total']).pvalue]
            spearman_rho_result[item] = [stats.spearmanr(sample[i], result_MC['total']).statistic]
            i += 1
    else:
        raise ValueError(f'{TT} does not exist.')
    
    # uncomment the following lines for saving using the naming convention in the WERF report (Tarallo et al., 2015), used for the remaining codes
    # result_MC.to_excel(f'MC/{TT}_MC.xlsx')
    # spearman_p_result.to_excel(f'spearman_p/{TT}_p.xlsx')
    # spearman_rho_result.to_excel(f'spearman_rho/{TT}_rho.xlsx')

#%% rename saved files

# uncomment the following lines to rename files, used for reporting
# for TT in final_code:
#     MC = pd.read_excel(f'MC/{TT}_MC.xlsx')
#     p = pd.read_excel(f'spearman_p/{TT}_p.xlsx')
#     rho = pd.read_excel(f'spearman_rho/{TT}_rho.xlsx')
    
#     MC.to_excel(f'MC_renamed/{crosswalk[TT]}_MC.xlsx')
#     p.to_excel(f'spearman_p_renamed/{crosswalk[TT]}_p.xlsx')
#     rho.to_excel(f'spearman_rho_renamed/{crosswalk[TT]}_rho.xlsx')

#%% import uncertainty and sensitivity results

CH4_5 = []
N2O_5 = []
NC_CO2_5 = []
elec_5 = []
NG_5 = []
NG_combustion_5 = []
NG_upstream_5 = []
solids_5 = []
solids_LF_5 = []
solids_LA_5 = []

CH4_50 = []
N2O_50 = []
NC_CO2_50 = []
elec_50 = []
NG_50 = []
NG_combustion_50 = []
NG_upstream_50 = []
solids_50 = []
solids_LF_50 = []
solids_LA_50 = []

CH4_95 = []
N2O_95 = []
NC_CO2_95 = []
elec_95 = []
NG_95 = []
NG_combustion_95 = []
NG_upstream_95 = []
solids_95 = []
solids_LF_95 = []
solids_LA_95 = []

total_MC = pd.DataFrame()
spearman_p = pd.DataFrame()
spearman_rho = pd.DataFrame()

TT_elec_uncertainty = pd.DataFrame()
TT_elec_chemical_uncertainty = pd.DataFrame()
TT_NG_uncertainty = pd.DataFrame()
TT_NG_chemical_uncertainty = pd.DataFrame()

for TT in final_code:
    breakdown_data_MC = pd.read_excel(f'MC/{TT}_MC.xlsx')
    
    CH4_5.append(breakdown_data_MC['CH4'].quantile(0.05))
    N2O_5.append(breakdown_data_MC['N2O'].quantile(0.05))
    NC_CO2_5.append(breakdown_data_MC['NC_CO2'].quantile(0.05))
    elec_5.append(breakdown_data_MC['elec'].quantile(0.05))
    NG_5.append(breakdown_data_MC['NG'].quantile(0.05))
    NG_combustion_5.append(breakdown_data_MC['NG_combustion'].quantile(0.05))
    NG_upstream_5.append(breakdown_data_MC['NG_upstream'].quantile(0.05))
    solids_5.append(breakdown_data_MC['solids'].quantile(0.05))
    solids_LF_5.append(breakdown_data_MC['solids_LF'].quantile(0.05))
    solids_LA_5.append(breakdown_data_MC['solids_LA'].quantile(0.05))
    
    CH4_50.append(breakdown_data_MC['CH4'].quantile(0.5))
    N2O_50.append(breakdown_data_MC['N2O'].quantile(0.5))
    NC_CO2_50.append(breakdown_data_MC['NC_CO2'].quantile(0.5))
    elec_50.append(breakdown_data_MC['elec'].quantile(0.5))
    NG_50.append(breakdown_data_MC['NG'].quantile(0.5))
    NG_combustion_50.append(breakdown_data_MC['NG_combustion'].quantile(0.5))
    NG_upstream_50.append(breakdown_data_MC['NG_upstream'].quantile(0.5))
    solids_50.append(breakdown_data_MC['solids'].quantile(0.5))
    solids_LF_50.append(breakdown_data_MC['solids_LF'].quantile(0.5))
    solids_LA_50.append(breakdown_data_MC['solids_LA'].quantile(0.5))
    
    CH4_95.append(breakdown_data_MC['CH4'].quantile(0.95))
    N2O_95.append(breakdown_data_MC['N2O'].quantile(0.95))
    NC_CO2_95.append(breakdown_data_MC['NC_CO2'].quantile(0.95))
    elec_95.append(breakdown_data_MC['elec'].quantile(0.95))
    NG_95.append(breakdown_data_MC['NG'].quantile(0.95))
    NG_combustion_95.append(breakdown_data_MC['NG_combustion'].quantile(0.95))
    NG_upstream_95.append(breakdown_data_MC['NG_upstream'].quantile(0.95))
    solids_95.append(breakdown_data_MC['solids'].quantile(0.95))
    solids_LF_95.append(breakdown_data_MC['solids_LF'].quantile(0.95))
    solids_LA_95.append(breakdown_data_MC['solids_LA'].quantile(0.95))
    
    total_MC[TT] = breakdown_data_MC['total']
    
    data_p = pd.read_excel(f'spearman_p/{TT}_p.xlsx')
    spearman_p = pd.concat([spearman_p, data_p], axis=0)
    
    data_rho = pd.read_excel(f'spearman_rho/{TT}_rho.xlsx')
    spearman_rho = pd.concat([spearman_rho, data_rho], axis=0)
    
    TT_elec_uncertainty[TT] = breakdown_data_MC['elec_MC']
    
    # no electricity used for chemicals for lagoons
    if TT[0] == 'L':
        TT_elec_chemical_uncertainty[TT] = [0]*10000
    else:
        elec_chemical_lower = min(energy_uncertainty[TT]['best_elec_usage_chemical'],
                                  energy_uncertainty[TT]['typical_elec_usage_chemical'])/10
        elec_chemical_baseline = max(energy_uncertainty[TT]['best_elec_usage_chemical'],
                                     energy_uncertainty[TT]['typical_elec_usage_chemical'])/10
        elec_chemical_upper = 2*elec_chemical_baseline - elec_chemical_lower
        elec_chemical_MC = shape.Uniform(elec_chemical_lower, elec_chemical_upper)
        
        try:
            TT_elec_chemical_uncertainty[TT] = elec_chemical_MC.sample(10000)
        except AssertionError:
            if elec_chemical_lower == elec_chemical_upper:
                TT_elec_chemical_uncertainty[TT] = [elec_chemical_baseline]*10000
    
    # no NG for lagoons
    if TT[0] == 'L':
        TT_NG_uncertainty[TT] = [0]*10000
        TT_NG_chemical_uncertainty[TT] = [0]*10000
    else:
        try:
            TT_NG_uncertainty[TT] = breakdown_data_MC['NG_MC']
        except KeyError:
            TT_NG_uncertainty[TT] = [energy_uncertainty[TT]['typical_NG_usage']/10]*10000
        
        NG_chemical_lower = min(energy_uncertainty[TT]['best_NG_usage_chemical'],
                                energy_uncertainty[TT]['typical_NG_usage_chemical'])/10
        NG_chemical_baseline = max(energy_uncertainty[TT]['best_NG_usage_chemical'],
                                   energy_uncertainty[TT]['typical_NG_usage_chemical'])/10
        NG_chemical_upper = 2*NG_chemical_baseline - NG_chemical_lower
        NG_chemical_MC = shape.Uniform(NG_chemical_lower, NG_chemical_upper)
        
        try:
            TT_NG_chemical_uncertainty[TT] = NG_chemical_MC.sample(10000)
        except AssertionError:
            if NG_chemical_lower == NG_chemical_upper:
                TT_NG_chemical_uncertainty[TT] = [NG_chemical_baseline]*10000

m3_to_plot = pd.DataFrame()
m3_to_plot['index'] = final_code
m3_to_plot['CH4_5'] = CH4_5
m3_to_plot['N2O_5'] = N2O_5
m3_to_plot['NC_CO2_5'] = NC_CO2_5
m3_to_plot['elec_5'] = elec_5
m3_to_plot['NG_5'] = NG_5
m3_to_plot['NG_combustion_5'] = NG_combustion_5
m3_to_plot['NG_upstream_5'] = NG_upstream_5
m3_to_plot['solids_5'] = solids_5
m3_to_plot['solids_LF_5'] = solids_LF_5
m3_to_plot['solids_LA_5'] = solids_LA_5
m3_to_plot['CH4_50'] = CH4_50
m3_to_plot['N2O_50'] = N2O_50
m3_to_plot['NC_CO2_50'] = NC_CO2_50
m3_to_plot['elec_50'] = elec_50
m3_to_plot['NG_50'] = NG_50
m3_to_plot['NG_combustion_50'] = NG_combustion_50
m3_to_plot['NG_upstream_50'] = NG_upstream_50
m3_to_plot['solids_50'] = solids_50
m3_to_plot['solids_LF_50'] = solids_LF_50
m3_to_plot['solids_LA_50'] = solids_LA_50
m3_to_plot['CH4_95'] = CH4_95
m3_to_plot['N2O_95'] = N2O_95
m3_to_plot['NC_CO2_95'] = NC_CO2_95
m3_to_plot['elec_95'] = elec_95
m3_to_plot['NG_95'] = NG_95
m3_to_plot['NG_combustion_95'] = NG_combustion_95
m3_to_plot['NG_upstream_95'] = NG_upstream_95
m3_to_plot['solids_95'] = solids_95
m3_to_plot['solids_LF_95'] = solids_LF_95
m3_to_plot['solids_LA_95'] = solids_LA_95
m3_to_plot.set_index('index', inplace=True)

WWTP_EF = m3_to_plot.copy()

data_order = [i for i in m3_to_plot.index]
updated_label_order = [crosswalk[i] for i in data_order]
m3_to_plot.reset_index(inplace=True)
m3_to_plot['new_TT'] = updated_label_order
m3_to_plot.set_index('new_TT', inplace=True)
m3_to_plot.drop('index', axis=1, inplace=True)

# distribute the flow rate to TTs for each WWTP
WWTP_TT_all = WWTP_TT.loc[:, final_code]
WWTP_TT_all = WWTP_TT_all.div(WWTP_TT['TT_IDENTIFIED'], axis=0)
WWTP_TT_all = WWTP_TT_all.mul(WWTP_TT['FLOW_2022_MGD_FINAL'], axis=0)

TT_flow = WWTP_TT_all.sum(axis=0)
TT_flow = TT_flow.loc[data_order]
TT_flow = TT_flow.set_axis(updated_label_order, axis=0)

TT_num = WWTP_TT[final_code].sum(axis=0)
TT_num = TT_num.loc[data_order]
TT_num = TT_num.set_axis(updated_label_order, axis=0)

total_MC.rename(columns=crosswalk, inplace=True)

# combine TTs with the only difference being the primary treatment since the difference between their median total per volume emissions are less than 5%
for TT in ['A1','A1e','A3','A4','A5','A6','E3']:
    assert (total_MC.median()['*'+TT] - total_MC.median()[TT])/total_MC.median()[TT] < 0.05

for TT in ['A1','A1e','A3','A4','A5','A6','E3']:
    total_MC['[*]'+TT] = (TT_flow[TT]*total_MC.sort_values(by=TT)[TT] +\
                          TT_flow['*'+TT]*total_MC.sort_values(by=TT)['*'+TT])/(TT_flow[TT] + TT_flow['*'+TT])
    
    m3_to_plot.loc['[*]'+TT] = (TT_flow[TT]*m3_to_plot.loc[TT] +\
                                TT_flow['*'+TT]*m3_to_plot.loc['*'+TT])/(TT_flow[TT] + TT_flow['*'+TT])

total_MC.drop(columns=['A1','*A1','A1e','*A1e','A3','*A3','A4',
                       '*A4','A5','*A5','A6','*A6','E3','*E3'], inplace=True)

m3_to_plot.drop(index=['A1','*A1','A1e','*A1e','A3','*A3','A4',
                       '*A4','A5','*A5','A6','*A6','E3','*E3'], inplace=True)

label_order = total_MC.median().sort_values().index

total_MC = total_MC[label_order]

m3_to_plot = m3_to_plot.loc[label_order]

per_m3_result = m3_to_plot.copy()

spearman_p.drop('Unnamed: 0', axis=1, inplace=True)
spearman_p['index'] = final_code
spearman_p.set_index('index', inplace=True)

spearman_rho.drop('Unnamed: 0', axis=1, inplace=True)
spearman_rho['index'] = final_code
spearman_rho.set_index('index', inplace=True)

#%% grid - data preparation

TT_characteristics = pd.DataFrame()
TT_characteristics.index = label_order
TT_characteristics.loc[TT_characteristics.index.str.contains('1|2', case=True), 'AD'] = 1
TT_characteristics.loc[TT_characteristics.index.str.contains('5|6', case=True), 'incineration'] = 1
TT_characteristics.loc[TT_characteristics.index.str.contains('D|E|F|G', case=True), 'N removal'] = 1
TT_characteristics.loc[TT_characteristics.index.str.contains('D|G', case=True), 'P removal'] = 1
TT_characteristics.loc[TT_characteristics.index.str.contains('e', case=True), 'CHP'] = 1

#%% grid - visualization

TT_characteristics_reversed = TT_characteristics[::-1]

fig, ax = plt.subplots(figsize=(32, 5))

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

ax = plt.gca()
ax.set_xlim([0.15, len(label_order)+0.85])
ax.set_ylim([-0.1, 0.9])

ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0, pad=5)
ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

ax.grid('on', color=a, alpha=0.5, linewidth=1.5)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('k')
ax.spines['right'].set_color('k')

plt.xticks(np.arange(1, len(label_order)+1, 1), label_order[::-1], rotation=90, fontname='Arial')
plt.yticks(np.arange(0, 1, 0.2), ['AD','incineration','N removal','P removal','CHP'], fontname='Arial')

for i in range(len(TT_characteristics_reversed.index)):
    for j in range(len(TT_characteristics_reversed.columns)):
        if TT_characteristics_reversed.iloc[i, j] == 1:
            ax.scatter(x=i+1,
                       y=j*0.2,
                       marker='o',
                       s=250,
                       c='k',
                       linewidths=1.5,
                       alpha=1,
                       edgecolor='k',
                       zorder=3)

#%% per volume uncertainty with breakdown

fig, ax = plt.subplots(figsize=(32, 7))

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

ax = plt.gca()
ax.set_xlim([0.15, len(label_order)+0.85])
ax.set_ylim([0, 3])

ax.tick_params(direction='out', length=7.5, width=1.5,
               bottom=True, top=False, left=True, right=False, pad=0)

plt.xticks(np.arange(1, len(label_order)+1, 1), label_order[::-1], rotation=90, fontname='Arial')
plt.yticks(np.arange(0, 4, 1), fontname='Arial')

ax_left = ax.twinx()
ax_left.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 4, 1), fontname='Arial')

ax_left.tick_params(direction='inout', length=15, width=1.5,
                    bottom=False, top=False, left=True, right=False,
                    labelcolor='none')

ax_right = ax.twinx()
ax_right.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 4, 1), fontname='Arial')

ax_right.tick_params(direction='in', length=7.5, width=1.5,
                     bottom=False, top=False, left=False, right=True,
                     labelcolor='none')

ax_right.tick_params(direction='in', length=7.5, width=1.5,
                     bottom=False, top=False, left=False, right=True, pad=0)

ax.set_ylabel('$\mathbf{Per\ volume\ GHG}$\n[kg ${CO_2}$-eq·${m^{-3}}$]',
              fontname='Arial',
              fontsize=28,
              labelpad=13,
              linespacing=0.8)

mathtext.FontConstantsBase.sup1 = 0.35

index = np.arange(1, len(label_order)+1, 1)

width = 0.4

bp = ax.boxplot(total_MC[total_MC.columns[::-1]],
                positions=index-0.5*width,
                widths=width,
                whis=[5, 95],
                showfliers=False, vert=True,
                boxprops = dict(linestyle='-', linewidth=1.5, color='k'),
                medianprops = dict(linestyle='-', linewidth=1.5, color='k'),
                whiskerprops=dict(linestyle='-', linewidth=1.5),
                capprops=dict(linestyle='-', linewidth=1.5))

plt.xticks(np.arange(1, len(label_order)+1, 1), label_order[::-1], rotation=90, fontname='Arial')

ax.scatter(x=index-0.5*width,
           y=total_MC[total_MC.columns[::-1]].mean(),
           marker='D',
           s=50,
           c='w',
           linewidths=1.5,
           edgecolors='k',
           alpha=1,
           zorder=3)

# onsite CH4, N2O, CO2
ax.bar(index+0.5*width,
       m3_to_plot[::-1]['CH4_50'],
       width=width,
       color=dr,
       edgecolor='k',
       linewidth=1.5)

ax.bar(index+0.5*width,
       m3_to_plot[::-1]['N2O_50'],
       width=width,
       color=r,
       edgecolor='k',
       linewidth=1.5,
       bottom=m3_to_plot[::-1]['CH4_50'])

ax.bar(index+0.5*width,
       m3_to_plot[::-1]['NC_CO2_50'],
       width=width,
       color=r,
       edgecolor='k',
       linewidth=0,
       alpha=0.5,
       bottom=m3_to_plot[::-1]['CH4_50']+m3_to_plot[::-1]['N2O_50'])

ax.bar(index+0.5*width,
       m3_to_plot[::-1]['NC_CO2_50'],
       width=width,
       color='none',
       edgecolor='k',
       linewidth=1.5,
       alpha=1,
       bottom=m3_to_plot[::-1]['CH4_50']+m3_to_plot[::-1]['N2O_50'])

# electricity
ax.bar(index+0.5*width,
       m3_to_plot[::-1]['elec_50'],
       width=width,
       color=y,
       edgecolor='k',
       linewidth=1.5,
       bottom=m3_to_plot[::-1]['CH4_50']+m3_to_plot[::-1]['N2O_50']+m3_to_plot[::-1]['NC_CO2_50'])

# NG
ax.bar(index+0.5*width,
       m3_to_plot[::-1]['NG_50'],
       width=width,
       color=b,
       edgecolor='k',
       linewidth=1.5,
       bottom=m3_to_plot[::-1]['CH4_50']+m3_to_plot[::-1]['N2O_50']+m3_to_plot[::-1]['NC_CO2_50']+m3_to_plot[::-1]['elec_50'])

#%% sensitivity - data preparation

spearman_order = ['COD_MC','TN_MC',
                  'CH4_EF_MC','CH4_CF_MC',
                  'N2O_EF_MC','N2O_CF_MC',
                  'fossil_COD_MC','assimilated_COD_MC',
                  'elec_MC','elec_CI_MC',
                  'NG_MC','NG_combustion_CI_MC','NG_upstream_CI_MC',
                  'sludge_yield_MC','TS_2_VS_MC','biosolids_reduction_MC',
                  'LF_CH4_EF_MC','LA_solids_N_MC','LA_N2O_N_EF_MC']

spearman_p = spearman_p[spearman_order]
spearman_rho = spearman_rho[spearman_order]

for i in range(0, spearman_p.shape[0]):
    for j in range(0, spearman_p.shape[1]):
        if (spearman_p.iloc[i, j] >= 0.05) or (np.isnan(spearman_p.iloc[i, j]) or spearman_rho.iloc[i, j] <= 0.2):
            spearman_rho.iloc[i, j] = 0

select_parameters = spearman_rho.loc[:, (spearman_rho != 0).any(axis=0)]
select_parameters.index = [crosswalk[i] for i in select_parameters.index]

select_parameters = select_parameters.loc[['*A1','A1','*A1e','A1e','*A2','*A3','A3',
                                           '*A4','A4','*A5','A5','*A6','A6',
                                           '*B1','*B1e','*B3','*B4','*B5','*B6',
                                           '*C1','*C1e','*C3','*C4','*C5','*C6',
                                           '*D1e','*D3',
                                           '*E1','*E1e','*E3','E3',
                                           'F1','F1e','F3','F4','F5','F6',
                                           '*G1','*G1-p','*G1e','*G1e-p','*G3',
                                           '*G4','*G5','*G6',
                                           'L-a','L-f','L-n','L-u'][::-1]]

select_parameters = select_parameters[::-1]

select_parameters = select_parameters.transpose()[::-1]

#%% sensitivity - visualization - annotated heat map

annotation = select_parameters[::-1].copy()
annotation = annotation.applymap(lambda x: '' if x == 0 else f'{x:.2f}')

color_map_Guest = colors.LinearSegmentedColormap.from_list('color_map_Guest', ['w', r, dr])

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

fig, ax = plt.subplots(figsize=(30, 3.63))

ax = plt.gca()

ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0, pad=7)

spearman_heatman = sns.heatmap(ax=ax,
                               data=select_parameters[::-1],
                               linewidths=1.5,
                               linecolor='black',
                               clip_on=False,
                               cmap=color_map_Guest,
                               cbar=False,
                               annot=annotation,
                               annot_kws={"size": 15},
                               fmt='',
                               yticklabels=['COD','TN','${CH_4}$ EF','${N_2O}$ EF','electricity','electricity EF'])

spearman_heatman.set_xticklabels(spearman_heatman.get_xticklabels(), rotation=90)

#%% total emission

TT_elec = []

for TT in WWTP_EF.index:
    breakdown_data_MC = pd.read_excel(f'MC/{TT}_MC.xlsx')
    
    TT_elec.append(breakdown_data_MC['elec_MC'].quantile(0.5))

assert (WWTP_EF.index != WWTP_TT_all.columns).sum() == 0

# direct emission in kg CO2-eq/day
WWTP_TT['CH4_median'] = WWTP_TT_all @ WWTP_EF['CH4_50']*MG_2_m3
WWTP_TT['N2O_median'] = WWTP_TT_all @ WWTP_EF['N2O_50']*MG_2_m3
WWTP_TT['NC_CO2_median'] = WWTP_TT_all @ WWTP_EF['NC_CO2_50']*MG_2_m3

# electricity emission in kg CO2-eq/day
WWTP_TT['electricity_median'] = (WWTP_TT_all @ TT_elec)*WWTP_TT['kg_CO2_kWh']

# NG emission in kg CO2-eq/day
WWTP_TT['natural_gas_combustion_median'] = WWTP_TT_all @ WWTP_EF['NG_combustion_50']*MG_2_m3
WWTP_TT['natural_gas_upstream_median'] = WWTP_TT_all @ WWTP_EF['NG_upstream_50']*MG_2_m3

# solids emission in kg CO2-eq/day
WWTP_TT['solids_landfilling_CH4_median'] = WWTP_TT['landfill']/365*5.65/1000*29.8
WWTP_TT['solids_land_application_N2O_median'] = WWTP_TT['land_application']/365*np.quantile(np.random.triangular(0.0122, 0.049, 0.062, 10000), 0.5)*0.01*N_2_N2O*273

WWTP_TT['total_median'] = WWTP_TT[['CH4_median','N2O_median','NC_CO2_median',
                                   'natural_gas_combustion_median',
                                   'natural_gas_upstream_median',
                                   'electricity_median',
                                   'solids_landfilling_CH4_median',
                                   'solids_land_application_N2O_median']].sum(axis=1)

#%% energy uncertainty - data preparation

def get_energy_uncertainty(quantile):
    quantile /= 100
    
    TT_elec_uncertainty_renamed = TT_elec_uncertainty.rename(columns=crosswalk)
    TT_elec_chemical_uncertainty_renamed = TT_elec_chemical_uncertainty.rename(columns=crosswalk)
    TT_NG_uncertainty_renamed = TT_NG_uncertainty.rename(columns=crosswalk)
    TT_NG_chemical_uncertainty_renamed = TT_NG_chemical_uncertainty.rename(columns=crosswalk)
    
    energy_uncertainty_result = pd.DataFrame()
    
    for dataset in [TT_elec_uncertainty_renamed, TT_elec_chemical_uncertainty_renamed, TT_NG_uncertainty_renamed, TT_NG_chemical_uncertainty_renamed]:
        for TT in ['A1','A1e','A3','A4','A5','A6','E3']:
            dataset['[*]'+TT] = (TT_flow[TT]*dataset.sort_values(by=TT)[TT] +\
                                 TT_flow['*'+TT]*dataset.sort_values(by=TT)['*'+TT])/(TT_flow[TT] + TT_flow['*'+TT])
        
        dataset.drop(columns=['A1','*A1','A1e','*A1e','A3','*A3','A4',
                              '*A4','A5','*A5','A6','*A6','E3','*E3'], inplace=True)
        
        dataset = dataset[label_order]
    
    # the baseline is median, use 5th/95th as error bars
    for TT in label_order:
        energy_uncertainty_result[TT] = {'total_electricity': TT_elec_uncertainty_renamed[TT].quantile(quantile),
                                         'total_electricity_50_5': TT_elec_uncertainty_renamed[TT].quantile(0.5) - TT_elec_uncertainty_renamed[TT].quantile(0.05),
                                         'total_electricity_95_50': TT_elec_uncertainty_renamed[TT].quantile(0.95) - TT_elec_uncertainty_renamed[TT].quantile(0.5),
                                         'chemical_electricity': TT_elec_chemical_uncertainty_renamed[TT].quantile(quantile),
                                         'total_NG': TT_NG_uncertainty_renamed[TT].quantile(quantile),
                                         'total_NG_50_5': TT_NG_uncertainty_renamed[TT].quantile(0.5) - TT_NG_uncertainty_renamed[TT].quantile(0.05),
                                         'total_NG_95_50': TT_NG_uncertainty_renamed[TT].quantile(0.95) - TT_NG_uncertainty_renamed[TT].quantile(0.5),
                                         'chemical_NG': TT_NG_chemical_uncertainty_renamed[TT].quantile(quantile)}
    
    return energy_uncertainty_result

energy_uncertainty_5 = get_energy_uncertainty(5)
energy_uncertainty_50 = get_energy_uncertainty(50)
energy_uncertainty_95 = get_energy_uncertainty(95)

#%% energy uncertainty - visualization - 50th
    
energy_uncertainty_50_to_plot = energy_uncertainty_50[energy_uncertainty_50.columns[::-1]]

fig, ax = plt.subplots(figsize=(32, 7))

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

width = 0.4

ax = plt.gca()
ax.set_xlim([0.15, len(label_order)+0.85])
ax.set_ylim([0, 10])

ax.tick_params(axis='x', direction='out', length=7.5, width=1.5, pad=0)
ax.tick_params(axis='y', direction='inout', length=15, width=1.5, pad=0)

plt.yticks(np.arange(0, 12, 2), fontname='Arial')

index = np.arange(1, len(label_order)+1, 1)

plt.xticks(index, label_order[::-1], rotation=90, fontname='Arial')

ax.bar(index-0.5*width,
       (energy_uncertainty_50_to_plot.loc['total_electricity'] - energy_uncertainty_50_to_plot.loc['chemical_electricity'])*kWh_2_MJ/MG_2_m3,
       width=width,
       color=y,
       edgecolor='k',
       linewidth=1.5)

# the baseline is median, use 5th/95th as error bars
ax.bar(index-0.5*width,
       energy_uncertainty_50_to_plot.loc['chemical_electricity']*kWh_2_MJ/MG_2_m3,
       yerr=(energy_uncertainty_50_to_plot.loc['total_electricity_50_5']*kWh_2_MJ/MG_2_m3, energy_uncertainty_50_to_plot.loc['total_electricity_95_50']*kWh_2_MJ/MG_2_m3),
       error_kw=dict(capsize=3.5, lw=1.5, capthick=1.5),
       width=width,
       color=y,
       edgecolor='k',
       linewidth=1.5,
       hatch='///',
       bottom=(energy_uncertainty_50_to_plot.loc['total_electricity'] - energy_uncertainty_50_to_plot.loc['chemical_electricity'])*kWh_2_MJ/MG_2_m3)

ax.bar(index+0.5*width,
       (energy_uncertainty_50_to_plot.loc['total_NG'] - energy_uncertainty_50_to_plot.loc['chemical_NG'])/MG_2_m3,
       width=width,
       color=b,
       edgecolor='k',
       linewidth=1.5)

ax.bar(index+0.5*width,
       energy_uncertainty_50_to_plot.loc['chemical_NG']/MG_2_m3,
       yerr=(energy_uncertainty_50_to_plot.loc['total_NG_50_5']/MG_2_m3, energy_uncertainty_50_to_plot.loc['total_NG_95_50']/MG_2_m3),
       error_kw=dict(capsize=3.5, lw=1.5, capthick=1.5),
       width=width,
       color=b,
       edgecolor='k',
       linewidth=1.5,
       hatch='///',
       bottom=(energy_uncertainty_50_to_plot.loc['total_NG'] - energy_uncertainty_50_to_plot.loc['chemical_NG'])/MG_2_m3)

ax.set_ylabel('$\mathbf{Upstream\ energy}$\n[MJ·${m^{-3}}$]', fontname='Arial',
              fontsize=28, labelpad=0, linespacing=0.8)

mathtext.FontConstantsBase.sup1 = 0.35

ax_left = ax.twinx()
ax_left.set_ylim(ax.get_ylim())

plt.yticks(np.arange(0, 12, 2), fontname='Arial')

ax_left.tick_params(direction='inout', length=15, width=1.5,
                    bottom=False, top=False, left=True, right=False,
                    pad=0, labelcolor='none')

ax_right = ax.twinx()
ax_right.set_ylim(ax.get_ylim())

plt.yticks(np.arange(0, 12, 2), fontname='Arial')

ax_right.tick_params(direction='in', length=7.5, width=1.5,
                     bottom=False, top=False, left=False, right=True,
                     pad=0, labelcolor='none')

#%% plot TTs number and flow

TT_flow_to_plot = TT_flow.copy()

TT_num_to_plot = TT_num.copy()

for TT in ['A1','A1e','A3','A4','A5','A6','E3']:
    TT_flow_to_plot.loc['[*]'+TT] = TT_flow_to_plot[TT] + TT_flow_to_plot['*'+TT]
    TT_num_to_plot.loc['[*]'+TT] = TT_num_to_plot[TT] + TT_num_to_plot['*'+TT]

TT_flow_to_plot.drop(index=['A1','*A1','A1e','*A1e','A3','*A3','A4',
                            '*A4','A5','*A5','A6','*A6','E3','*E3'], inplace=True)

TT_num_to_plot.drop(index=['A1','*A1','A1e','*A1e','A3','*A3','A4',
                           '*A4','A5','*A5','A6','*A6','E3','*E3'], inplace=True)

TT_flow_to_plot = TT_flow_to_plot.loc[label_order]

TT_num_to_plot = TT_num_to_plot.loc[label_order]

fig, ax = plt.subplots(figsize=(32, 7))

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

width = 0.8

ax = plt.gca()
ax.set_xlim([0.15, len(label_order)+0.85])
ax.set_ylim([0, 12])

plt.yticks(np.arange(0, 14, 2), fontname='Arial')

ax.tick_params(axis='x', direction='out', length=7.5, width=1.5, pad=0)
ax.tick_params(axis='y', direction='inout', length=15, width=1.5, pad=0)

index = np.arange(1, len(label_order)+1, 1)

ax.bar(index, TT_flow_to_plot[::-1]*MG_2_m3/1000000000*365, width=width,
       color=a, edgecolor='none', alpha=0.5)
ax.bar(index, TT_flow_to_plot[::-1]*MG_2_m3/1000000000*365, width=width,
       color='none', edgecolor='k', linewidth=1.5)

plt.xticks(index, label_order[::-1], rotation=90, fontname='Arial')

ax.set_ylabel('$\mathbf{Flow}$\n[billion ${m^{3}}$·${year^{-1}}$]',
              fontname='Arial', fontsize=28, linespacing=0.8)

mathtext.FontConstantsBase.sup1 = 0.35

ax_right = ax.twinx()

ax_right.set_ylim([0, 6])

plt.yticks(np.arange(0, 7, 1), fontname='Arial')

ax_right.tick_params(direction='inout', length=15, width=1.5,
                     bottom=False, top=False, left=False, right=True, pad=0)

ax_right.scatter(index, TT_num_to_plot[::-1]/1000, s=250, color='w', edgecolor='k', linewidth=1.5)

ax_right.set_ylabel('$\mathbf{Count}$ [k]', fontname='Arial', fontsize=28, labelpad=13)

#%% annual plot - visualization - data preparation

m3_to_annual = WWTP_EF.copy()*MG_2_m3
m3_to_annual.reset_index(inplace=True)
m3_to_annual['new_TT'] = m3_to_annual['index'].apply(lambda x: crosswalk[x])
m3_to_annual.set_index('new_TT', inplace=True)
m3_to_annual.drop(columns='index', inplace=True)

annual_plot = m3_to_annual.multiply(TT_flow, axis=0)

# kg/day to MMT/year
annual_plot = annual_plot*365/1000000000

annual_to_plot = annual_plot.copy()

for TT in ['A1','A1e','A3','A4','A5','A6','E3']:
    annual_to_plot.loc['[*]'+TT] = annual_to_plot.loc[TT] + annual_to_plot.loc['*'+TT]

annual_to_plot.drop(index=['A1','*A1','A1e','*A1e','A3','*A3','A4',
                           '*A4','A5','*A5','A6','*A6','E3','*E3'], inplace=True)

annual_to_plot = annual_to_plot.loc[label_order]

annual_result = annual_to_plot.copy()

#%% annual uncertainty with breakdown

fig, ax = plt.subplots(figsize=(32, 7))

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

ax = plt.gca()
ax.set_xlim([0.15, len(label_order)+0.85])
ax.set_ylim([0, 16])

ax.tick_params(direction='out', length=7.5, width=1.5,
               bottom=True, top=False, left=True, right=False, pad=0)

plt.xticks(np.arange(1, len(label_order)+1, 1), label_order[::-1], rotation=90, fontname='Arial')
plt.yticks(np.arange(0, 20, 4), fontname='Arial')

ax_left = ax.twinx()
ax_left.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 20, 4), fontname='Arial')

ax_left.tick_params(direction='inout', length=15, width=1.5,
                    bottom=False, top=False, left=True, right=False,
                    labelcolor='none')

ax_right = ax.twinx()
ax_right.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 20, 4), fontname='Arial')

ax_right.tick_params(direction='in', length=7.5, width=1.5,
                     bottom=False, top=False, left=False, right=True,
                     labelcolor='none')

ax.set_ylabel('$\mathbf{Annual\ GHG}$\n[MMT ${CO_2}$-eq·${year^{-1}}$]',
              fontname='Arial',
              fontsize=28,
              labelpad=0,
              linespacing=0.8)

mathtext.FontConstantsBase.sup1 = 0.35

index = np.arange(1, len(label_order)+1, 1)

width = 0.4

bp = ax.boxplot(total_MC[total_MC.columns[::-1]]*TT_flow_to_plot[::-1]*MG_2_m3/1000000000*365,
                positions=index-0.5*width,
                widths=width,
                whis=[5, 95],
                showfliers=False, vert=True,
                boxprops = dict(linestyle='-', linewidth=1.5, color='k'),
                medianprops = dict(linestyle='-', linewidth=1.5, color='k'),
                whiskerprops=dict(linestyle='-', linewidth=1.5),
                capprops=dict(linestyle='-', linewidth=1.5))

plt.xticks(np.arange(1, len(label_order)+1, 1), label_order[::-1], rotation=90, fontname='Arial')

ax.scatter(x=index-0.5*width,
           y=(total_MC[total_MC.columns[::-1]]*TT_flow_to_plot[::-1]*MG_2_m3/1000000000*365).mean(),
           marker='D',
           s=50,
           c='w',
           linewidths=1.5,
           edgecolors='k',
           alpha=1,
           zorder=3)

# onsite CH4, N2O, CO2
ax.bar(index+0.5*width,
       annual_to_plot[::-1]['CH4_50'],
       width=width,
       color=dr,
       edgecolor='k',
       linewidth=1.5)

ax.bar(index+0.5*width,
       annual_to_plot[::-1]['N2O_50'],
       width=width,
       color=r,
       edgecolor='k',
       linewidth=1.5,
       bottom=annual_to_plot[::-1]['CH4_50'])

ax.bar(index+0.5*width,
       annual_to_plot[::-1]['NC_CO2_50'],
       width=width,
       color=r,
       edgecolor='k',
       linewidth=0,
       alpha=0.5,
       bottom=annual_to_plot[::-1]['CH4_50']+annual_to_plot[::-1]['N2O_50'])

ax.bar(index+0.5*width,
       annual_to_plot[::-1]['NC_CO2_50'],
       width=width,
       color='none',
       edgecolor='k',
       linewidth=1.5,
       alpha=1,
       bottom=annual_to_plot[::-1]['CH4_50']+annual_to_plot[::-1]['N2O_50'])

# electricity
ax.bar(index+0.5*width,
       annual_to_plot[::-1]['elec_50'],
       width=width,
       color=y,
       edgecolor='k',
       linewidth=1.5,
       bottom=annual_to_plot[::-1]['CH4_50']+annual_to_plot[::-1]['N2O_50']+annual_to_plot[::-1]['NC_CO2_50'])

# NG
ax.bar(index+0.5*width,
       annual_to_plot[::-1]['NG_50'],
       width=width,
       color=b,
       edgecolor='k',
       linewidth=1.5,
       bottom=annual_to_plot[::-1]['CH4_50']+annual_to_plot[::-1]['N2O_50']+annual_to_plot[::-1]['NC_CO2_50']+annual_to_plot[::-1]['elec_50'])

#%% visualization in U.S. (data preparation)

WWTP_TT_results = WWTP_TT.copy()

TT_indentifier = WWTP_TT_results[final_code].apply(lambda x: x > 0)
WWTP_TT_results['TT'] = TT_indentifier.apply(lambda x: list(np.array(final_code)[x.values]), axis=1)

TT_indentifier.rename(columns=crosswalk, inplace=True)
final_code_array = np.array([crosswalk[i] for i in final_code])
WWTP_TT_results_output = WWTP_TT_results.copy()
WWTP_TT_results_output['TT'] = TT_indentifier.apply(lambda x: list(final_code_array[x.values]), axis=1)

# the lon and lat are in NAD83 (EPSG:4269)
WWTP_visual = gpd.GeoDataFrame(WWTP_TT_results, crs='EPSG:4269',
                               geometry=gpd.points_from_xy(x=WWTP_TT_results.LONGITUDE,
                                                           y=WWTP_TT_results.LATITUDE))

US = gpd.read_file('US_data/cb_2018_us_state_500k.shp')
US = US[['STUSPS','geometry']]

WWTP_visual = WWTP_visual.to_crs(crs='EPSG:4326')
US = US.to_crs(crs='EPSG:4326')

assert (WWTP_visual.LATITUDE == None).sum() == 0
assert (WWTP_visual.LONGITUDE == None).sum() == 0

# note the area of the marker is proportional to the emission
def add_TT_marker(dataset, option, TT, color, edgecolor, title):
    fig, ax = plt.subplots(figsize=(30, 30))
    
    US[~US['STUSPS'].isin(non_continental)].plot(ax=ax, color='white',
                                                 edgecolor='black', linewidth=3)
    
    if isinstance(TT, list):
        other_emission = dataset[dataset.TT.apply(lambda x: len([i for i in TT if i in x]) == 0)]
        other_emission = other_emission.sort_values(by=option, ascending=False)
        other_emission.plot(ax=ax, markersize=other_emission[option]/400, alpha=0.15,
                            color='none', edgecolor='k', linewidth=1.5)

        TT_emission = dataset[dataset.TT.apply(lambda x: len([i for i in TT if i in x]) != 0)]
        TT_emission = TT_emission.sort_values(by=option, ascending=False)
        TT_emission.plot(ax=ax, markersize=TT_emission[option]/400, alpha=1,
                         color=color, edgecolor=edgecolor, linewidth=1.5)
    else:
        other_emission = dataset[dataset.TT.apply(lambda x: TT not in x)]
        other_emission = other_emission.sort_values(by=option, ascending=False)
        other_emission.plot(ax=ax, markersize=other_emission[option]/400, alpha=0.15,
                            color='none', edgecolor='k', linewidth=1.5)

        TT_emission = dataset[dataset.TT.apply(lambda x: TT in x)]
        TT_emission = TT_emission.sort_values(by=option, ascending=False)
        TT_emission.plot(ax=ax, markersize=TT_emission[option]/400, alpha=1,
                         color=color, edgecolor=edgecolor, linewidth=1.5)
    
    color_1 = color_2 = color_3 = color_4 = 'w'
    
    max_size = max(TT_emission[option]/400)
    min_size = min(TT_emission[option]/400)
    
    if max_size > 2000000000/365/400:
        raise ValueError('add another layer of legend')
    elif max_size > 1000000000/365/400:
        color_1 = color_2 = color_3 = color_4 = color
    elif max_size > 250000000/365/400:
        color_2 = color_3 = color_4 = color
    elif max_size > 10000000/365/400:
        color_3 = color_4 = color
    else:
        color_4 = color
        
    if min_size > 2000000000/365/400:
        color_1 = color_2 = color_3 = color_4 = 'w'
        raise ValueError('add another layer of legend')
    elif min_size > 1000000000/365/400:
        color_2 = color_3 = color_4 = 'w'
    elif min_size > 250000000/365/400:
        color_3 = color_4 = 'w'
    elif min_size > 10000000/365/400:
        color_4 = 'w'
    
    rectangle_edge = Rectangle((-123.3, 24.52), 16.055, 6.483,
                               color='k', lw=3, fc='none', alpha=1)
    ax.add_patch(rectangle_edge)
    
    ax.scatter(x=-120.46, y=27.1, marker='o', s=2000000000/365/400, c=color_1, linewidths=3,
               alpha=1, edgecolor='k')
    ax.scatter(x=-120.46, y=27.1, marker='o', s=1000000000/365/400, c=color_2, linewidths=3,
               alpha=1, edgecolor='k')
    ax.scatter(x=-120.46, y=27.1, marker='o', s=250000000/365/400, c=color_3, linewidths=3,
               alpha=1, edgecolor='k')
    ax.scatter(x=-120.46, y=27.1, marker='o', s=10000000/365/400, c=color_4, linewidths=3,
               alpha=1, edgecolor='k')
    
    plt.figtext(0.179, 0.385, '[MMT ${CO_2}$-eq·${year^{-1}}$]', fontdict={'family':'Arial','fontsize': 42,'color':'k','fontweight':'bold'})
    plt.figtext(0.245, 0.364, '1st layer: 2', fontdict={'family':'Arial','fontsize': 42,'color':'k','style':'italic'})
    plt.figtext(0.245, 0.346, '2nd layer: 1', fontdict={'family':'Arial','fontsize': 42,'color':'k','style':'italic'})
    plt.figtext(0.245, 0.328, '3rd layer: 0.25', fontdict={'family':'Arial','fontsize': 42,'color':'k','style':'italic'})
    plt.figtext(0.245, 0.310, '4th layer: 0.01', fontdict={'family':'Arial','fontsize': 42,'color':'k','style':'italic'})
    
    ax.set_aspect(1.27)
    
    ax.set_axis_off()
    
    if title:
        if isinstance(TT, list):
            ax.set_title('[*]'+crosswalk[TT[0]], fontdict={'family':'Arial','fontsize': 50,'color':'k','fontweight':'bold'})
        else:
            ax.set_title(crosswalk[TT], fontdict={'family':'Arial','fontsize': 50,'color':'k','fontweight':'bold'})

#%% visualization in U.S. (all)

for TT in final_code:
    if crosswalk[TT] == 'A1':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'B1'], o, do, True)
    elif crosswalk[TT] == 'A1e':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'B1E'], o, do, True)
    elif crosswalk[TT] == 'A3':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'B2'], o, do, True)
    elif crosswalk[TT] == 'A4':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'B3'], o, do, True)
    elif crosswalk[TT] == 'A5':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'B5'], o, do, True)
    elif crosswalk[TT] == 'A6':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'B6'], o, do, True)
    elif crosswalk[TT] == 'E3':
        add_TT_marker(WWTP_visual, 'total_median', [TT, 'E2P'], o, do, True)
    elif crosswalk[TT] in ['*A1','*A1e','*A3','*A4','*A5','*A6','*E3']:
        pass
    else:
        add_TT_marker(WWTP_visual, 'total_median', TT, o, do, True)

#%% visualization in U.S. (highlighting all WWTPs with anaerobic digestion)

add_TT_marker(WWTP_visual, 'total_median', ['B1','B1E','B4','C1','C1E','D1','D1E','F1','F1E','G1','G1E',
                                            'H1','H1E','I1','I1E','N1','N1E','O1','O1E'], r, dr, False)

#%% visualization in U.S. (highlighting F1 -> *E1 and F1E -> *E1e)

add_TT_marker(WWTP_visual, 'total_median', ['F1','F1E'], b, db, False)

#%% visualization in U.S. (highlighting I1 -> F1 and I1E -> F1e)

add_TT_marker(WWTP_visual, 'total_median', ['I1','I1E'], g, dg, False)

#%% visualization in U.S. (highlighting all lagoons)

add_TT_marker(WWTP_visual, 'total_median', ['LAGOON_AER','LAGOON_ANAER','LAGOON_FAC','LAGOON_UNCATEGORIZED'], y, dy, False)

#%% emission fraction vs facility number fraction

facility_data = WWTP_TT_results_output.copy()

sorted_facilities = facility_data.sort_values(by='total_median', ascending=False).reset_index(drop=True)
sorted_facilities['cumulative_emissions'] = sorted_facilities['total_median'].cumsum()
total_emissions = sorted_facilities['total_median'].sum()
sorted_facilities['cumulative_distribution'] = sorted_facilities['cumulative_emissions']/total_emissions

sorted_facilities['facility_rank'] = sorted_facilities.index + 1
sorted_facilities['facility_fraction'] = sorted_facilities['facility_rank']/len(sorted_facilities)

fig, ax = plt.subplots(figsize=(6, 6))
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

ax = plt.gca()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax.tick_params(direction='inout', length=15, width=1.5,
               bottom=True, top=False, left=True, right=False, pad=0)

plt.xticks(np.arange(0, 1.2, 0.2), fontname='Arial')
plt.yticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
plt.xticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_top.tick_params(direction='in', length=7.5, width=1.5,
                   bottom=False, top=True, left=False, right=False,
                   labelcolor='none')

ax_bottom = ax.twinx()
ax_bottom.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_bottom.tick_params(direction='in', length=7.5, width=1.5,
                      bottom=False, top=False, left=False, right=True,
                      labelcolor='none')

plt.plot(sorted_facilities['facility_fraction'],
         sorted_facilities['cumulative_distribution'],
         linewidth=1.5,
         marker='o',
         color='k',
         markersize=pi*1.5**2)

ax.set_xlabel('$\mathbf{Fraction\ of\ total\ facilities}$',
              fontname='Arial',
              fontsize=28,
              labelpad=0)

ax.set_ylabel('$\mathbf{Fraction\ of\ total\ emissions}$',
              fontname='Arial',
              fontsize=28,
              labelpad=0)

mathtext.FontConstantsBase.sup1 = 0.35

#%% emission fraction vs flow fraction

flow_sorted = facility_data.sort_values(by='FLOW_2022_MGD_FINAL', ascending=False).reset_index(drop=True)
flow_sorted['cumulative_flow'] = flow_sorted['FLOW_2022_MGD_FINAL'].cumsum()
total_flow = flow_sorted['FLOW_2022_MGD_FINAL'].sum()
flow_sorted['flow_fraction'] = flow_sorted['cumulative_flow']/total_flow

flow_sorted['cumulative_emissions'] = flow_sorted['total_median'].cumsum()
total_emissions = flow_sorted['total_median'].sum()
flow_sorted['emissions_fraction'] = flow_sorted['cumulative_emissions']/total_emissions

fig, ax = plt.subplots(figsize=(6, 6))
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

ax = plt.gca()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax.tick_params(direction='inout', length=15, width=1.5,
               bottom=True, top=False, left=True, right=False, pad=0)

plt.xticks(np.arange(0, 1.2, 0.2), fontname='Arial')
plt.yticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
plt.xticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_top.tick_params(direction='in', length=7.5, width=1.5,
                   bottom=False, top=True, left=False, right=False,
                   labelcolor='none')

ax_bottom = ax.twinx()
ax_bottom.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_bottom.tick_params(direction='in', length=7.5, width=1.5,
                      bottom=False, top=False, left=False, right=True,
                      labelcolor='none')

plt.plot(flow_sorted['flow_fraction'],
         flow_sorted['emissions_fraction'],
         linewidth=1.5,
         marker='o',
         color='k',
         markersize=pi*1.5**2,
         zorder=0)

plt.plot([0,1],
         [0,1],
         linewidth=2,
         color=r,
         zorder=1)

ax.set_xlabel('$\mathbf{Fraction\ of\ total\ flow}$',
              fontname='Arial',
              fontsize=28,
              labelpad=0)

#%% emission fraction vs emissions magnitude

fig, ax = plt.subplots(figsize=(6, 6))

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['hatch.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'mathtext.bf': 'Arial: bold'})

ax = plt.gca()
ax.set_xlim([0, 1.4])
ax.set_ylim([0, 1])

ax.tick_params(direction='inout', length=15, width=1.5,
               bottom=True, top=False, left=True, right=False, pad=0)

plt.xticks(np.arange(0, 1.6, 0.2), fontname='Arial')
plt.yticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
plt.xticks(np.arange(0, 1.6, 0.2), fontname='Arial')

ax_top.tick_params(direction='in', length=7.5, width=1.5,
                   bottom=False, top=True, left=False, right=False,
                   labelcolor='none')

ax_bottom = ax.twinx()
ax_bottom.set_ylim(ax.get_ylim())
plt.yticks(np.arange(0, 1.2, 0.2), fontname='Arial')

ax_bottom.tick_params(direction='in', length=7.5, width=1.5,
                      bottom=False, top=False, left=False, right=True,
                      labelcolor='none')

plt.plot(sorted_facilities['total_median']/1000000000*365,
         sorted_facilities['cumulative_distribution'],
         linewidth=1.5,
         marker='o',
         color='k',
         markersize=pi*1.5**2)

ax.set_xlabel('$\mathbf{Median\ emissions\ magnitude}$\n[MMT ${CO_2}$-eq·${year^{-1}}$]',
              fontname='Arial',
              fontsize=28,
              labelpad=0,
              linespacing=0.8)

mathtext.FontConstantsBase.sup1 = 0.35

#%% numbers calculation

# =============================================================================
# Characterizing U.S. wastewater treatment facilities
# =============================================================================

WWTP_TT_results_output['all_TT'] = WWTP_TT_results_output['TT'].apply(lambda x: ', '.join(map(str, x)))

sum(WWTP_TT_results_output['all_TT'].str.contains('A'))
sum(WWTP_TT_results_output['all_TT'].str.contains('A'))/len(WWTP_TT_results_output)

# TT_num_output cannot be used to count facility numbers, but flow is ok
TT_flow_output = TT_flow_to_plot.copy()

sum(TT_flow_output[i] for i in TT_flow_output.index if 'A' in i)*MG_2_m3/1000000000*365
sum(TT_flow_output[i] for i in TT_flow_output.index if 'A' in i)/sum(TT_flow_output)

sum(WWTP_TT_results_output['all_TT'].str.contains('L'))
sum(TT_flow_output[i] for i in TT_flow_output.index if 'L' in i)/sum(TT_flow_output)

sum(WWTP_TT_results_output['all_TT'].str.contains('3'))/len(WWTP_TT_results_output)
sum(WWTP_TT_results_output['all_TT'].str.contains('1|2'))/len(WWTP_TT_results_output)

# based on facilities
WWTP_TT_results_output[WWTP_TT_results_output['all_TT'].str.contains('1|2')]['FLOW_2022_MGD_FINAL'].sum()/WWTP_TT_results_output['FLOW_2022_MGD_FINAL'].sum()
WWTP_TT_results_output[WWTP_TT_results_output['all_TT'].str.contains('3')]['FLOW_2022_MGD_FINAL'].sum()/WWTP_TT_results_output['FLOW_2022_MGD_FINAL'].sum()
# based on TTs
sum(TT_flow_output[i] for i in TT_flow_output.index if ('1' in i) | ('2' in i))/sum(TT_flow_output)
sum(TT_flow_output[i] for i in TT_flow_output.index if '3' in i)/sum(TT_flow_output)

sum(WWTP_TT_results_output['all_TT'].str.contains('e'))
sum(WWTP_TT_results_output['all_TT'].str.contains('e'))/sum(WWTP_TT_results_output['all_TT'].str.contains('1|2'))
WWTP_TT_results_output[WWTP_TT_results_output['all_TT'].str.contains('e')]['FLOW_2022_MGD_FINAL'].mean()*MG_2_m3
WWTP_TT_results_output[WWTP_TT_results_output['all_TT'].str.contains('e')]['FLOW_2022_MGD_FINAL'].mean()

e_flow_mean = WWTP_TT_results_output[WWTP_TT_results_output['all_TT'].str.contains('e')]['FLOW_2022_MGD_FINAL'].mean()
(flow_sorted['FLOW_2022_MGD_FINAL'] < e_flow_mean).sum()/len(flow_sorted)

sum(WWTP_TT_results_output['all_TT'].str.contains('A3,|A3$'))/len(WWTP_TT_results_output)
sum(WWTP_TT_results_output['all_TT'].str.contains('A3,|A3$'))

TT_flow_output['[*]A3'].sum()/sum(TT_flow_output)
sum(TT_flow_output[i] for i in TT_flow_output.index if '[*]A3' in i)*MG_2_m3/1000000000*365

# [*]A1 + [*]A1e
sum(WWTP_TT_results_output['all_TT'].str.contains('A1'))/len(WWTP_TT_results_output)
TT_flow_output[['[*]A1','[*]A1e']].sum()/sum(TT_flow_output)

sum(WWTP_TT_results_output['all_TT'].str.contains('D|E|F|G'))/len(WWTP_TT_results_output)
sum(TT_flow_output[i] for i in TT_flow_output.index if ('D' in i) | ('E' in i) | ('F' in i) | ('G' in i))/sum(TT_flow_output)
sum(TT_flow_output[i] for i in TT_flow_output.index if ('D' in i) | ('E' in i) | ('F' in i) | ('G' in i))*MG_2_m3/1000000000*365

# =============================================================================
# Electricity and natural gas requirements
# =============================================================================

energy_uncertainty_5.loc['total_electricity']/MG_2_m3
energy_uncertainty_50.loc['total_electricity']/MG_2_m3
energy_uncertainty_95.loc['total_electricity']/MG_2_m3

energy_uncertainty_5.loc['total_NG']/MG_2_m3
energy_uncertainty_50.loc['total_NG']/MG_2_m3
energy_uncertainty_95.loc['total_NG']/MG_2_m3

top_10_energy_TTs = energy_uncertainty_50.loc['total_electricity'].sort_values(ascending=False).index[0:10]

TT_elec_output = TT_elec_uncertainty.rename(columns=crosswalk)
TT_elec_chemical_output = TT_elec_chemical_uncertainty.rename(columns=crosswalk)

1 - (TT_elec_chemical_output[top_10_energy_TTs]/TT_elec_output[top_10_energy_TTs]).mean()

# =============================================================================
# Carbon dioxide, methane, and nitrous oxide emissions
# =============================================================================

star_A1 = pd.read_excel('MC/B1_MC.xlsx')
star_A1e = pd.read_excel('MC/B1E_MC.xlsx')
star_A2 = pd.read_excel('MC/B4_MC.xlsx')
star_A3 = pd.read_excel('MC/B2_MC.xlsx')
star_A4 = pd.read_excel('MC/B3_MC.xlsx')
star_A5 = pd.read_excel('MC/B5_MC.xlsx')
star_A6 = pd.read_excel('MC/B6_MC.xlsx')
A1 = pd.read_excel('MC/C1_MC.xlsx')
A1e = pd.read_excel('MC/C1E_MC.xlsx')
A3 = pd.read_excel('MC/C2_MC.xlsx')
A4 = pd.read_excel('MC/C3_MC.xlsx')
A5 = pd.read_excel('MC/C5_MC.xlsx')
A6 = pd.read_excel('MC/C6_MC.xlsx')
star_B1 = pd.read_excel('MC/O1_MC.xlsx')
star_B1e = pd.read_excel('MC/O1E_MC.xlsx')
star_B3 = pd.read_excel('MC/O2_MC.xlsx')
star_B4 = pd.read_excel('MC/O3_MC.xlsx')
star_B5 = pd.read_excel('MC/O5_MC.xlsx')
star_B6 = pd.read_excel('MC/O6_MC.xlsx')
star_C1 = pd.read_excel('MC/D1_MC.xlsx')
star_C1e = pd.read_excel('MC/D1E_MC.xlsx')
star_C3 = pd.read_excel('MC/D2_MC.xlsx')
star_C4 = pd.read_excel('MC/D3_MC.xlsx')
star_C5 = pd.read_excel('MC/D5_MC.xlsx')
star_C6 = pd.read_excel('MC/D6_MC.xlsx')
star_D1e = pd.read_excel('MC/N1E_MC.xlsx')
star_D3 = pd.read_excel('MC/N2_MC.xlsx')
star_E1 = pd.read_excel('MC/F1_MC.xlsx')
star_E1e = pd.read_excel('MC/F1E_MC.xlsx')
E3 = pd.read_excel('MC/E2_MC.xlsx')
star_E3 = pd.read_excel('MC/E2P_MC.xlsx')
F1 = pd.read_excel('MC/I1_MC.xlsx')
F1e = pd.read_excel('MC/I1E_MC.xlsx')
F3 = pd.read_excel('MC/I2_MC.xlsx')
F4 = pd.read_excel('MC/I3_MC.xlsx')
F5 = pd.read_excel('MC/I5_MC.xlsx')
F6 = pd.read_excel('MC/I6_MC.xlsx')
star_G1 = pd.read_excel('MC/G1_MC.xlsx')
star_G1_p = pd.read_excel('MC/H1_MC.xlsx')
star_G1e = pd.read_excel('MC/G1E_MC.xlsx')
star_G1e_p = pd.read_excel('MC/H1E_MC.xlsx')
star_G3 = pd.read_excel('MC/G2_MC.xlsx')
star_G4 = pd.read_excel('MC/G3_MC.xlsx')
star_G5 = pd.read_excel('MC/G5_MC.xlsx')
star_G6 = pd.read_excel('MC/G6_MC.xlsx')
L_a = pd.read_excel('MC/LAGOON_AER_MC.xlsx')
L_f = pd.read_excel('MC/LAGOON_FAC_MC.xlsx')
L_n = pd.read_excel('MC/LAGOON_ANAER_MC.xlsx')
L_u = pd.read_excel('MC/LAGOON_UNCATEGORIZED_MC.xlsx')

all_TT_data = [('*A1', star_A1), ('*A1e', star_A1e), ('*A2', star_A2), ('*A3', star_A3), ('*A4', star_A4),
               ('*A5', star_A5), ('*A6', star_A6), ('A1', A1), ('A1e', A1e), ('A3', A3), ('A4', A4),
               ('A5', A5), ('A6', A6), ('*B1', star_B1), ('*B1e', star_B1e), ('*B3', star_B3),
               ('*B4', star_B4), ('*B5', star_B5), ('*B6', star_B6), ('*C1', star_C1), ('*C1e', star_C1e),
               ('*C3', star_C3), ('*C4', star_C4), ('*C5', star_C5), ('*C6', star_C6), ('*D1e', star_D1e),
               ('*D3', star_D3), ('*E1', star_E1), ('*E1e', star_E1e), ('E3', E3), ('*E3', star_E3),
               ('F1', F1), ('F1e', F1e), ('F3', F3), ('F4', F4), ('F5', F5), ('F6', F6), ('*G1', star_G1),
               ('*G1-p', star_G1_p), ('*G1e', star_G1e), ('*G1e-p', star_G1e_p), ('*G3', star_G3),
               ('*G4', star_G4), ('*G5', star_G5), ('*G6', star_G6), ('L-a', L_a), ('L-f', L_f),
               ('L-n', L_n), ('L-u', L_u)]

all_TT_data_dict = {i:j for (i, j) in all_TT_data}

def get_quantiles(data):
    print(f'5th percentile: {np.quantile(data, 0.05):.2g}')
    print(f'50th percentile: {np.quantile(data, 0.5):.2g}')
    print(f'95th percentile: {np.quantile(data, 0.95):.2g}')

biological_emission_finder = pd.DataFrame()
CH4_N2O_50_finder = []
NC_CO2_50_finder = []

for TT in final_code:
    breakdown_data_MC = pd.read_excel(f'MC/{TT}_MC.xlsx')
    CH4_N2O_50_finder.append((breakdown_data_MC[['CH4','N2O']].sum(axis=1)/breakdown_data_MC[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5))
    NC_CO2_50_finder.append((breakdown_data_MC['NC_CO2']/breakdown_data_MC[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5))

biological_emission_finder['CH4_N2O'] = CH4_N2O_50_finder
biological_emission_finder['NC_CO2'] = NC_CO2_50_finder
biological_emission_finder.index = [crosswalk[i] for i in final_code]

biological_emission_finder['CH4_N2O'].min()
biological_emission_finder['CH4_N2O'].max()

biological_emission_finder[biological_emission_finder['NC_CO2']>0.11]

star_E1_e = pd.concat([star_E1, star_E1e])

(star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.05)
(star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5)
(star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.95)

(star_E1_e['CH4']/star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.05)
(star_E1_e['CH4']/star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5)
(star_E1_e['CH4']/star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.95)

(star_E1_e['N2O']/star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.05)
(star_E1_e['N2O']/star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5)
(star_E1_e['N2O']/star_E1_e[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.95)

L_n_f = pd.concat([L_n, L_f])

(L_n_f[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.05)
(L_n_f[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5)
(L_n_f[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.95)

(L_n_f['CH4']/L_n_f[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.05)
(L_n_f['CH4']/L_n_f[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.5)
(L_n_f['CH4']/L_n_f[['CH4','N2O','NC_CO2']].sum(axis=1)).quantile(0.95)

E_data = [('*E1', star_E1), ('*E1e', star_E1e), ('E3', E3), ('*E3', star_E3)]

E_biological_annual = sum(np.random.choice(TT_data[['CH4','N2O','NC_CO2']].sum(axis=1), 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in E_data)

np.quantile(E_biological_annual, 0.05)
np.quantile(E_biological_annual, 0.5)
np.quantile(E_biological_annual, 0.95)

A_data = [('*A1', star_A1), ('*A1e', star_A1e), ('*A2', star_A2), ('*A3', star_A3), ('*A4', star_A4),
          ('*A5', star_A5), ('*A6', star_A6), ('A1', A1), ('A1e', A1e), ('A3', A3), ('A4', A4), ('A5', A5), ('A6', A6)]

A_biological_annual = sum(np.random.choice(TT_data[['CH4','N2O','NC_CO2']].sum(axis=1), 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in A_data)

np.quantile(A_biological_annual, 0.05)
np.quantile(A_biological_annual, 0.5)
np.quantile(A_biological_annual, 0.95)

L_data = [('L-a', L_a), ('L-f', L_f), ('L-n', L_n), ('L-u', L_u)]

L_biological_annual = sum(np.random.choice(TT_data[['CH4','N2O','NC_CO2']].sum(axis=1), 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in L_data)

np.quantile(L_biological_annual, 0.05)
np.quantile(L_biological_annual, 0.5)
np.quantile(L_biological_annual, 0.95)

# =============================================================================
# Total emissions by treatment configuration and nation-wide
# =============================================================================

total_MC_output = total_MC.copy()

total_MC_output.quantile(0.05)
total_MC_output.quantile(0.5)
total_MC_output.quantile(0.95)

(star_D1e[['elec','NG']].sum(axis=1)/star_D1e['total']).quantile(0.05)
(star_D1e[['elec','NG']].sum(axis=1)/star_D1e['total']).quantile(0.5)
(star_D1e[['elec','NG']].sum(axis=1)/star_D1e['total']).quantile(0.95)

(star_D3[['elec','NG']].sum(axis=1)/star_D3['total']).quantile(0.05)
(star_D3[['elec','NG']].sum(axis=1)/star_D3['total']).quantile(0.5)
(star_D3[['elec','NG']].sum(axis=1)/star_D3['total']).quantile(0.95)

(star_G1[['CH4','N2O','NC_CO2']].sum(axis=1)/star_G1['total']).quantile(0.05)
(star_G1[['CH4','N2O','NC_CO2']].sum(axis=1)/star_G1['total']).quantile(0.5)
(star_G1[['CH4','N2O','NC_CO2']].sum(axis=1)/star_G1['total']).quantile(0.95)

(star_G1[['elec','NG']].sum(axis=1)/star_G1['total']).quantile(0.05)
(star_G1[['elec','NG']].sum(axis=1)/star_G1['total']).quantile(0.5)
(star_G1[['elec','NG']].sum(axis=1)/star_G1['total']).quantile(0.95)

solids_emission_finder = pd.DataFrame()
solids_50_finder = []

for TT in final_code:
    breakdown_data_MC = pd.read_excel(f'MC/{TT}_MC.xlsx')
    solids_50_finder.append((breakdown_data_MC['solids']/breakdown_data_MC['total']).quantile(0.5))

solids_emission_finder['solids'] = solids_50_finder
solids_emission_finder.index = [crosswalk[i] for i in final_code]

solids_emission_finder.max()
solids_emission_finder

(star_C4['solids']/star_C4['total']).quantile(0.05)
(star_C4['solids']/star_C4['total']).quantile(0.5)
(star_C4['solids']/star_C4['total']).quantile(0.95)

for TT in [i for i in total_MC_output.columns if 'e' in i]:
    try:
        print((total_MC_output.quantile(0.5)[TT] - total_MC_output.quantile(0.5)[TT.replace('e','')])/total_MC_output.quantile(0.5)[TT]*100)
    except KeyError:
        pass

# CH4
annual_CH4 = sum(np.random.choice(TT_data['CH4'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_CH4)

# N2O
annual_N2O = sum(np.random.choice(TT_data['N2O'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_N2O)

# NC CO2
annual_NC_CO2 = sum(np.random.choice(TT_data['NC_CO2'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_NC_CO2)

# electricity
balnc_area_flow = WWTP_TT_all.copy()
balnc_area_flow['balancing_area'] = WWTP_TT['r']
balnc_area_flow = balnc_area_flow.groupby(by='balancing_area').sum()

total = []
for i in balnc_area_flow.index:
    total.append([j*np.random.uniform(balnc_area[balnc_area['r'] == i]['kg_CO2_kWh']*0.8, balnc_area[balnc_area['r'] == i]['kg_CO2_kWh']*1.2, 10000) for j in balnc_area_flow.loc[i]])

TT_flow_elec = pd.DataFrame([sum(np.random.choice(total[i][j], 10000) for i in range(len(total))) for j in range(len(total[0]))])
TT_flow_elec.index = balnc_area_flow.columns
TT_flow_elec = TT_flow_elec.transpose()
TT_flow_elec.rename(columns=crosswalk, inplace=True)

annual_elec = sum(np.random.choice(TT_data['elec_MC'], 10000)*np.random.choice(TT_flow_elec[TT], 10000)/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_elec)

assert np.quantile(annual_elec, 0.5) < np.quantile(annual_N2O, 0.5)

# NG combustion
annual_NG_combustion = sum(np.random.choice(TT_data['NG_combustion'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_NG_combustion)

# NG upstream
annual_NG_upstream = sum(np.random.choice(TT_data['NG_upstream'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_NG_upstream)

# NG
annual_NG = sum(np.random.choice(TT_data[['NG_combustion','NG_upstream']].sum(axis=1), 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data)
get_quantiles(annual_NG)

# landfilling
annual_solids_LF = np.random.uniform(WWTP_TT['landfill'].sum()*0.8, WWTP_TT['landfill'].sum()*1.2, 10000)*np.random.uniform(5.65*0.9/1000, 5.65*1.1/1000, 10000)*np.random.uniform(29.8*0.9, 29.8*1.1)/1000000000
get_quantiles(annual_solids_LF)

# land application
annual_solids_LA = np.random.uniform(WWTP_TT['land_application'].sum()*0.8, WWTP_TT['land_application'].sum()*1.2, 10000)*np.random.triangular(0.0122, 0.049, 0.062, 10000)*np.random.uniform(0.002, 0.018, 10000)*N_2_N2O*np.random.uniform(273*0.9, 273*1.1, 10000)/1000000000
get_quantiles(annual_solids_LA)

# solids
annual_solids = np.random.choice(annual_solids_LF, 10000) + np.random.choice(annual_solids_LA, 10000)
get_quantiles(annual_solids)

# total
annual_total = sum(np.random.choice(i, 10000) for i in [annual_CH4, annual_N2O, annual_NC_CO2, annual_elec, annual_NG_combustion, annual_NG_upstream, annual_solids_LF, annual_solids_LA])
get_quantiles(annual_total)

# CH4 percentage
annual_CH4_total = np.random.choice(annual_CH4, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_CH4_total)

# N2O percentage
annual_N2O_total = np.random.choice(annual_N2O, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_N2O_total)

# NC CO2 percentage
annual_NC_CO2_total = np.random.choice(annual_NC_CO2, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_NC_CO2_total)

# electricity percentage
annual_elec_total = np.random.choice(annual_elec, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_elec_total)

# NG combustion percentage
annual_NG_combustion_total = np.random.choice(annual_NG_combustion, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_NG_combustion_total)

# NG upstream percentage
annual_NG_upstream_total = np.random.choice(annual_NG_upstream, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_NG_upstream_total)

# landfilling percentage
annual_solids_LF_total = np.random.choice(annual_solids_LF, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_solids_LF_total)

# land application percentage
annual_solids_LA_total = np.random.choice(annual_solids_LA, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_solids_LA_total)

# solids percentage
annual_solids_total = np.random.choice(annual_solids, 10000)/np.random.choice(annual_total, 10000)
get_quantiles(annual_solids_total)

# =============================================================================
# National distribution of GHG emissions from WWTP
# =============================================================================

WWTP_w_lagoon_total = WWTP_TT_results_output[WWTP_TT_results_output['all_TT'].str.contains('L')]['total_median'].sum()
WWTP_total = WWTP_TT_results_output['total_median'].sum()
WWTP_w_lagoon_total/WWTP_total

top_10_percent_emitters = WWTP_TT_results_output.sort_values(by='total_median', ascending=False).iloc[0:round(len(WWTP_TT)/10)]
top_10_percent_emitters['total_median'].sum()/WWTP_total

top_10_emitters = WWTP_TT_results_output.sort_values(by='total_median', ascending=False).iloc[0:10]
10/len(WWTP_TT)
top_10_emitters['FLOW_2022_MGD_FINAL'].sum()/WWTP_TT_results_output['FLOW_2022_MGD_FINAL'].sum()
top_10_emitters['total_median'].sum()*365/1000000000
top_10_emitters['total_median'].sum()/WWTP_total

# =============================================================================
# Discussion and Conclusion
# =============================================================================

annual_CH4_N2O_total = (np.random.choice(annual_CH4, 10000) + np.random.choice(annual_N2O, 10000))/np.random.choice(annual_total, 10000)
get_quantiles(annual_CH4_N2O_total)

total_MC_output.quantile(0.05)
total_MC_output.quantile(0.5)
total_MC_output.quantile(0.95)

AD_annual_CH4 = sum(np.random.choice(TT_data['CH4'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data if ('1' in TT) or ('2' in TT))
other_annual_CH4 = sum(np.random.choice(TT_data['CH4'], 10000)*TT_flow[TT]*MG_2_m3/1000000000*365 for (TT, TT_data) in all_TT_data if ('1' not in TT) and ('2' not in TT))
AD_annual_CH4_random = np.random.choice(AD_annual_CH4, 10000)
other_annual_CH4_random = np.random.choice(other_annual_CH4, 10000)
CH4_AD_total = AD_annual_CH4_random/(AD_annual_CH4_random + other_annual_CH4_random)
get_quantiles(CH4_AD_total)
get_quantiles(AD_annual_CH4)

emission_reduction_CHP = []
for TT in [i for i in total_MC_output.columns if 'e' in i]:
    try:
        emission_reduction_CHP.append(total_MC_output.quantile(0.5)[TT] - total_MC_output.quantile(0.5)[TT.replace('e','')])
    except KeyError:
        pass
np.mean(emission_reduction_CHP)

average_emission_AD = WWTP_EF.loc[[TT for TT in WWTP_EF.index if ('1' in crosswalk[TT]) or ('2' in crosswalk[TT])]]['CH4_50'].mean()
average_emission_AD
average_emission_AeD = WWTP_EF.loc[[TT for TT in WWTP_EF.index if '3' in crosswalk[TT]]]['CH4_50'].mean()
average_emission_AD - average_emission_AeD

(np.quantile(np.random.choice(annual_total, 10000), 0.5) - 28.3)/28.3

annual_CH4_N2O = np.random.choice(annual_CH4, 10000) + np.random.choice(annual_N2O, 10000)
get_quantiles(annual_CH4_N2O)

CH4_random = np.random.choice(annual_CH4, 10000)
N2O_random = np.random.choice(annual_N2O, 10000)

annual_CH4_CH4_N2O = CH4_random/(CH4_random + N2O_random)
get_quantiles(annual_CH4_CH4_N2O)

# =============================================================================
# Abstract
# =============================================================================
(np.quantile(annual_CH4_N2O, 0.5) - 21.9)/21.9