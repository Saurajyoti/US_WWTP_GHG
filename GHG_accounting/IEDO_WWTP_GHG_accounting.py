#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Inventory of wastewater plants in the United States and their associated energy
usage and greenhouse gas emissions

The code is developed by:
    Jianan Feng <jiananf2@illinois.edu>
  
References:
[1] Argonne National Laboratory. GREET Model.
[2] Song, C. et al. Methane Emissions from Municipal Wastewater Collection and
    Treatment Systems. Environmental Science & Technology, 2023.
[3] IPCC. 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas
    Inventories. 2019.
[4] Song, C. et al. Oversimplification and misestimation of nitrous oxide
    emissions from wastewater treatment plants. Nature Sustainability, 2024.
[5] Law, Y. et al. Fossil organic carbon in wastewater and its fate in treatment
    plants. Water Research, 2013.
[6] Seiple, T. E. et al. Municipal wastewater sludge as a sustainable bioresource 
    in the United States. Journal of Environmental Management, 2017.
[7] Metcalf & Eddy Inc. Wastewater Engineering: Treatment and Resource Recovery
    (5th edition). McGraw-Hill Professional, 2013.
[8] US EPA. Clean Watersheds Needs Survey (CWNS) – 2022 Report and Data. 2024.
[9] US EPA. Landfill Gas Emissions Model (LandGEM). 2023.
[10] Rigby, H. et al. A critical review of nitrogen mineralization in
     biosolids-amended soil, the associated fertilizer value for crop production
     and potential for emissions to the environment. Science of The Total
     Environment, 2016.
'''

#%% initialization

import numpy as np, pandas as pd
import matplotlib.colors as colors
from colorpalette import Color
from statsmodels.stats.weightstats import DescrStatsW

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
CH4_factor = 29.8
N2O_factor = 273
# convert N2O-N to N2O
N_2_N2O = 44/28
# onsite NG combustion emission, kg CO2 eq/MJ, [1]
NG_combustion = 56.3/1000
# upstream NG emission, kg CO2 eq/MJ, [1]
upstream_NG_GHG = 12.7/1000

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

#%% energy emission factors

# efs: emission factors
efs = pd.read_excel('treatment_train_energy_typical.xlsx','All Trains (For Code)')
efs.fillna(0, inplace=True)
efs.rename(columns={'Unnamed: 0':'TT'}, inplace=True)
efs.set_index('TT', inplace=True)

efs = efs.transpose()
efs = efs[['Total Electricity Usage [kWh/d] (including chemical production)',
           'Total Electricity Usage [kWh/d] (excluding chemical production)',
           'CHP Electricity Generation [kWh/d]',
           'Total Natural Gas Usage [MJ/d] (including chemical production)',
           'Total Natural Gas Usage [MJ/d] (excluding chemical production)']]

efs['electricity_kWh_d'] = efs['Total Electricity Usage [kWh/d] (including chemical production)'] -\
                           efs['CHP Electricity Generation [kWh/d]']
efs['electricity_chem_kWh_d'] = efs['Total Electricity Usage [kWh/d] (including chemical production)'] -\
                                efs['Total Electricity Usage [kWh/d] (excluding chemical production)']
efs['electricity_other_kWh_d'] = efs['electricity_kWh_d'] - efs['electricity_chem_kWh_d']

efs.rename(columns={'Total Natural Gas Usage [MJ/d] (including chemical production)':'natural_gas_MJ_d'},
           inplace=True)
efs['natural_gas_chem_MJ_d'] = efs['natural_gas_MJ_d'] -\
                               efs['Total Natural Gas Usage [MJ/d] (excluding chemical production)']
efs['natural_gas_other_MJ_d'] = efs['Total Natural Gas Usage [MJ/d] (excluding chemical production)']

# the plant size is 10 MGD
efs['electricity_kWh_MG'] = efs['electricity_kWh_d']/10
efs['electricity_chem_kWh_MG'] = efs['electricity_chem_kWh_d']/10
efs['electricity_other_kWh_MG'] = efs['electricity_other_kWh_d']/10
efs['natural_gas_MJ_MG'] = efs['natural_gas_MJ_d']/10
efs['natural_gas_chem_MJ_MG'] = efs['natural_gas_chem_MJ_d']/10
efs['natural_gas_other_MJ_MG'] = efs['natural_gas_other_MJ_d']/10
efs = efs[['electricity_kWh_MG','electricity_chem_kWh_MG','electricity_other_kWh_MG',
           'natural_gas_MJ_MG','natural_gas_chem_MJ_MG','natural_gas_other_MJ_MG']]

# no CHP, no electricity used for chemicals, and no natura gas for lagoons
efs.loc['LAGOON_AER'] = [1386, 0, 1386, 0, 0, 0]
efs.loc['LAGOON_ANAER'] = [660, 0, 660, 0, 0, 0]
efs.loc['LAGOON_FAC'] = [660, 0, 660, 0, 0, 0]

efs.sort_index(inplace=True)

#%% CH4 emission factors

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

# g CH4/m3, [2]
w_AD_CH4_ef = 12.770
wo_AD_wo_lagoon_CH4_ef = 3.767

# calculate CH4 EF for different treatment trains (kg CO2-eq/MG)
efs.loc[efs.index.isin(w_AD), 'CH4'] =\
    w_AD_CH4_ef*CH4_factor/1000*MG_2_m3
    
efs.loc[efs.index.isin(wo_AD_wo_lagoon), 'CH4'] =\
    wo_AD_wo_lagoon_CH4_ef*CH4_factor/1000*MG_2_m3

# CH4 for aerobic lagoon is 0, [3]
efs.loc[efs.index == 'LAGOON_AER','CH4'] = 0
# anaerobic shallow lagoon and facultative lagoon: 0.05 kg CH4/kg COD, COD = 400 mg/L, [3]
efs.loc[efs.index.isin(['LAGOON_ANAER','LAGOON_FAC']), 'CH4'] =\
    (400*0.05)*CH4_factor/1000*MG_2_m3

#%% N2O emission factors

# influent TN = 40 mg-N/L
# N2O EF, [4]
# organics removal
efs.loc[efs.index.str[0].isin(['B','C','D','O']), 'N2O'] =\
    40/1000000*1000*0.0027775513853397726*N_2_N2O*N2O_factor*MG_2_m3

# nitrification
efs.loc[efs.index.str[0].isin(['E','F']), 'N2O'] =\
    40/1000000*1000*0.013213536264442496*N_2_N2O*N2O_factor*MG_2_m3

# BNR
efs.loc[efs.index.str[0].isin(['G','H','I','N']), 'N2O'] =\
    40/1000000*1000*0.011239961834303045*N_2_N2O*N2O_factor*MG_2_m3

# EFs of N2O for lagoons, [3]
efs.loc[efs.index == 'LAGOON_AER','N2O'] = 40/1000000*1000*0.016*N_2_N2O*N2O_factor*MG_2_m3
efs.loc[efs.index == 'LAGOON_ANAER','N2O'] = 0
efs.loc[efs.index == 'LAGOON_FAC','N2O'] = 0

#%% non-combustion CO2

# kg CO2 eq/MG, [5]
efs['NC_CO2'] = 79.4936478

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

WWTP_TT['TT_IDENTIFIED'] = WWTP_TT[['B1','B1E','B2','B3','B4','B5','B6',
                                    'C1','C1E','C2','C3','C5','C6',
                                    'D1','D1E','D2','D3','D5','D6',
                                    'E2','E2P',
                                    'F1','F1E',
                                    'G1','G1E','G2','G3','G5','G6',
                                    'H1','H1E',
                                    'I1','I1E','I2','I3','I5','I6',
                                    'N1','N1E','N2',
                                    'O1','O1E','O2','O3','O5','O6',
                                    'LAGOON_AER','LAGOON_ANAER',
                                    'LAGOON_FAC','LAGOON_UNCATEGORIZED']].sum(axis=1)

assert (WWTP_TT['TT_IDENTIFIED'] >= 1).sum() == len(WWTP_TT)

non_continental = ['HI','VI','MP','GU','AK','AS','PR']

WWTP_TT = WWTP_TT[~WWTP_TT['STATE'].isin(non_continental)]

#%% add LAGOON_UNCATEGORIZED to efs

# flow weighted average of aerobic and (anaerobic+facultative) lagoons in the contiguous U.S.
L_AE_flow = (WWTP_TT['LAGOON_AER']/WWTP_TT['TT_IDENTIFIED']*WWTP_TT['FLOW_2022_MGD_FINAL']).sum()
L_AN_FA_flow = ((WWTP_TT['LAGOON_ANAER']+WWTP_TT['LAGOON_FAC'])/\
                WWTP_TT['TT_IDENTIFIED']*WWTP_TT['FLOW_2022_MGD_FINAL']).sum()

L_AE_flow_ratio = L_AE_flow/(L_AE_flow+L_AN_FA_flow)
L_AN_FA_flow_ratio = L_AN_FA_flow/(L_AE_flow+L_AN_FA_flow)

efs.loc['LAGOON_UNCATEGORIZED'] = efs.loc['LAGOON_AER']*L_AE_flow_ratio +\
    efs.loc['LAGOON_ANAER']*L_AN_FA_flow_ratio

#%% find TT codes used in WWTPs

final_code = np.intersect1d(WWTP_TT.columns.values, efs.index).copy()

# keep efs that are in final_code
efs = efs[efs.index.isin(final_code)]

# remove TTs that no WWTP is assined to
for i in efs.index:
    if WWTP_TT[i].sum() == 0:
        efs.drop(i, axis=0, inplace=True)
        index = np.argwhere(final_code==i)
        final_code = np.delete(final_code, index)

efs = efs.reindex(index=sorted(efs.index))

# make sure each TT occur up to once for every WWTP
assert WWTP_TT[final_code].max().max() == 1
assert WWTP_TT[final_code].min().min() == 0

#%% data preparation - NG and electricity

m3_plot = efs.copy()

# natural gas
m3_plot['NG_onsite'] = m3_plot['natural_gas_MJ_MG']*NG_combustion
m3_plot['NG_upstream'] = m3_plot['natural_gas_MJ_MG']*upstream_NG_GHG

# electricity
# balancing area
balnc_area = pd.read_excel('WWTP_baseline_trains_8.xlsx','Balance_Area')

# upstream electricity GHG data, convert to kg CO2 eq/MWh, [1]
upstream_elec_GHG = UEG = {# item          :   kg CO2e/MWh
                            'natural_gas'  :   24/1000*kWh_2_MJ*1000,
                            'coal'         :   18/1000*kWh_2_MJ*1000,
                            'nuclear'      :   1.9/1000*kWh_2_MJ*1000,
                            'wind'         :   2.86/1000*kWh_2_MJ*1000,
                            'solar'        :   10.48/1000*kWh_2_MJ*1000,
                            'biomass'      :   19.02/1000*kWh_2_MJ*1000,
                            'geothermal'   :   1.35/1000*kWh_2_MJ*1000,
                            'hydro'        :   2.08/1000*kWh_2_MJ*1000}

# balancing_area['co2_gen_mmt'] is the onsite emission at the power plant
balnc_area['CO2_kg_total'] = balnc_area['co2_gen_mmt']*(10**9) +\
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

balnc_area = balnc_area.loc[balnc_area['t'] == 2020]
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

for TT in m3_plot.index:
    elec_weighted = DescrStatsW(data=WWTP_TT[WWTP_TT[TT] != 0]['kg_CO2_kWh'],
                                weights=WWTP_TT[WWTP_TT[TT] != 0]['FLOW_2022_MGD_FINAL']/\
                                    WWTP_TT[WWTP_TT[TT] != 0]['TT_IDENTIFIED'])
    
    # national average electricity grid
    m3_plot.loc[m3_plot.index == TT, 'elec_average'] =\
        balnc_area['kg_CO2_kWh'].mean()*m3_plot.loc[m3_plot.index == TT, 'electricity_kWh_MG']
    m3_plot.loc[m3_plot.index == TT, 'elec_std'] =\
        balnc_area['kg_CO2_kWh'].std()*m3_plot.loc[m3_plot.index == TT, 'electricity_kWh_MG']
    
    # flow-weight average
    m3_plot.loc[m3_plot.index == TT, 'elec_weighted_average'] =\
        elec_weighted.mean*m3_plot.loc[m3_plot.index == TT, 'electricity_kWh_MG']
    m3_plot.loc[m3_plot.index == TT, 'elec_weighted_std'] =\
        elec_weighted.std*m3_plot.loc[m3_plot.index == TT, 'electricity_kWh_MG']

#%% data preparation - biosolids

# biosolids data in tonne/year
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

# surface disposal = landfill
# other_management = 50% landfilled + 50% land-applied
WWTP_TT.loc[WWTP_TT['Management Practice Type(s)'].notna(), 'landfill'] =\
    (WWTP_TT['Amount of Biosolids Managed - Surface Disposal'] +\
     WWTP_TT['Amount of Biosolids Managed - Other Management Practice']/2)*1000
        
WWTP_TT.loc[WWTP_TT['Management Practice Type(s)'].notna(), 'land_application'] =\
    (WWTP_TT['Amount of Biosolids Managed - Land Applied'] +\
     WWTP_TT['Amount of Biosolids Managed - Other Management Practice']/2)*1000
        
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

# biosolids production, [6]
biosolids_w_primary = 0.2636
biosolids_wo_primary = 0.131

# assume 60% VSS, 42.5% VSS reduction for AD and 47.5% for AeD, [7]
TSS_2_VSS = 0.6
reduction_AD = 0.425
reduction_AeD = 0.475
# use the flow weighted average for uncategorized lagoon
reduction_uncategorized = reduction_AeD*L_AE_flow_ratio + reduction_AD*L_AN_FA_flow_ratio

coefficient = WWTP_TT['FLOW_2022_MGD_FINAL']/WWTP_TT['TT_IDENTIFIED']

# theoretical biosolids amount in kg/year
WWTP_TT['theoretical_biosolids'] = (WWTP_TT[TT_w_primary_AD].sum(axis=1)*\
                                        coefficient*biosolids_w_primary*\
                                            (1-TSS_2_VSS*reduction_AD) +\
                                    WWTP_TT[TT_w_primary_AeD].sum(axis=1)*\
                                        coefficient*biosolids_w_primary*\
                                            (1-TSS_2_VSS*reduction_AeD) +\
                                    WWTP_TT[TT_w_primary_none].sum(axis=1)*\
                                        coefficient*biosolids_w_primary +\
                                    WWTP_TT[TT_wo_primary_AD].sum(axis=1)*\
                                        coefficient*biosolids_wo_primary*\
                                            (1-TSS_2_VSS*reduction_AD) +\
                                    WWTP_TT[TT_wo_primary_AeD].sum(axis=1)*\
                                        coefficient*biosolids_wo_primary*\
                                            (1-TSS_2_VSS*reduction_AeD) +\
                                    WWTP_TT[TT_wo_primary_L_u].sum(axis=1)*\
                                        coefficient*biosolids_wo_primary*\
                                            (1-TSS_2_VSS*reduction_uncategorized) +\
                                    WWTP_TT[TT_wo_primary_none].sum(axis=1)*\
                                        coefficient*biosolids_wo_primary)*MG_2_m3*365

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

WWTP_TT.loc[data_22_LF_LA, 'landfill'] = WWTP_TT['theoretical_biosolids']/2
WWTP_TT.loc[data_22_LF_LA, 'land_application'] = WWTP_TT['theoretical_biosolids']/2

data_22_LF_IN = (WWTP_TT['Management Practice Type(s)'].isna()) &\
                (WWTP_TT['LANDFILL'].notna()) &\
                (WWTP_TT['LAND_APP'].notna()) &\
                (WWTP_TT['FBI_y'].notna()) &\
                (WWTP_TT['MHI_y'].notna()) &\
                (WWTP_TT['LANDFILL'] != 0) &\
                (WWTP_TT['LAND_APP'] == 0) &\
                ((WWTP_TT['FBI_y'] != 0) |\
                 (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_LF_IN, 'landfill'] = WWTP_TT['theoretical_biosolids']/2
WWTP_TT.loc[data_22_LF_IN, 'incineration'] = WWTP_TT['theoretical_biosolids']/2

data_22_LA_IN = (WWTP_TT['Management Practice Type(s)'].isna()) &\
                (WWTP_TT['LANDFILL'].notna()) &\
                (WWTP_TT['LAND_APP'].notna()) &\
                (WWTP_TT['FBI_y'].notna()) &\
                (WWTP_TT['MHI_y'].notna()) &\
                (WWTP_TT['LANDFILL'] == 0) &\
                (WWTP_TT['LAND_APP'] != 0) &\
                ((WWTP_TT['FBI_y'] != 0) |\
                 (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_LA_IN, 'land_application'] = WWTP_TT['theoretical_biosolids']/2
WWTP_TT.loc[data_22_LA_IN, 'incineration'] = WWTP_TT['theoretical_biosolids']/2

data_22_all = (WWTP_TT['Management Practice Type(s)'].isna()) &\
              (WWTP_TT['LANDFILL'].notna()) &\
              (WWTP_TT['LAND_APP'].notna()) &\
              (WWTP_TT['FBI_y'].notna()) &\
              (WWTP_TT['MHI_y'].notna()) &\
              (WWTP_TT['LANDFILL'] != 0) &\
              (WWTP_TT['LAND_APP'] != 0) &\
              ((WWTP_TT['FBI_y'] != 0) |\
               (WWTP_TT['MHI_y'] != 0))

WWTP_TT.loc[data_22_all, 'landfill'] = WWTP_TT['theoretical_biosolids']/4
WWTP_TT.loc[data_22_all, 'land_application'] = WWTP_TT['theoretical_biosolids']/4
WWTP_TT.loc[data_22_all, 'incineration'] = WWTP_TT['theoretical_biosolids']/2

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
    WWTP_TT.loc[TT_disposal, 'landfill'] = WWTP_TT['theoretical_biosolids']/2
    WWTP_TT.loc[TT_disposal, 'land_application'] = WWTP_TT['theoretical_biosolids']/2

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
             coefficient*biosolids_w_primary +\
                 WWTP_TT[TT_wo_primary_IN].sum(axis=1)*\
                     coefficient*biosolids_wo_primary)*MG_2_m3*365

    assert (WWTP_TT.loc[TT_IN_disposal,'theoretical_biosolids']-\
            WWTP_TT.loc[TT_IN_disposal,'incineration']).min() > 0
    
    WWTP_TT.loc[TT_IN_disposal, 'landfill'] = (WWTP_TT['theoretical_biosolids']-\
                                               WWTP_TT['incineration'])/2
    WWTP_TT.loc[TT_IN_disposal, 'land_application'] = (WWTP_TT['theoretical_biosolids']-\
                                                       WWTP_TT['incineration'])/2

WWTP_TT.fillna({'landfill': 0}, inplace=True)
WWTP_TT.fillna({'land_application': 0}, inplace=True)
WWTP_TT.fillna({'incineration': 0}, inplace=True)

assert WWTP_TT[['landfill','land_application','incineration']].sum(axis=1).min() > 0

# 5.65 kg CH4 per tonne biosolids, [9]
LF_CH4 = 5.65/1000
# 0.05 kg N per kg biosolids land applied, [10]
LA_biosolids_N = 0.05
# 0.01 kg N2O-N/kg N, [3]
LA_N2O_N = 0.01

# emission from landfill (CH4) and land application (N2O) in kg CO2-eq/day
WWTP_TT['LF_CH4'] = WWTP_TT['landfill']/365*LF_CH4*CH4_factor
WWTP_TT['LA_N2O'] = WWTP_TT['land_application']/365*LA_biosolids_N*LA_N2O_N*N_2_N2O*N2O_factor

# for the per m3 plot, if no incineration,
# evenly divide the biosolids amount to landfill and land applicaton
for TT in TT_w_IN:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] = 0
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] = 0

for TT in TT_w_primary_AD:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_w_primary*(1-TSS_2_VSS*reduction_AD)/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_w_primary*(1-TSS_2_VSS*reduction_AD)/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

for TT in TT_w_primary_AeD:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_w_primary*(1-TSS_2_VSS*reduction_AeD)/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_w_primary*(1-TSS_2_VSS*reduction_AeD)/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

for TT in TT_w_primary_lime:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_w_primary/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_w_primary/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

# use LA_N2O_N_AD for 'LAGOON_ANAER' and 'LAGOON_FAC'
for TT in TT_wo_primary_AD:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_wo_primary*(1-TSS_2_VSS*reduction_AD)/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_wo_primary*(1-TSS_2_VSS*reduction_AD)/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

# use LA_N2O_N_AeD for 'LAGOON_AER'
for TT in TT_wo_primary_AeD:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_wo_primary*(1-TSS_2_VSS*reduction_AeD)/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_wo_primary*(1-TSS_2_VSS*reduction_AeD)/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

for TT in TT_wo_primary_L_u:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_wo_primary*(1-TSS_2_VSS*reduction_uncategorized)/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_wo_primary*(1-TSS_2_VSS*reduction_uncategorized)/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

for TT in TT_wo_primary_lime:
    m3_plot.loc[m3_plot.index == TT, 'biosolids_CH4'] =\
        biosolids_wo_primary/2*LF_CH4*CH4_factor
    m3_plot.loc[m3_plot.index == TT, 'biosolids_N2O'] =\
        biosolids_wo_primary/2*LA_biosolids_N*\
            LA_N2O_N*N_2_N2O*N2O_factor

m3_plot['biosolids_CH4'] *= MG_2_m3
m3_plot['biosolids_N2O'] *= MG_2_m3

#%% total emission

# distribute the flow rate to TTs for each WWTP
WWTP_TT_all = WWTP_TT.loc[:, final_code]
WWTP_TT_all = WWTP_TT_all.div(WWTP_TT['TT_IDENTIFIED'], axis=0)
WWTP_TT_all = WWTP_TT_all.mul(WWTP_TT['FLOW_2022_MGD_FINAL'], axis=0)

# electricity emission in kg CO2-eq/day
WWTP_TT['electricity_emission'] = (WWTP_TT_all @ efs['electricity_kWh_MG'])*WWTP_TT['kg_CO2_kWh']
WWTP_TT['electricity_emission'] = WWTP_TT['electricity_emission'].fillna(0)

# electricity emission in kg CO2-eq/day
WWTP_TT['onsite_NG_emission'] = (WWTP_TT_all @ efs['natural_gas_MJ_MG'])*NG_combustion
WWTP_TT['onsite_NG_emission'] = WWTP_TT['onsite_NG_emission'].fillna(0)
WWTP_TT['upstream_NG_emission'] = (WWTP_TT_all @ efs['natural_gas_MJ_MG'])*upstream_NG_GHG
WWTP_TT['upstream_NG_emission'] = WWTP_TT['upstream_NG_emission'].fillna(0)

# direct emission in kg CO2-eq/day
WWTP_TT['CH4_emission'] = WWTP_TT_all @ efs['CH4']
WWTP_TT['N2O_emission'] = WWTP_TT_all @ efs['N2O']
WWTP_TT['CO2_emission'] = WWTP_TT_all @ efs['NC_CO2']

print('\nonsite CH4: ' +\
      f'{(WWTP_TT["CH4_emission"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
print('\nonsite N2O: ' +\
      f'{(WWTP_TT["N2O_emission"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
print('\nonsite CO2: ' +\
      f'{(WWTP_TT["CO2_emission"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
print('\nlandfill CH4: ' +\
      f'{(WWTP_TT["LF_CH4"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
print('\nland application N2O: ' +\
      f'{(WWTP_TT["LA_N2O"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
print('\nelectricity: ' +\
      f'{(WWTP_TT["electricity_emission"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
print('\nonsite natural gas: ' +\
      f'{(WWTP_TT["onsite_NG_emission"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')

print('\nupstream natural gas: ' +\
      f'{(WWTP_TT["upstream_NG_emission"].sum()*365/(10**9)):.2f} MMT CO2-eq/year.')
    
total = (WWTP_TT[['CH4_emission','N2O_emission','CO2_emission',
                  'LF_CH4','LA_N2O','electricity_emission',
                  'onsite_NG_emission','upstream_NG_emission']].sum()).sum()

print(f'\ntotal: {(total*365/(10**9)):.2f} MMT CO2-eq/year.')