# Functions used in baseline paper

# Author: Sahar H El Abbadi
# Start date: Sept 20 2024

############ Unit conversions ############

# Convert to billion m3/yr
def mgd_to_Bm3_yr(mgd):
    gallons_per_m3 = 264.172
    days_per_year = 365
    return mgd * days_per_year * (1 / gallons_per_m3) * (1 / 1000)


def kg_per_day_to_mmt_per_year(kg_co2_per_day):
    """
    Converts the given value in kilograms of CO₂-equivalent per day (kg CO₂-eq/day) to
    million metric tons of CO₂-equivalent per year (MMT CO₂-eq/year).

    Parameters:
    kg_co2_per_day (float): The CO₂-equivalent emissions in kilograms per day.

    Returns:
    float: The equivalent CO₂ emissions in million metric tons per year (MMT CO₂-eq/year).
    """

    # Step 1: Convert kilograms (kg) to metric tons.
    metric_tons_per_day = kg_co2_per_day / 1000

    # Step 2: Convert metric tons per day to metric tons per year.
    metric_tons_per_year = metric_tons_per_day * 365

    # Step 3: Convert metric tons to million metric tons (MMT).
    mmt_per_year = metric_tons_per_year / 1_000_000

    return mmt_per_year

############ Treatment train manipulation ############

# Calculate flow
def calc_key_stats(df, total_facilities):
    """ Calculate count fraction, flow fraction, and emissions fraction for a dataframe.
    Input dataframe must be a subset of tt_data"""

    df['count_fraction'] = df['count'] / total_facilities * 100
    df['flow_fraction'] = df['flow_Bm3yr'] / df['flow_Bm3yr'].sum() * 100
    df['emissions_fraction'] = df['total_emissions_annual_mmtCO2_per_year'] / df[
        'total_emissions_annual_mmtCO2_per_year'].sum() * 100
    return df

def filter_facilities_by_process(processes, df, match_all=False, print_tt=False, match_type='substring'):
    """
    Calculates the fraction of facilities containing any or all of the specified processes.

    Args:
    - processes (str or list of str): A single process or a list of processes to search for.
    - df (DataFrame): The dataframe containing facility data.
    - match_all (bool): If True, checks if all processes are present in each row ('AND' condition).
                        If False, checks if any process is present in each row ('OR' condition).
    - print_tt (bool): If True, prints detailed info on the unique TT entries.
    - match_type (str): Defines the matching logic.
                        Options are:
                          'substring' - Matches if the process is found as part of any item in the list.
                          'exact'     - Matches if the process exactly matches an item in the list.
                          'starts_with' - Matches if the process is a prefix (e.g., 'G3' matches 'G3' and 'G3-p').

    Returns:
    - num_facilities (int): Number of facilities containing the process(es).
    - fraction_of_facilities (float): Fraction of facilities containing the process(es).
    """

    total_facilities = len(df)

    # Convert `processes` to a list if a single string is provided
    if isinstance(processes, str):
        processes = [processes]

    # Define filter function based on `match_type` and `match_all` parameters
    if match_type == 'substring':
        if match_all:
            filter_func = lambda x: all(any(process in item for item in x) for process in processes) if isinstance(x, list) else False
        else:
            filter_func = lambda x: any(any(process in item for item in x) for process in processes) if isinstance(x, list) else False
    elif match_type == 'exact':
        if match_all:
            filter_func = lambda x: all(process in x for process in processes) if isinstance(x, list) else False
        else:
            filter_func = lambda x: any(process in x for process in processes) if isinstance(x, list) else False
    elif match_type == 'starts_with':
        if match_all:
            filter_func = lambda x: all(any(item.startswith(process) for item in x) for process in processes) if isinstance(x, list) else False
        else:
            filter_func = lambda x: any(any(item.startswith(process) for item in x) for process in processes) if isinstance(x, list) else False
    else:
        raise ValueError(f"Invalid match_type: {match_type}. Choose from 'substring', 'exact', or 'starts_with'.")

    # Apply the filter function to the DataFrame
    facilities_with_process = df[df['TT'].apply(filter_func)]

    return facilities_with_process

def calculate_facility_fraction_for_process(processes, df, match_all=False, print_tt=False, match_type='substring'):
    """
    Calculates the fraction of facilities containing any or all of the specified processes.

    Args:
    - processes (str or list of str): A single process or a list of processes to search for.
    - df (DataFrame): The dataframe containing facility data.
    - match_all (bool): If True, checks if all processes are present in each row ('AND' condition).
                        If False, checks if any process is present in each row ('OR' condition).
    - print_tt (bool): If True, prints detailed info on the unique TT entries.

    Returns:
    - num_facilities (int): Number of facilities containing the process(es).
    - fraction_of_facilities (float): Fraction of facilities containing the process(es).
    """

    ## TODO delete this commented out section if all code runs as expected
    #
    #
    # # Convert `processes` to a list if a single string is provided
    # if isinstance(processes, str):
    #     processes = [processes]
    #
    # # Define filter function based on `match_all` parameter
    # if match_all:
    #     # Check if **all** processes are present in any element of each row
    #     filter_func = lambda x: all(any(process in item for item in x) for process in processes) if isinstance(x,
    #                                                                                                            list) else False
    # else:
    #     # Check if **any** process is present in any element of each row
    #     filter_func = lambda x: any(any(process in item for item in x) for process in processes) if isinstance(x,
    #                                                                                                            list) else False

    total_facilities = len(df)

    # Apply the filter function to the DataFrame using filter_facilities_by_process
    facilities_with_process = filter_facilities_by_process(processes, df, match_all=match_all, print_tt=print_tt, match_type=match_type)
    num_facilities = len(facilities_with_process)
    fraction_of_facilities = num_facilities / total_facilities * 100

    # Get unique entries in the 'TT' column as lists (convert to tuples for counting)
    unique_tt_entries = facilities_with_process['TT'].apply(tuple).value_counts()

    # Print summary information
    print(f'Processes of interest: {processes} ({"ALL" if match_all else "ANY"})')
    print(f'Number of facilities: {num_facilities:,}')
    print(f'Fraction of total facilities: {fraction_of_facilities:.4n}%')

    if print_tt:
        print(f'Treatment trains containing {processes}:')
        for entry, count in unique_tt_entries.items():
            print(f'{entry}: {count:,}')

    return num_facilities, fraction_of_facilities


######### For working with the treatment train results spreadsheet #########
def combine_liquid_treatment(label):
    """ Create a label for liquid treatment train"""
    if 'A' in label: return 'A_combined'
    if 'B' in label: return 'B_combined'
    if 'C' in label: return 'C_combined'
    if 'D' in label: return 'D_combined'
    if 'E' in label: return 'E_combined'
    if 'F' in label: return 'F_combined'
    if 'G' in label: return 'G_combined'
    if 'L' in label: return 'L_combined'
    return label  # Leave other labels unchanged


# Combine solids treatment
def combine_solids_treatment(solids_label):
    """ Create a label for solid treatment train"""
    if '1' in solids_label: return '1'
    if '2' in solids_label: return '2'
    if '3' in solids_label: return '3'
    if '4' in solids_label: return '4'
    if '5' in solids_label: return '5'
    if '6' in solids_label: return '6'
    if solids_label in ['L-a', 'L-n', 'L-f', 'L-u']: return 'L-all'
    return solids_label


# Combine treatment trains that are the same but with and without primary treatment
def combine_primary(label):
    if label in ['A1', '*A1', '*A1e', 'A1e']: return '(*)A1(e)'
    if label in ['A3', '*A3']: return '(*)A3'
    if label in ['A4', '*A4']: return '(*)A4'
    if label in ['A5', '*A5']: return '(*)A5'
    if label in ['A6', '*A6']: return '(*)A6'
    if label in ['E3', '*E3']: return '(*)E3'
    # if label in ['L-a', 'L-n', 'L-f', 'L-u']: return 'L-all'
    return label

def combine_treatment_target(label):
    """ Create a label for liquid treatment train"""
    if any(x in label for x in ['A', 'B', 'C']):
        return 'organics removal'
    if any(x in label for x in ['D', 'E', 'F', 'G']):
        return 'nutrient removal'
    return label  # Leave other labels unchanged

# Define a custom aggregation function to concatenate labels
def concat_labels(labels):
    return ', '.join(labels)


def process_treatment_data(tt_data, combine_function, group_col_name, total_facilities, sort_by='count_fraction'):
    """
    Process treatment data by combining based on the specified treatment function, grouping, and sorting.

    Parameters:
    tt_data (DataFrame): The input data containing treatment train information.
    combine_function (function): Function used to combine treatment labels (e.g., combine_treatment_target, combine_primary).
    group_col_name (str): The name for the new combined group column.
    total_facilities (int): The total number of facilities for calculating key stats.
    sort_by (str): The column by which to sort the results. Default is 'count_fraction'.

    Returns:
    DataFrame: The processed and sorted treatment data.
    """
    # Create a new label that combines treatment trains
    tt_data[group_col_name] = tt_data['code'].apply(combine_function)

    # Group by the combined treatment trains and aggregate other columns
    aggregated_columns = {col: 'sum' for col in tt_data.columns if col not in ['code', group_col_name]}
    tt_grouped = tt_data.groupby(group_col_name).agg(aggregated_columns).reset_index()

    # Apply the label concatenation to store original treatment trains
    tt_grouped['labels_combined'] = tt_data.groupby(group_col_name)['code'].apply(concat_labels).values

    # Calculate fractional count, flow, and emissions fraction
    tt_grouped = calc_key_stats(tt_grouped, total_facilities)

    # Select relevant columns (can be customized based on needs)
    relevant_cols = [group_col_name, 'flow_Bm3yr', 'flow_fraction',
                     'total_emissions_annual_mmtCO2_per_year', 'emissions_fraction', 'labels_combined']

    # Sort by the specified column (count_fraction, flow_fraction, etc.)
    sorted_data = tt_grouped.sort_values(by=sort_by, ascending=False)

    # Display the sorted data (can be limited to top rows)
    display(sorted_data[relevant_cols].head(10))

    return sorted_data