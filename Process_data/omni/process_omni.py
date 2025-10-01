from Download_data.omni import download_omni as dd_omni
import pandas as pd
import cdflib
import datetime
import numpy as np


############ Functions to read and process cdf files for omni data #############

def filter_metadata_OMNI(original_metadata):

    '''
    Filters and renames metadata keys from an input metadata dictionary using a
    predefined mapping. It constructs a new dictionary containing only the
    filtered and renamed metadata. It ensures that only keys listed in
    `relevant_keys` are included in the filtered metadata. If a key from the
    relevant set is missing in the original metadata, its corresponding value
    in the returned dictionary will be `None`.

    Args:
        - original_metadata (dict): A dictionary containing the original metadata
          with various key-value pairs.

    Returns:
        - filtered_metadata (dict): A dictionary containing only the relevant metadata
          with keys renamed.
    '''

    relevant_keys = ['FIELDNAM', 'CATDESC', 'VALIDMIN', 'VALIDMAX', 'FILLVAL', 'UNITS', 'VAR_TYPE', 'VAR_NOTES', 'DEPEND_0']
    rename_mapping = {
        'FIELDNAM': 'var_name',
        'CATDESC': 'description',
        'VALIDMIN': 'min_valid',
        'VALIDMAX': 'max_valid',
        'FILLVAL': 'fill_value',
        'UNITS': 'units',
        'VAR_TYPE': 'var_type',
        'VAR_NOTES': 'var_notes',
        'DEPEND_0': 'dependency'}

    filtered_metadata = {}

    for key in relevant_keys:
        renamed_key = rename_mapping.get(key)
        try:
            value = original_metadata[key]
        except KeyError:
            value= None

        filtered_metadata[renamed_key] = value

    return filtered_metadata


def read_CDFfile_OMNI(cdf_path, relevant_var, rename_mapping):

    '''
    Load and parse an OMNI CDF file to extract selected variables and their
    metadata into a structured format for analysis.

    Args:
        - cdf_path (str): The path to the CDF file to be loaded.
        - relevant_var (list): A list of variable names (str) to extract from the
          CDF file (e.g., 'KP', 'Epoch'). These should match the variable names in
          the CDF file.

    Returns:
        - list: A list containing the following elements:
            1. df_cdf (pandas.DataFrame): A DataFrame containing scalar variables.
            2. dict_cdf_metadata (dict): A dictionary with metadata for the extracted
               variables, filtered and renamed using  the `filter_metadata_OMNI`
               function.

    Raises:
        - KeyError: If any of the variables in `relevant_var` are not found in
          the CDF file.
    '''

    cdf = cdflib.CDF(cdf_path)
    dict_cdf = {}
    dict_cdf_metadata = {}

#    variables = cdf.cdf_info().zVariables
#    if len(variables) == 0:
#        print('Resolution de 1h, hay que usar rVariables')
#        variables = cdf.cdf_info().rVariables
#    print('cdf variables', variables)

    #Esto se puede hacer altiro como dataframe
    for var in relevant_var:
        var_metadata = cdf.varattsget(var)
        filtered_metadata = filter_metadata_OMNI(var_metadata)

        renamed_key = rename_mapping.get(var)
        dict_cdf_metadata[renamed_key] = filtered_metadata
        dict_cdf[renamed_key] = cdf[var]

    df_cdf = pd.DataFrame(dict_cdf)

    if 'Epoch' in relevant_var:
#        print('epoch')
        df_cdf['epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(df_cdf['epoch'].values))

    return [df_cdf, dict_cdf_metadata]


def clean_CDFfile_OMNI(df, dict_metadata, interp=False):

    '''
    Clean OMNI data by replacing fill values for missing data (as specified in
    the metadata) with NaN values. It can also linearly interpolate the NaN values
    in the DataFrame 'df' if specified.

    Args:
        - df (pandas.DataFrame): The input DataFrame containing scalar OMNI data.
        - dict_metadata (dict): A dictionary containing metadata for the variables in `df`,
          including fill values and valid ranges.
        - interp (bool, optional): If True, applies linear interpolation to the
          NaN values in the DataFrame `df`. Default is False.

    Returns:
        - df_clean (pandas.DataFrame): The cleaned DataFrame, with fill values replaced by NaN and
          optionally interpolated if `interp=True`.
    '''

    df_clean = pd.DataFrame()

    for var in df.columns:
        fill_value = dict_metadata[var]['fill_value']
        min_valid = dict_metadata[var]['min_valid']
        max_valid = dict_metadata[var]['max_valid']

        df_clean[var] = df[var].replace(fill_value, np.nan)
#        df_clean.loc[df_clean[var] > max_valid, var] = np.nan
#        df_clean.loc[df_clean[var] < min_valid, var] = np.nan
    if interp:
        print('Interpolating bad OMNI data')
        df = df.interpolate(axis=0)

    return df_clean


def load_CDFfiles_OMNI(start_date, end_date, local_root_dir, relevant_var, rename_mapping, res, type):

    '''
    Load and process OMNI CDF files for a specified date range, extracting selected variables
    and associated metadata. The cleaned and processed data is returned for further analysis.

    Args:
        - start_date (datetime.date): The start date of the range to process.
        - end_date (datetime.date): The end date of the range to process.
        - local_root_dir (str): Path to the local directory containing the CDF files.
        - relevant_var (list): List of variable names to extract from the CDF files.
        - res (str): The time resolution of the data file ('1h', '5min' or '1min')
        - typ (str): The type of OMNI data file to process (hro or hro2).

    Returns:
        - tuple: A tuple containing:
            1. df_omni (pandas.DataFrame): A DataFrame with cleaned scalar parameters f
               or the selected variables.
            2. dict_omni_metadata (dict): A dictionary containing the metadata
               for the scalar parameters.
    '''

    print(f'\nPROCESSING OMNI {res.upper()} {type.upper()} DATA')

    if res != "1h":
#        print(res, ' resolution')
        date_array = pd.date_range(start=start_date, end=end_date, freq='MS')
    else:
#        print('1h resolution')
        date_array = pd.date_range(start=start_date, end=end_date, freq='6MS')

    flag = True
    for date in date_array:
        filename = dd_omni.get_filename_OMNI(date, res, type)
        local_dir = dd_omni.get_local_dir_OMNI(date, local_root_dir, res, type)
        path = local_dir + filename
        df, dict_omni_metadata = read_CDFfile_OMNI(path, relevant_var, rename_mapping)
        df_clean = clean_CDFfile_OMNI(df, dict_omni_metadata)

        if flag:
            df_omni = df_clean.copy()
            flag=False
        else:
            df_omni = pd.concat([df_omni, df_clean])

    df_omni = df_omni.reset_index(drop=True)

    print('----')
    print("DONE")
    print('----')

    return df_omni, dict_omni_metadata
