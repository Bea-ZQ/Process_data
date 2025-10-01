import pandas as pd
import cdflib
import datetime
import numpy as np
import glob
from Download_data.rbsp import download_ect as dd_ect


###################### Function to obtain local filenames ##########################

def get_local_filepath_ECT(date, local_root_dir, probe, instrument, level = '3'):

    '''
    Constructs and retrieves the full local file path for RBSP ECT data files
    based on the provided date, probe, instrument, and data level. It searches
    within the appropriate local directory and uses a wildcard pattern (*) to
    match versions of the file and retrieves the first file path that matches
    the pattern.

    Args:
        - date (datetime.date): The date for the data file.
        - local_root_dir (str): The root directory where the RBSP ECT data files
          are stored locally.
        - probe (str): The RBSP probe identifier ('a' or 'b').
        - instrument (str): The ECT instrument name ('mageis' or 'rept').
        - level (str, optional): The data processing level ('2' or '3').
          Defaults to '3'.

    Returns:
        - filepath (str): The full local file path of the RBSP ECT data file
          that matches the pattern.

    Raises:
        - IndexError: If no file matching the configuration date is found in
          the local directory.
    '''

    filename = f"rbsp{probe}_*_ect-{instrument}-*_{date.strftime('%Y%m%d')}_v*.cdf"

    local_dir = dd_ect.get_local_dir_ECT(date, local_root_dir, probe, instrument, level)
    # link de interés: https://docs.python.org/es/3/library/glob.html
#    print(glob.glob(local_dir + filename))
    try:
        filepath = glob.glob(local_dir + filename)[0]
#    print('LOCAL PATH', filepath)
    except:
        filepath = 0

    return filepath


############ Functions to read and process cdf files for rept data #############

def filter_metadata_ECT(original_metadata):

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

    relevant_keys = ['FIELDNAM', 'CATDESC', 'SCALETYP', 'VALIDMIN', 'VALIDMAX', 'FILLVAL','UNITS', 'VAR_TYPE', 'VAR_NOTES']
    rename_mapping = {
        'FIELDNAM': 'var_name',
        'CATDESC': 'desc',
        'SCALETYP': 'scale',
        'VALIDMIN': 'min_valid',
        'VALIDMAX': 'max_valid',
        'FILLVAL': 'fill_value',
        'UNITS': 'units',
        'VAR_TYPE': 'var_type',
        'VAR_NOTES': 'notes'}

    filtered_metadata = {}

    for key in relevant_keys:
        renamed_key = rename_mapping.get(key)
        try:
            value = original_metadata[key]
        except KeyError:
            value= None

        filtered_metadata[renamed_key] = value

    return filtered_metadata


def read_CDFfile_ECT(cdf_path, key, relevant_var, rename_mapping):

    '''
    Load and parse a RBSP ECT CDF file.

    This function reads and processes a CDF file containing data from the RBSP
    ECT suite. It extracts relevant variables, metadata, and key flux data arrays
    into a structured format for analysis. For the 'Position' variable, each
    spatial dimension is stored as a separate key in the output DataFrame and metadata.

    Args:
        - cdf_path (str): The path to the CDF file to be loaded.
        - key (str): The key flux variable to load. Currently, only 'fedu' is supported.
        - relevant_var (list): A list of variable names (str) to extract from the
          CDF file (e.g., 'Position', 'Epoch').

    Returns:
        - tuple: A tuple containing two elements:
            1. Data and Metadata for Relevant 1D Variables:
                - df_cdf (pandas.DataFrame): A DataFrame containing scalar parameters
                - dict_cdf_metadata (dict): A dictionary with metadata for the extracted
                  variables, filtered and renamed using `filter_metadata_ECT`.

            2. FEDU Data and Metadata (only if `key='fedu'`):
                - fedu_data (numpy.ndarray): A multidimensional NumPy array containing
                  FEDU data.
                - dict_fedu_metadata (dict): A dictionary containing important
                  metadata for the extracted fedu data,  filtered and renamed
                  using `filter_metadata_ECT`

    Raises:
        - KeyError: If any of the variables in `relevant_var` are not found in the CDF file.
    '''

    cdf = cdflib.CDF(cdf_path)
    dict_cdf = {}
    dict_cdf_metadata = {}

    #Esto se puede hacer altiro como dataframe
    for var in relevant_var:
        var_metadata = cdf.varattsget(var)
        filtered_metadata = filter_metadata_ECT(var_metadata)

        renamed_key = rename_mapping.get(var)

        if var == 'Position':
            dict_cdf_metadata[renamed_key+'1'] = filtered_metadata
            dict_cdf_metadata[renamed_key+'2'] = filtered_metadata
            dict_cdf_metadata[renamed_key+'3'] = filtered_metadata

            dict_cdf[renamed_key+'1'] = cdf[var][:,0]
            dict_cdf[renamed_key+'2'] = cdf[var][:,1]
            dict_cdf[renamed_key+'3'] = cdf[var][:,2]
        else:
            dict_cdf_metadata[renamed_key] = filtered_metadata
            dict_cdf[renamed_key] = cdf[var]

    df_cdf = pd.DataFrame(dict_cdf)

    if 'Epoch' in relevant_var:
#        print('epoch')
        df_cdf['epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(df_cdf['epoch'].values))
    if key == 'fedu':
        fedu_data = cdf['FEDU']
        fedu_metadata = cdf.varattsget('FEDU')
        dict_fedu_metadata = filter_metadata_ECT(fedu_metadata)
        dict_fedu_metadata['energy_values'] = cdf['FEDU_Energy']
        dict_fedu_metadata['energy_labels'] = cdf['FEDU_ENERGY_LABL'][0]
        dict_fedu_metadata['alpha_values'] = cdf['FEDU_Alpha']
        dict_fedu_metadata['alpha_labels'] = cdf['FEDU_PA_LABL'][0]

    return [df_cdf, dict_cdf_metadata], [fedu_data, dict_fedu_metadata]


def clean_CDFfile_ECT(df, dict_metadata, f, f_metadata, interp=False):

    '''
    Clean RBSP ECT data by replacing fill values for missing data (as specified
    in the metadata) with NaN values. It can also linearly interpolate the NaN
    values in the DataFrame 'df' if specified.

    Args:
        - df (pandas.DataFrame): The input DataFrame containing scalar RBSP ECT data.
        - dict_metadata (dict): A dictionary containing metadata for the variables in `df`,
          including fill values and valid ranges.
        - f (numpy.ndarray): A multidimensional array containing flux data.
        - f_metadata (dict): Metadata for the flux array `f`, including fill
          values and valid ranges.
        - interp (bool, optional): If True, applies linear interpolation to the
          NaN values in the DataFrame `df`. Default is False.

    Returns:
        - tuple: A tuple containing:
            1. df_clean (pandas.DataFrame): The cleaned DataFrame, where fill values
               are replaced with NaN, and optionally interpolated.
            2. f_clean (numpy.ndarray): The cleaned flux array, where fill values are
               replaced with NaN.
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
        print('Interpolating bad ECT data')
        df = df.interpolate(axis=0)

    # Estamos limpiando fedus
    fill_value = f_metadata['fill_value']
    min_valid = f_metadata['min_valid']
    max_valid = f_metadata['max_valid']

    f_clean = np.copy(f)
    f_clean[f_clean == fill_value] = np.nan
    f_clean[f_clean < min_valid] = np.nan
    return df_clean, f_clean


def load_CDFfiles_ECT(start_date, end_date, local_root_dir, relevant_var, rename_mapping, probe, instrument, level = '3', key='fedu'):

    '''
    Load and process RBSP ECT CDF files for a specified date range and selected
    RBSP probes (A, B, or both). It extracts scalar parameters, multi-dimensional
    flux data and associated metadata. The cleaned and processed data is returned
    for further analysis.

    Args:
        - start_date (datetime.date): The start date of the range to process (inclusive).
        - end_date (datetime.date): The end date of the range to process (inclusive).
        - local_root_dir (str): Path to the local directory containing the CDF files.
        - relevant_var (list): List of variable names to extract from the CDF files.
        - probe (str): The satellite identifier ('a', 'b', or 'both'). If 'both',
          data for both probes will be loaded.
        - instrument (str): The instrument name whose data will be processed
          ('rept' or 'mageis').
        - level (str, optional): Data processing level ('2' or '3'). Default is '3'.
        - key (str, optional): Key to specify the flux data to extract. Default is 'fedu'.
          Currently onle 'fedu' is supported.

    Returns:
        - list: A list containing the processed data for the selected probes:
            - For each probe, the list includes:
                1. ect_info (pandas.DataFrame): A DataFrame with cleaned scalar parameters.
                2. fedu (numpy.ndarray): A NumPy array with cleaned multi-dimensional FEDU data.
                3. ect_metadata (dict): Metadata for the scalar parameters in `ect_info`.
                4. fedu_metadata (dict): Metadata for the multi-dimensional FEDU data.
    '''

    print('\nPrint para ver si efectivamente se actualiza el paquete Proccess_data ECT V3')
    print(f'\nPROCESSING ECT-{instrument.upper()} INSTRUMENT DATA')

    date_array = pd.date_range(start=start_date, end=end_date, freq='D')
    output = []

    if probe=='both':
        probes_list = ['a', 'b']
    else:
        probes_list = [probe]

    print(probes_list)

    for p in probes_list:
        f = []
        flag = True
        print(p)

        for date in date_array:
            filepath = get_local_filepath_ECT(date, local_root_dir, p, instrument, '3')
            if filepath == 0:
                print('No file in local')
                continue
            ect, fedu = read_CDFfile_ECT(filepath, key, relevant_var, rename_mapping)

            ect_df, ect_metadata = ect
            fedu_data, fedu_metadata = fedu

            ect_df_clean, fedu_data_clean = clean_CDFfile_ECT(ect_df, ect_metadata, fedu_data, fedu_metadata)

            if flag:
                ect_info = ect_df_clean.copy()
                flag=False

            else:
                ect_info = pd.concat([ect_info, ect_df_clean])

            f.append(fedu_data_clean)

        ect_info = ect_info.reset_index(drop=True)
        fedu = np.concatenate(f, axis=0)

        output.append([ect_info, fedu, ect_metadata, fedu_metadata])

    print('----')
    print("DONE")
    print('----')

    return output
