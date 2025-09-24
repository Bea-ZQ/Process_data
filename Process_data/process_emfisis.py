import pandas as pd
import cdflib
import datetime
import numpy as np
import glob
import pathlib
from Download_data import download_emfisis as dd_emf
import os
import matplotlib.pyplot as plt
#  Data is not scientifically valid when the calibration is active
# METtime is corrected s/c clock time marking start of measurement.
# Bx, By, Bz, Bmag are magnetic field values in corresponding coordinate system.
# Bmag is average of all underlying |B| values within the averaging interval.
# Brms is the total standard deviation for measurements within the averaging interval.
# delta, lambda are latitude and longitude of the field vector  where (0,0) is the positive Bx direction.
# X, Y, Z is spacecraft location in corresponding coordinate system.
# Badpoint value for fill or otherwise removed measurements is -99999.9 
# ir is range (0-3), iC is Calibration current (1=on), iM is MAG data invalid (iM=1), and if is fill (if=1) flags
#calstate, magInvalid y magFill son flags de calidad indican si el dato es valido (0) o no (1), ya sea por estar calibrando, otros errores o es un dato de relleno


def get_local_filepath_EMFISIS(date, local_root_dir, probe,level = '3',coordinates = 'geo', interval= '4'):

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


    filename = f"rbsp-{probe}_magnetometer_{interval}sec-{coordinates}_emfisis-l{level}_{date.strftime('%Y%m%d')}_v*.cdf"

    # rbsp-a_magnetometer_4sec-geo_emfisis-l3_20140226_v1.3.2.cdf

    local_dir = dd_emf.get_local_dir_EMFISIS(date, local_root_dir, probe,level)
    filepath = glob.glob(os.path.join(local_dir , filename))[0]
#    print('LOCAL PATH', filepath)

    return filepath

def filter_metadata_EMFISIS(original_metadata):

    '''
    Filters and renames metadata keys from an input metadata dictionary using a
    predefined mapping. It constructs a new dictionary containing only the
    filtered and renamed metadata. It ensures that only keys listed in
    relevant_keys are included in the filtered metadata. If a key from the
    relevant set is missing in the original metadata, its corresponding value
    in the returned dictionary will be None.

    Args:
        - original_metadata (dict): A dictionary containing the original metadata
          with various key-value pairs.

    Returns:
        - filtered_metadata (dict): A dictionary containing only the relevant metadata
          with keys renamed.
    '''


    relevant_keys = ['CATDESC', 'FIELDNAM', 'FILLVAL', 'LABLAXIS', 'UNITS', 'VALIDMIN', 'VALIDMAX', 'VAR_TYPE', 'SCALETYP', 'MONOTON', 'TIME_BASE']
    rename_mapping = {
        'CATDESC': 'desc',
        'FIELDNAM': 'var_name',
        'FILLVAL': 'fill_value',
        'LABLAXIS': 'axis_name',
        'UNITS': 'units',
        'VALIDMIN': 'min_value',
        'VALIDMAX': 'max_value',
        'VAR_TYPE': 'var_type',
        'SCALETYP': 'scale',
        'MONOTON': 'Mon_increase',
        'TIME_BASE': 'time_base?'}

    filtered_metadata = {}

    for key in relevant_keys:
        renamed_key = rename_mapping.get(key)
        try:
            value = original_metadata[key]
        except KeyError:
            value= None

        filtered_metadata[renamed_key] = value


    return pd.DataFrame.from_dict(filtered_metadata, orient='index')


def read_CDFfile_EMFISIS(cdf_path, relevant_var, rename_mapping):

    '''
    Load and parse a RBSP ECT CDF file.

    This function reads and processes a CDF file containing data from the RBSP
    EMFISIS suite. It extracts relevant variables, metadata, and key flux data arrays
    into a structured format for analysis. For the 'Position' variable, each
    spatial dimension is stored as a separate key in the output DataFrame and metadata.

    Args:
        - cdf_path (str): The path to the CDF file to be loaded.
        - relevant_var (list): A list of variable names (str) to extract from the
          CDF file (e.g., 'Position', 'Epoch').

    Returns:
        - tuple: A tuple containing two elements:
            1. Data and Metadata for Relevant 1D Variables:
                - df_cdf (pandas.DataFrame): A DataFrame containing scalar parameters
                - dict_cdf_metadata (dict): A dictionary with metadata for the extracted
                  variables, filtered and renamed using filter_metadata_ECT.
    Raises:
        - KeyError: If any of the variables in relevant_var are not found in the CDF file.
    '''

    cdf = cdflib.CDF(cdf_path)
    dict_cdf = {}
    dict_cdf_metadata = {}
    rename_mapping['magInvalid'] = 'is valid?'
    rename_mapping['calState'] = 'calibrating?'
    rename_mapping['magFill'] = 'did fill?'
    rename_mapping['Magnitude'] = '|B|'
    Obligatorios = ['magFill','magInvalid', 'calState','Magnitude']
    relevant_var.extend(Obligatorios)
    #Esto se puede hacer altiro como dataframe
    for var in relevant_var:
        var_metadata = cdf.varattsget(var)
        filtered_metadata = filter_metadata_EMFISIS(var_metadata)

        renamed_key = rename_mapping.get(var)

        if var == 'coordinates':
            dict_cdf_metadata[renamed_key+'1'] = filtered_metadata
            dict_cdf_metadata[renamed_key+'2'] = filtered_metadata
            dict_cdf_metadata[renamed_key+'3'] = filtered_metadata

            dict_cdf[renamed_key+'1'] = cdf[var][:,0]
            dict_cdf[renamed_key+'2'] = cdf[var][:,1]
            dict_cdf[renamed_key+'3'] = cdf[var][:,2]
        elif var == 'Mag':
            dict_cdf_metadata[renamed_key+'-x1'] = filtered_metadata
            dict_cdf_metadata[renamed_key+'-x2'] = filtered_metadata
            dict_cdf_metadata[renamed_key+'-x3'] = filtered_metadata

            dict_cdf[renamed_key+'-x1'] = cdf[var][:,0]
            dict_cdf[renamed_key+'-x2'] = cdf[var][:,1]
            dict_cdf[renamed_key+'-x3'] = cdf[var][:,2]
            
        else:
            dict_cdf_metadata[renamed_key] = filtered_metadata
            dict_cdf[renamed_key] = cdf[var]

    df_cdf = pd.DataFrame(dict_cdf)

    if 'Epoch' in relevant_var:
#        print('epoch')
        df_cdf['epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(df_cdf['epoch'].values))


    return [df_cdf, dict_cdf_metadata]


def clean_CDFfile_EMFISIS(df,interp=False):
    #cleans the data from error value replacing with nan if:
    #    -it's a fill value
    #    -it's a value taken while calibrating
    #    -it's an invalid value
    # All of this criteria are taken according to the meta data

    
   cols_to_check = ['did fill?', 'calibrating?', 'is valid?']

# Collect all columns ending with "x1", "x2", or "x3"
   suffixes = ("x1", "x2", "x3", "|B|")
   target_cols = [c for c in df.columns if str(c).endswith(suffixes)]

# Build mask
   mask = df[cols_to_check].eq(1).any(axis=1)

# Replace with NaN in those target columns
   df.loc[mask, target_cols] = np.nan

   if interp:
       df_clean = df.interpolate(axis = 0)
       return df_clean
   return df
   


def load_CDFfiles_EMFISIS(start_date, end_date, local_root_dir, relevant_var, rename_mapping, probe, level = '3',Interpol = False):

    '''
    Load and process RBSP EMFISIS CDF files for a specified date range and selected
    RBSP probes (A, B, or both). It extracts scalar parameters  and associated metadata. The cleaned and processed data is returned
    for further analysis.

    meta data means thing such as measure units, max min value, name of variable etc.
    Args:
        - start_date (datetime.date): The start date of the range to process (inclusive).
        - end_date (datetime.date): The end date of the range to process (inclusive).
        - local_root_dir (str): Path to the local directory containing the CDF files.
        - relevant_var (list): List of variable names to extract from the CDF files.
        - probe (str): The satellite identifier ('a', 'b', or 'both'). If 'both',
          data for both probes will be loaded..
        - level (str, optional): Data processing level ('2' or '3'). Default is '3'.

    Returns:
        - list: A list containing the processed data for the selected probes:
            - For each probe, the list includes:
                1. emfisis_info (pandas.DataFrame): A DataFrame with cleaned scalar parameters.
                2. emfisis_metadata (dict): Metadata for the scalar parameters in emfisis_info.
    '''

    print('\nPrint para ver si efectivamente se actualiza el paquete Proccess_data EMFISIS V3')

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
            filepath = get_local_filepath_EMFISIS(date, local_root_dir, p,'3')

            emfisis= read_CDFfile_EMFISIS(filepath, relevant_var, rename_mapping)

            emfisis_df, emfisis_metadata = emfisis
            
            emfisis_df_clean = clean_CDFfile_EMFISIS(emfisis_df,Interpol)

            if flag:
                emfisis_info = emfisis_df_clean.copy()
                flag=False

            else:
                emfisis_info = pd.concat([emfisis_info, emfisis_df_clean])

        emfisis_info = emfisis_info.reset_index(drop=True)
        output.append([emfisis_info, emfisis_metadata])

    print('----')
    print("DONE")
    print('----')

    return output
 