"""
helper functions for data processing

"""

"""
Read data

Examples
--------
from prepare_ml_dataset import prepare_ml_dataset 

x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_ml_dataset('972237', 'o')

x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_ml_dataset('972237', 'o', recalc = False, plot_data = False, save_data = True, release = 'rel05', dL01=True, average_time = 300, coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], feature_names=[ 'scaled_symh', 'scaled_ae','scaled_asyh', 'scaled_asyd'], forecast = True, number_history = 7)

"""


import os
import numpy as np
import pandas as pd
import math
# import swifter
# import ml_wrappers
from time_string import time_string
# import warnings
import fnmatch
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import re

import plot_functions
import prepare_fulldata
import initialize_var

random_seed = 42
np.random.seed(random_seed)
np.set_printoptions(precision=4)

# def convert_data_to_dL01(l):
#     n_l = len(l)
#     l10 = np.floor(l*10)
#     mask = np.full(n_l, True)
#     l_pre=100    
#     for i in range(n_l):
#         if l10[i] != l_pre:
#             mask[i] = False
#             l_pre = l10[i]
#     return(mask)

def read_csv_data(data_dir, release):
    df_data = pd.DataFrame()
    probes = ['a','b']
    
    for iprobe in probes:
        print("Reading csv data for probe" + iprobe, end="\r")
        df = pd.read_csv(data_dir + 'rbsp' + iprobe.capitalize() + '_data_' + release + '_fulldata.csv')  
        df['probe'] = iprobe
        if iprobe == probes[0]:
            df_data = df
        else:
            df_data = pd.concat([df_data, df], ignore_index=True)           

    df_data['Datetime'] = df_data['time'].swifter.apply(time_string)

    return df_data
 
def get_good_index(df_data, data_settings, dL01, forecast):
    """
    There are non valid data in the situ plasma, geomagnetic indexes data and solar wind data. Sometimes, the solar wind and index data are pre-processed (interpolated etc.) We need data with no NaN or Inf for the model. Indexes of valid data are created. 
    
    We have previousely reviewed that all coordinates data and all indexes data do not have NaN or Inf data. If solar wind parameters are used, we need to add index_good_sw into the final good index.

    """
    
    index_good_coor = (df_data['l'] > data_settings["l_min"]) & (df_data['l'] > data_settings["l_max"]) 
    index_good_rel05 = ((df_data['Datetime'] < '2017-10-29') | (df_data['Datetime'] > '2017-11-01')) 
    index_good_y = np.isfinite(df_data[data_settings["y_name"]]) # we take out 0 measurement data because we are using the log

    index_good = index_good_coor & index_good_rel05 & index_good_y

    for feature_name in set(data_settings["feature_names"]):
        index_good = index_good & np.isfinite(df_data[feature_name])
    
    if 'ae' in data_settings["raw_feature_names"]:
        index_good = index_good & (df_data['ae'] > 0)
    
    if 'al' in data_settings["raw_feature_names"]:
        index_good = index_good & (df_data['al'] > 0) 
    
    if dL01:
        index_good = index_good & get_dL01_mask(df_data)
    
    if forecast:
        remove_features_by_time(data_settings["feature_history_names"], '*_0h')
        
    return index_good, data_settings

def get_dL01_mask(df_data):
    l10 = df_data["l"].swifter.apply(lambda x: np.floor(x*10))
    l10_pre = np.append(0,np.array(df_data.loc[0:(df_data.shape[0]-2), "l10"]))
    index_mask = l10 == l10_pre
    
    return index_mask

def remove_features_by_time(feature_history_names, pattern):
    matching_feature_names = fnmatch.filter(feature_history_names, pattern)
    
    for ifeature_name in matching_feature_names:
        feature_history_names.remove(ifeature_name)
    return feature_history_names
        
def find_matching_strings_in_columns(df, pattern):
    """
    Finds all matching strings in the columns of a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to search within.
        pattern (str): The regular expression pattern to search for.

    Returns:
        dict: A dictionary where keys are column names and values are lists of matching strings.
             If no match is found in a column, the value will be an empty list.
    """
    matching_strings = {}
    for col in df.columns:
        matches = re.findall(pattern, col)
        matching_strings[col] = matches
    return matching_strings    
  
def create_ml_data(df_data, index_train, index_valid,index_test, y_name, coor_names, history_feature_names):
    y_train = np.array(df_data.loc[index_train, y_name],dtype='float')
    y_valid = np.array(df_data.loc[index_valid, y_name],dtype='float')
    y_test  = np.array(df_data.loc[index_test, y_name],dtype='float')

    x_train = np.array(df_data.loc[index_train, coor_names + history_feature_names], dtype='float')
    x_valid = np.array(df_data.loc[index_valid, coor_names + history_feature_names], dtype='float')
    x_test  = np.array(df_data.loc[index_test,  coor_names + history_feature_names], dtype='float')

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def create_ml_indexes(df_data, test_ts, test_te, index_good, train_size=0.8):
    """
    Functions for create train and validation data set with a given test data set the function keeps the "episode time" to 2 days

    Args:
        df_data (df): _description_
        test_ts (time string): _description_
        test_te (time string): _description_
        index_good (index): _description_
        train_size (float, optional): _description_. Defaults to 0.8.

    Returns:
        index_train: index of df_data for training data
        index_valid: index of df_data for validation data
        index_test: index of df_data for test data
    """
    
    index_test = (df_data['Datetime'] >= test_ts ) & (df_data['Datetime'] <= test_te ) & index_good

    #If the test set is randomly split
    #episode_train_full,episode_test=train_test_split(episodes, test_size=0.01, train_size=1.0, random_state=42)
    #episode_train,episode_valid=train_test_split(episode_train_full, test_size=0.2, train_size=0.8, random_state=42)
    
    t0 = min(df_data['time'])
    # t1 = max(df_data['time'])

    episode_time= 86400.0*2 # 2 days
    
    # N_episode = np.ceil((t1-t0)/episode_time).astype(int)
    
    df_data['episodes'] = df_data['time'].apply(lambda x: math.floor((x-t0)/episode_time))

    # episode_test = np.array(df_data.loc[index_test,'episodes'])

    episode_train, episode_valid = train_test_split(np.unique(df_data.loc[index_good & ~index_test,'episodes']), test_size=1-train_size, train_size=train_size, random_state=42)

    episode_train = np.array(episode_train)
    episode_valid = np.array(episode_valid)
    
    index_train = index_good & df_data.loc[:,'episodes'].apply(lambda x: x in episode_train) & ~index_test
    index_valid = index_good & df_data.loc[:,'episodes'].apply(lambda x: x in episode_valid) & ~index_test

    np.set_printoptions(precision=3,suppress=True)
    print(sum(index_train), sum(index_valid), sum(index_test))   
    
    return index_train, index_valid, index_test

def save_csv_data(x_train, x_valid, x_test, y_train, y_valid, y_test, dataset_csv):
    pd.DataFrame(x_train).to_csv(dataset_csv["x_train"], index=False) 
    pd.DataFrame(y_train).to_csv(dataset_csv["y_train"], index=False) 
    pd.DataFrame(x_valid).to_csv(dataset_csv["x_valid"], index=False) 
    pd.DataFrame(y_valid).to_csv(dataset_csv["y_valid"], index=False) 
    pd.DataFrame(x_test).to_csv(dataset_csv["x_test"], index=False) 
    pd.DataFrame(y_test).to_csv(dataset_csv["y_test"], index=False) 

def load_csv_data(dataset_csv):
    x_train = pd.read_csv(dataset_csv["x_train"], index_col=False)
    y_train = pd.read_csv(dataset_csv["y_train"], index_col=False)
    x_valid = pd.read_csv(dataset_csv["x_valid"], index_col=False)
    y_valid = pd.read_csv(dataset_csv["y_valid"], index_col=False)
    x_test = pd.read_csv(dataset_csv["x_test"], index_col=False)
    y_test = pd.read_csv(dataset_csv["y_test"], index_col=False)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def load_test_data(dataset_csv):
    x_test = pd.read_csv(dataset_csv["x_test"], index_col=False)
    y_test = pd.read_csv(dataset_csv["y_test"], index_col=False)
    
    return x_test, y_test

def load_training_data(dataset_csv):
    x_train = pd.read_csv(dataset_csv["x_train"], index_col=False)
    y_train = pd.read_csv(dataset_csv["y_train"], index_col=False)
    x_valid = pd.read_csv(dataset_csv["x_valid"], index_col=False)
    y_valid = pd.read_csv(dataset_csv["y_valid"], index_col=False)

    return x_train, x_valid, y_train, y_valid

def print_model(self):
    print(self.data_settings)

def plot_plasma_data(df_data, index_good,y_name,y_name_log, data_view_dir):        
    time_array = df_data.loc[index_good,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)
    
    plot_functions.view_data(df_data, index_good, [y_name,y_name_log], [y_name,y_name_log], time_array, figname = data_view_dir + 'rbsp_'+y_name)

def plot_index_data(df_data, index_good, data_view_dir):
    time_array = df_data.loc[index_good,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)

    plot_functions.view_data(df_data, index_good, ['mlt',"cos0",'sin0','l','scaled_l','lat','scaled_lat'], ['MLT','cos theta','sin theta','L','scaled L','LAT','scaled LAT'], time_array, figname = data_view_dir + 'coor')
    
def plot_sw_data(df_data, index_good, data_view_dir):
    time_array = df_data.loc[index_good,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)

    plot_functions.view_data(df_data, index_good, ['swp',"scaled_swp",'swn','scaled_swn','swv','scaled_swv','by','scaled_by',"bz","scaled_bz"], ['SW P','scaled SW P','SW N','scaled SW N','SW V','scaled SW V','IMF By','scaled IMF By','IMF Bz','scaled IMF Bz'], time_array, figname = data_view_dir + 'sw')

def save_df_data(df_data, index_good, index_train, index_valid, index_test, dataset_csv):
            
    df_data["index_good"] = index_good
    df_data["index_train"] = index_train
    df_data["index_valid"] = index_valid
    df_data["index_test"] = index_test

    df_data.to_csv(dataset_csv["df_data"], index=False)
    return True

def prepare_ml_dataset(energy, species, recalc = False, plot_data = False, save_data = True, dL01=True, average_time = 300, raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'],  forecast = False, number_history = 7, test_ts = '2017-01-01', test_te = '2018-01-01'):

    feature_names = ["scaled_" + s for s in raw_feature_names]
    # y_name = initialize_var.create_y_name(energy, species)
    # log_y_name = initialize_var.create_log_y_name(energy, species)
    
    dataset_csv, data_settings = initialize_var.initialize_data_var(energy=energy, species=species, raw_feature_names = raw_feature_names, forecast = forecast, number_history = number_history, test_ts=test_ts, test_te=test_te, dL01=dL01)
        
    if os.path.exists(dataset_csv["x_train"]) & (recalc != True):
        x_train, x_valid, x_test, y_train, y_valid, y_test  = load_csv_data(dataset_csv)
    else:
        df_data, directories, fulldataset_csv, fulldata_settings = prepare_fulldata.load_fulldata(energy, species, recalc = recalc, average_time = average_time, raw_feature_names = raw_feature_names, number_history = number_history, save_data = save_data, plot_data = plot_data)

        index_good = get_good_index(df_data, data_settings, dL01, forecast)
            
        #set test set. Here we use one year (2017) of data for test set 
        index_train, index_valid, index_test = create_ml_indexes(df_data,  data_settings["test_ts"], data_settings["test_te"] , index_good)
        
        # Each round, one can only train one y. If train more than one y, need to  repeat from here
        x_train, x_valid, x_test, y_train, y_valid, y_test = create_ml_data(df_data, index_train, index_valid, index_test, data_settings["y_name_log"], data_settings["coor_names"], data_settings["feature_history_names"])  
        
        print("shapes of x_train, x_valid, x_test, y_train, y_valid, y_test ")
        print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape, y_test.shape)

        if plot_data:
            plot_plasma_data(df_data, index_good, data_settings["y_name"], data_settings["y_name_log"], directories["data_view_dir"])
            
            plot_index_data(df_data, index_good, directories["data_view_dir"])
            
            plot_sw_data(df_data, index_good, directories["data_view_dir"])
        
        if save_data:
            save_df_data(df_data, index_good, index_train, index_valid, index_test, dataset_csv)

            save_csv_data(x_train, x_valid, x_test, y_train, y_valid, y_test , dataset_csv)

    return x_train, x_valid, x_test, y_train, y_valid, y_test       

def __main__():
    if __name__ == "__name__":
        prepare_ml_dataset(['972237'], ['h'], recalc = False)

