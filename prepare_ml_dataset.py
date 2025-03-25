"""
helper functions for data processing

"""

"""
Read data

Examples
--------
from prepare_ml_dataset import prepare_ml_dataset 

x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_ml_dataset('972237', 'o')

"""


import os
import numpy as np
# import pandas as pd
import modin.pandas as pd

import math
# import swifter
# import ml_wrappers
# from time_string import time_string
# import warnings
import fnmatch
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import re

import plot_functions
import prepare_fulldata
import initialize_var

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

def get_dL01_mask(df_full):
    l10 = df_full["l"].swifter.apply(lambda x: np.floor(x*10))
    l10_pre = np.append(0,np.array(l10[0:(len(l10)-1)]))
    index_mask = l10 == l10_pre
    
    return index_mask
 
def get_good_index(df_full, data_settings, fulldata_settings):
    """
    There are non valid data in the situ plasma, geomagnetic indexes data and solar wind data. Sometimes, the solar wind and index data are pre-processed (interpolated etc.) We need data with no NaN or Inf for the model. Indexes of valid data are created. 
    
    We have previousely reviewed that all coordinates data and all indexes data do not have NaN or Inf data. If solar wind parameters are used, we need to add index_good_sw into the final good index.

    """
    print(df_full.shape[0])

    index_good_coor = (df_full['l'] > data_settings["l_min"]) & (df_full['l'] < data_settings["l_max"]) 
    
    index_good_rel05 = ((df_full[fulldata_settings["datetime_name"]] < '2017-10-29') | (df_full[fulldata_settings["datetime_name"]] > '2017-11-01')) 
    
    index_good_y = np.isfinite(df_full[data_settings["y_name"]]) # we take out 0 measurement data because we are using the log
    # print(sum(index_good_coor))
    # print(sum(index_good_rel05))
    # print(sum(index_good_y))
    index_good = index_good_coor & index_good_rel05 & index_good_y
    # print(sum(index_good))
    for raw_feature_name in set(data_settings["raw_feature_names"]):
        index_good = index_good & np.isfinite(df_full[raw_feature_name])
    # print(sum(index_good))
    if data_settings["dL01"]:
        index_good = index_good & get_dL01_mask(df_full)
    
    return index_good

def match_feature_by_time(feature_history_names, pattern):
    return fnmatch.filter(feature_history_names, pattern)

def remove_index_features_by_time(feature_history_names, pattern):
    matching_feature_names = fnmatch.filter(feature_history_names, pattern)
    for ifeature_name in matching_feature_names:
        if ifeature_name in ['symh','asyh','asyd','ae','kp']:
            feature_history_names.remove(ifeature_name)  
    return feature_history_names 

def remove_features_by_time(feature_history_names, pattern):
    matching_feature_names = fnmatch.filter(feature_history_names, pattern)
    for ifeature_name in matching_feature_names:
        feature_history_names.remove(ifeature_name)  
    return feature_history_names 

def create_ml_data(df_data, index_train, index_valid,index_test, y_name, coor_names, history_feature_names):
    y_train = np.array(df_data.loc[index_train, y_name],dtype='float')
    y_valid = np.array(df_data.loc[index_valid, y_name],dtype='float')
    y_test  = np.array(df_data.loc[index_test, y_name],dtype='float')

    x_train = np.array(df_data.loc[index_train, coor_names + history_feature_names], dtype='float')
    x_valid = np.array(df_data.loc[index_valid, coor_names + history_feature_names], dtype='float')
    x_test  = np.array(df_data.loc[index_test,  coor_names + history_feature_names], dtype='float')

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def create_ml_indexes(df_data, data_settings, test_ts, test_te, train_size=0.8):
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
    
    
    random_seed = 42
    np.random.seed(random_seed)
    
    index_test = (df_data[data_settings['datetime_name']] >= test_ts ) & (df_data[data_settings['datetime_name']] <= test_te )# & index_good

    #If the test set is randomly split
    #episode_train_full,episode_test=train_test_split(episodes, test_size=0.01, train_size=1.0, random_state=42)
    #episode_train,episode_valid=train_test_split(episode_train_full, test_size=0.2, train_size=0.8, random_state=42)
    
    t0 = min(df_data['time'])
    # t1 = max(df_data['time'])

    episode_time= 86400.0*2 # 2 days
    
    # N_episode = np.ceil((t1-t0)/episode_time).astype(int)
    
    df_data['episodes'] = df_data['time'].apply(lambda x: math.floor((x-t0)/episode_time))

    # episode_train, episode_valid = train_test_split(np.unique(df_data.loc[index_good & ~index_test,'episodes']), test_size=1-train_size, train_size=train_size, random_state=42)

    episode_train, episode_valid = train_test_split(np.unique(df_data.loc[~index_test,'episodes']), test_size=1-train_size, train_size=train_size, random_state=42)

    episode_train = np.array(episode_train)
    episode_valid = np.array(episode_valid)
    
    index_train = df_data.loc[:,'episodes'].apply(lambda x: x in episode_train) & ~index_test # & index_good
    index_valid = df_data.loc[:,'episodes'].apply(lambda x: x in episode_valid) & ~index_test  #& index_good

    np.set_printoptions(precision=3,suppress=True)
    print(sum(index_train), sum(index_valid), sum(index_test))   
    
    return index_train, index_valid, index_test

def save_csv_data(x_train, x_valid, x_test, y_train, y_valid, y_test, dataset_csv):
    
    np.savetxt(dataset_csv["x_train"], x_train, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["y_train"], y_train, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["x_valid"], x_valid, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["y_valid"], y_valid, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["x_test"], x_test, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["y_test"], y_test, delimiter=',', fmt='%f')

    # pd.DataFrame(x_train).to_csv(dataset_csv["x_train"], index=False) 
    # pd.DataFrame(y_train).to_csv(dataset_csv["y_train"], index=False) 
    # pd.DataFrame(x_valid).to_csv(dataset_csv["x_valid"], index=False) 
    # pd.DataFrame(y_valid).to_csv(dataset_csv["y_valid"], index=False) 
    # pd.DataFrame(x_test).to_csv(dataset_csv["x_test"], index=False) 
    # pd.DataFrame(y_test).to_csv(dataset_csv["y_test"], index=False) 

def load_csv_data(dataset_csv):
    x_train = np.genfromtxt(dataset_csv["x_train"], delimiter=',', dtype='float32')
    x_valid = np.genfromtxt(dataset_csv["x_valid"], delimiter=',', dtype='float32')
    x_test = np.genfromtxt(dataset_csv["x_test"], delimiter=',', dtype='float32')
    y_train = np.genfromtxt(dataset_csv["y_train"], delimiter=',', dtype='float32')
    y_valid = np.genfromtxt(dataset_csv["y_valid"], delimiter=',', dtype='float32')
    y_test = np.genfromtxt(dataset_csv["y_test"], delimiter=',', dtype='float32')
    # x_train = pd.read_csv(dataset_csv["x_train"], index_col=False)
    # y_train = pd.read_csv(dataset_csv["y_train"], index_col=False)
    # x_valid = pd.read_csv(dataset_csv["x_valid"], index_col=False)
    # y_valid = pd.read_csv(dataset_csv["y_valid"], index_col=False)
    # x_test = pd.read_csv(dataset_csv["x_test"], index_col=False)
    # y_test = pd.read_csv(dataset_csv["y_test"], index_col=False)
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

def plot_y_data(df_y, y_name,log_y_name, datetime_name, filename):  
    print("start plot y data")    
         
    plot_functions.view_data(df_y, [y_name,log_y_name], [y_name,log_y_name], df_y[datetime_name].reset_index(drop=True), figname = filename)

def plot_coor_data(df_coor,  coor_names, datetime_name, filename):
    print("start plot coor data")         

    plot_functions.view_data(df_coor,  coor_names, coor_names, df_coor[datetime_name].reset_index(drop=True),  figname = filename)
    
def plot_feature_data(df_feature, scaled_feature_names, datetime_name, filename):
    print("start plot feature data")         

    plot_functions.view_data(df_feature,  scaled_feature_names, scaled_feature_names, df_feature[datetime_name].reset_index(drop=True),  figname = filename)

def save_df_data(df_data,  index_train, index_valid, index_test, dataset_csv):
    df_data["index_train"] = index_train
    df_data["index_valid"] = index_valid
    df_data["index_test"] = index_test

    df_data.to_csv(dataset_csv["df_data"], index=False)
    return True

def prepare_ml_dataset(energy, species, recalc = False, plot_data = False, save_data = True, dL01=True, average_time = 300, raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'],  forecast = "none", number_history = 7, test_ts = '2017-01-01', test_te = '2018-01-01'):
    
    np.set_printoptions(precision=4)
    
    dataset_csv, data_settings, directories = initialize_var.initialize_data_var(energy=energy, species=species, raw_feature_names = raw_feature_names, forecast = forecast, number_history = number_history, test_ts=test_ts, test_te=test_te, dL01=dL01)
    
    if os.path.exists(dataset_csv["x_train"]) & (recalc != True):
        x_train, x_valid, x_test, y_train, y_valid, y_test  = load_csv_data(dataset_csv)
    else:
        df_data, directories, fulldataset_csv, fulldata_settings = prepare_fulldata.load_fulldata(energy, species, recalc = False, raw_feature_names = raw_feature_names, number_history = number_history, save_data = save_data, plot_data = plot_data)
        
        df_full = prepare_fulldata.read_probes_data(directories["rawdata_dir"], fulldata_settings)
        
        df_data[[fulldata_settings['doubletime_name']]] = df_full[[fulldata_settings['doubletime_name']]]
        
        # df_data[[fulldata_settings['doubletime_name']]+fulldata_settings["raw_coor_names"]+ fulldata_settings["raw_feature_names"]] = df_full[[fulldata_settings['doubletime_name']]+fulldata_settings["raw_coor_names"]+ fulldata_settings["raw_feature_names"]]
        
        index_good = get_good_index(df_full, data_settings, fulldata_settings)

        if data_settings["forecast"] == "all":
            data_settings["feature_history_names"] = remove_features_by_time(fulldata_settings["feature_history_names"], "*_0h")
        elif data_settings["forecast"] == "index":
            data_settings["feature_history_names"] = remove_index_features_by_time(fulldata_settings["feature_history_names"], "*_0h")
        else:
            data_settings["feature_history_names"] = fulldata_settings["feature_history_names"]            
        
        df_data = df_data.loc[index_good,[fulldata_settings['doubletime_name'], fulldata_settings['datetime_name'],data_settings['y_name'], data_settings['log_y_name']]+ fulldata_settings['coor_names']+fulldata_settings['feature_history_names']]

        df_full = df_full.loc[index_good, :]

        #set test set. Here we use one year (2017) of data for test set 
        index_train, index_valid, index_test = create_ml_indexes(df_data,  fulldata_settings, data_settings["test_ts"], data_settings["test_te"])
        
        # Each round, one can only train one y. If train more than one y, need to  repeat from here
        x_train, x_valid, x_test, y_train, y_valid, y_test = create_ml_data(df_data, index_train, index_valid, index_test, data_settings["log_y_name"], fulldata_settings["coor_names"], data_settings["feature_history_names"])  
        
        print("shapes of x_train, x_valid, x_test, y_train, y_valid, y_test ")
        print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape, y_test.shape)

        if save_data:
            save_df_data(df_full.loc[index_good, [fulldata_settings['datetime_name'], data_settings["y_name"]] + fulldata_settings["raw_coor_names"] + fulldata_settings["raw_feature_names"]], index_train, index_valid, index_test, dataset_csv)

            save_csv_data(x_train, x_valid, x_test, y_train, y_valid, y_test , dataset_csv)
            
        print("start plot data") 
        
        if plot_data:
            plot_y_data(df_data.loc[index_good, [ fulldata_settings['datetime_name'],  data_settings["y_name"],data_settings["log_y_name"]]], data_settings["y_name"],data_settings["log_y_name"],  fulldata_settings['datetime_name'], fulldataset_csv["df_y"]+ data_settings["log_y_name"])
            
            plot_coor_data(df_data.loc[index_good, [fulldata_settings['datetime_name'],fulldata_settings["coor_names"]  ]], fulldata_settings["coor_names"],  fulldata_settings['datetime_name'], fulldataset_csv["df_coor"])
                        
            plot_feature_data(df_data.loc[index_good, [fulldata_settings['datetime_name'],fulldata_settings["feature_names"] ]], fulldata_settings["feature_names"],  fulldata_settings['datetime_name'], fulldataset_csv["df_coor"])
        
    return x_train, x_valid, x_test, y_train, y_valid, y_test       

def prepare_ml_dataset_batch(raw_feature_names =  ['symh','asyh','ae','asyd'] , number_history_arr = [7,8],forecast_arr = ["all", "index","none"],dL01_arr = [True, False],  species_arr = ['h', 'o'],energy_arr = ['972237', '51767680']   ):
    '''
        raw_feature_names:    #['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz']

    '''

    for number_history in number_history_arr:
        for forecast in forecast_arr:
            for dL01 in dL01_arr:
                for species in species_arr:
                    for energy in energy_arr:
                        prepare_ml_dataset(energy, species, recalc = True, plot_data = True, save_data = True, dL01=dL01, forecast = forecast, number_history =number_history,raw_feature_names =  raw_feature_names)

def __main__():
    if __name__ == "__name__":
        prepare_ml_dataset_batch()

