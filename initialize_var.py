

"""
helper functions for initialize some setting variables 

"""

"""

Examples
--------

"""

import os
# import pickle
import numpy as np
import json

def create_y_name(energy, species, property = 'flux'):
    return species + '_'+property+'_' + energy

def create_log_y_name(energy, species, property = 'flux'):
    return 'log_' + species + '_'+property+'_' + energy

def initialize_dir(release):    
    path = "output_" + release + "/"

    directories = {
        "rawdata_dir" : path + "rawdata/", 
        "fulldata_dir" : path + "fulldata/",
        "data_view_dir" : path + "data_view/",
        "training_output_dir" : path + "training/", 
        "model_setting_compare_dir" : path + "model_setting_compare/"  ,
        "ml_data" : path + "ml_data/", 
        "model_dir": path + "model/"}
    
    return directories

def create_directories(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok = True)
   
def initialize_fulldatacsv(directories):    
    fulldataset_csv = {
        "fulldata_settings_filename" : directories['fulldata_dir'] + "fulldata_settings",
        "df_y" : directories['fulldata_dir'] + "df_hope",
        "df_feature" : directories['fulldata_dir'] + "df_feature_history",
        "df_coor" : directories['fulldata_dir'] + "df_coor",
        "fulldata_settings": directories['fulldata_dir'] + "data_setting",
        } 
    return fulldataset_csv
    
def initialize_datacsv(directories, species, energy, number_history):    
    dataset_csv = {
        "df_y"    : directories["ml_data"] + species+'_'+energy + '_' +  "df_y.csv",
        "df_coor" : directories["ml_data"] + "df_coor.csv",
        "df_feature" : directories["ml_data"] + "df_features.csv",
        "df_data" : directories["ml_data"] + species+'_'+energy + '_' +  "df_data.csv",
        "x_train" : directories["ml_data"] + species+'_'+energy + '_' + "x_train.csv",
        "y_train" : directories["ml_data"] + species+'_'+energy + '_' +"y_train.csv",
        "x_valid" : directories["ml_data"] + species+'_'+energy + '_' +  "x_valid.csv",
        "y_valid" : directories["ml_data"] + species+'_'+energy + '_' + "y_valid.csv",
        "x_test"  : directories["ml_data"] + species+'_'+energy + '_' + "x_test.csv",
        "y_test"  : directories["ml_data"] + species+'_'+energy + '_' + "y_test.csv" 
        } 
    return dataset_csv
    
def initialize_fulldata_settings(release, average_time, raw_coor_names,  coor_names, raw_feature_names, number_history, history_resolution, energy, species):
    """Here we create the settings to store the attributes of the dataset, selected feature and history, and model.

    Args:
        average_time (float): _description_
        coor_names (array): _description_
        feature_names (array): _description_
        number_history (float): _description_
        history_resolution (float): _description_

    Returns:
        fulldata_settings: _description_
    """
    
    y_names = []
    for ien in energy:
        for isp in species:
            y_names.append( isp +"_flux_" + ien)
    
    fulldata_settings = {
    "release" : release,
    "average_time" : average_time,
    "number_history" : number_history,
    "history_resolution" : history_resolution,
    "raw_coor_names" : raw_coor_names,
    "coor_names": coor_names,
    "raw_feature_names" : raw_feature_names,
    "feature_names" : ["scaled_" + str(x) for x in raw_feature_names],
    "datetime_name" : "DateTime",
    "doubletime_name" : "time",
    "y_names": y_names,
    "log_y_names":["log_" + str(x) for x in y_names],
    "feature_history_names":[]
    }
    
    return fulldata_settings

def initialize_data_settings(energy , species, number_history, raw_feature_names, dL01, forecast, l_min = 1, l_max = 8, rel05_invalid_time = ['2017-10-29', '2017-11-01'], test_ts = '2017-01-01', test_te = '2018-01-01'):
    y_name =  create_y_name(energy, species)
    log_y_name = create_log_y_name(energy, species)
    
    data_settings = {
        "energy" : energy,
        "species" : species,
        "raw_feature_names" : raw_feature_names,
        "feature_names" : ["scaled_" + str(x) for x in raw_feature_names],
        "number_history" : number_history,
        "dL01" : dL01,
        "l_min" : l_min,
        "l_max" : l_max,
        "rel05_invalid_time": rel05_invalid_time,
        "y_name":y_name,
        "log_y_name":log_y_name,
        "forecast" : forecast,
        "test_ts" : test_ts,
        "test_te" : test_te
    }
    
    return data_settings

def initialize_model_settings(layer, n_neurons, dropout_rate,patience, learning_rate, epochs,  batch_size):
    
    model_settings = {
        "layer" : layer,
        "n_neurons" : n_neurons,
        "dropout_rate" : dropout_rate,
        "patience" : patience,
        "learning_rate" : learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }
    
    return model_settings

def initialize_fulldata_var(release= 'rel05', average_time = 300, raw_coor_names= ["mlt","l","lat"], coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'], number_history = 30, history_resolution = 2*3600., energy = (np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237, 834.421, 716.163, 614.578, 527.484, 452.702, 388.543, 333.459, 286.184, 245.592, 210.769, 180.870, 155.262, 133.243, 114.319, 98.138, 84.209, 72.320, 62.049, 53.255, 45.728, 39.185, 33.627, 28.914, 24.763, 21.246, 18.291, 15.688, 13.437, 11.537, 9.919, 8.512, 7.316, 6.261, 5.347, 4.643, 3.940, 3.377, 2.955, 2.533, 2.181, 1.829, 1.548, 1.337, 1.196, 0.985]) * 1000.).astype(int).astype(str), species = ['h','o']):

    directories = initialize_dir(release)
    create_directories(directories.values())
    
    fulldataset_csv = initialize_fulldatacsv(directories)   

    fulldata_settings = initialize_fulldata_settings(release, average_time, raw_coor_names,  coor_names, raw_feature_names, number_history, history_resolution, energy, species)
    
    # with open(fulldataset_csv["fulldata_settings_filename"]+'.pkl', 'wb') as file:
    #     pickle.dump(fulldata_settings, file)
    
    with open(fulldataset_csv["fulldata_settings_filename"]+'.json', 'w') as file:
        json.dump(fulldata_settings, file, indent = 4)
    
    return directories, fulldataset_csv, fulldata_settings

def initialize_data_var(energy, species, raw_feature_names, number_history, dL01, forecast, test_ts = '2017-01-01', test_te = '2018-01-01'):
    directories, fulldataset_csv, fulldata_settings = initialize_fulldata_var(raw_feature_names = raw_feature_names, number_history=number_history)   
    
    for raw_feature_name in raw_feature_names:
        directories["ml_data"] = directories["ml_data"] + raw_feature_name + '_'
        
    directories["ml_data"]  = directories["ml_data"] + "history" + str(number_history) + "days_dL01" + str(dL01) + "_forecast" + forecast +'/'
    create_directories(directories.values())
    # print(directories["ml_data"])
    
    dataset_csv = initialize_datacsv(directories = directories, species = species, energy =energy, number_history =number_history)   

    data_settings = initialize_data_settings(energy=energy , species=species, number_history=number_history, raw_feature_names=raw_feature_names, dL01=dL01, forecast=forecast, test_ts = test_ts, test_te = test_te)
            
    return dataset_csv, data_settings, directories

def initialize_model_var(layer, n_neurons, dropout_rate,patience, learning_rate, epochs,  batch_size, energy , species, release, average_time, raw_coor_names,  coor_names, raw_feature_names, number_history, history_resolution):
    
    initialize_data_var(energy =energy, species=species, release=release, average_time=average_time, raw_coor_names=raw_coor_names,  coor_names=coor_names, raw_feature_names=raw_feature_names, number_histor=number_history, history_resolution=history_resolution)
    
    
    directories = initialize_dir(release)
    create_directories(directories.values())
    
    model_settings = initialize_model_settings(layer, n_neurons, dropout_rate,patience, learning_rate, epochs,  batch_size)
                
    return directories, model_settings