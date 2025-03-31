"""
Functions to prepare raw csv data from IDL program, which extracted data from CDF file.

"""

"""
Read data

Examples
--------


"""


import os
import numpy as np
import pandas as pd
# import modin.pandas as pd
from time_string import time_string

import plot_functions
import initialize_var

np.set_printoptions(precision=4)

def scale_arr(arr):
    index = np.isfinite(arr)
    valid_arr = arr[index]
    max_value = max(valid_arr)
    min_value = min(valid_arr)
    mid_value = (max_value + min_value)/2
    scale_value = max_value - min_value
    scaled_arr = (arr - mid_value)/scale_value*2
    # print(max_value, min_value, mid_value, scale_value)
    
    return(scaled_arr, mid_value, scale_value)

def scale_var(df_full, varname, fulldata_settings):
    scaled_var, fulldata_settings[varname+'_mid_vlaue'],  fulldata_settings[varname+'_scale_vlaue'] = scale_arr(df_full[varname])

    return scaled_var, fulldata_settings

def read_probes_data(data_dir, fulldata_settings):
    df_full = pd.DataFrame()
    probes = ['a','b']
    
    for iprobe in probes:
        print("Reading csv data for probe " + iprobe, end="\r")
        df = pd.read_csv(data_dir + 'rbsp' + iprobe.capitalize() + '_data_' + fulldata_settings["release"] + '_fulldata.csv')  
        df['probe'] = iprobe
        if iprobe == probes[0]:
            df_full = df
        else:
            df_full = pd.concat([df_full, df], ignore_index=True)           

    df_full[fulldata_settings["datetime_name"]] = df_full['time'].apply(lambda x : time_string(x)).astype('datetime64[ns]')
    
    return df_full

def plot_coor_data(df_coor, df_full, raw_coor_names, coor_names, datetime_name, filename = 'dataview_coor'):
    # print(df_full.columns)
    plot_functions.view_data(df_full, raw_coor_names, raw_coor_names, df_full[datetime_name].astype('datetime64[ns]').reset_index(drop=True), figname = filename)
    plot_functions.view_data(df_coor, coor_names, coor_names, df_full[datetime_name].astype('datetime64[ns]').reset_index(drop=True), figname = filename+'_scaled')

def scale_corrdinates(df_full, fulldata_settings, doubletime_name, outputfilename, save_data = True, plot_data = True):       
    df_cos = df_full['mlt'].apply(lambda x: np.cos(x*np.pi/12.0))
    df_sin = df_full['mlt'].apply(lambda x: np.sin(x*np.pi/12.0))
    df_l, fulldata_settings = scale_var(df_full, 'l', fulldata_settings)
    df_lat, fulldata_settings = scale_var(df_full, 'lat', fulldata_settings)
    
    # df_coor = pd.DataFrame(np.array([df_full[doubletime_name],df_cos,df_sin,df_l,df_lat, df_full["mlt"], df_full["l"], df_full["lat"]]), columns=[doubletime_name, 'cos0','sin0','scaled_l','scaled-l', "mlt",'l','lat']).reset_index(drop=True)
    
    df_coor = pd.DataFrame({doubletime_name:df_full[doubletime_name],"cos0":df_cos, "sin0":df_sin, "scaled_l":df_l,"scaled_lat":df_lat })
    # df_coor = pd.concat([df_coor, df_full[["mlt",'l','lat']]], axis=1)
    
    if save_data:
        df_coor.to_csv(outputfilename + '.csv', index=False)
        
    if plot_data:            
        plot_coor_data(df_coor, df_full, fulldata_settings["raw_coor_names"], fulldata_settings["coor_names"], fulldata_settings["datetime_name"], filename = outputfilename)
        
    return df_coor, fulldata_settings

def load_coor(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = True, plot_data = True):
    coor_filename =  fulldataset_csv["df_coor"] + ".csv"
    
    if os.path.exists(coor_filename) & (recalc != True):
        df_coor = pd.read_csv(coor_filename, index_col=False)
        # with open(fulldataset_csv["fulldata_settings_filename"]+'.pkl', 'rb') as file:
        #     fulldata_settings = pickle.load(file)
    else:
        if len(df_full) == 0:
            df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
        df_coor, fulldata_settings = scale_corrdinates(df_full, fulldata_settings,  fulldata_settings["datetime_name"], fulldataset_csv["df_coor"], save_data = save_data, plot_data = plot_data)
            
    return df_coor, df_full, fulldata_settings

def plot_y_data(df_full, y_names, datetime_name, filename = 'dataview_y'):
    plot_functions.view_data(df_full, y_names, y_names,  df_full[datetime_name].astype('datetime64[ns]').reset_index(drop=True), figname = filename)

def calculate_log_for_y(df_y, y_name, fulldata_settings, log_y_filename, datetime_name, save_data = True, plot_data = True):
    log_y_name = "log_"+y_name
    # y_names = [i for i in df_full.columns if re.findall(r'^[a-z]_flux_', i)]  
    index = df_y[y_name] == 0 
    df_y.loc[index, y_name] = 1e-5

    # Here we intergrated over for geomatrics and convert the unit  first and then take the log   np.log10(x*1e3*4*math.pi))
    df_y[log_y_name] = np.log10(df_y[y_name])
    
    fulldata_settings["y_names"].append(y_name)
    fulldata_settings["log_y_names"].append("log_"+y_name) #["log_" + str(x) for x in y_name]
    
    if save_data:
        df_y[[datetime_name, y_name, log_y_name]].to_csv(log_y_filename + '.csv', index = False)
        
    if plot_data:
        plot_y_data(df_y, [y_name,log_y_name], fulldata_settings["datetime_name"], log_y_filename)
    
    return df_y, fulldata_settings

def load_y(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = True, plot_data = True, energy_bins=(np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237, 834.421, 716.163, 614.578, 527.484, 452.702, 388.543, 333.459, 286.184, 245.592, 210.769, 180.870, 155.262, 133.243, 114.319, 98.138, 84.209, 72.320, 62.049, 53.255, 45.728, 39.185, 33.627, 28.914, 24.763, 21.246, 18.291, 15.688, 13.437, 11.537, 9.919, 8.512, 7.316, 6.261, 5.347, 4.643, 3.940, 3.377, 2.955, 2.533, 2.181, 1.829, 1.548, 1.337, 1.196, 0.985]) * 1000.).astype(int).astype(str), species_arr=['h','o'] ):   

    for species in species_arr:
        for energy in energy_bins:
            y_name = species + '_flux_'+energy
            log_y_name = 'log_' + y_name
            log_y_filename = fulldataset_csv["df_y"] + '_'+log_y_name
            print(log_y_filename)

            if os.path.exists(log_y_filename+'.csv') & (recalc != True):
                idf_y = pd.read_csv(log_y_filename+'.csv', index_col=False)
            else:
                if len(df_full) == 0:
                    df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
                idf_y = df_full[[fulldata_settings["datetime_name"], y_name]]
                idf_y, fulldata_settings = calculate_log_for_y(idf_y, y_name, fulldata_settings, log_y_filename, fulldata_settings["datetime_name"], save_data = save_data, plot_data = plot_data)
                
            if not 'df_y' in locals():
                df_y = idf_y
            else:
                print(idf_y.columns)
                df_y = pd.concat([df_y, idf_y[[y_name, log_y_name]]], axis=1)
                
    return df_y, df_full, fulldata_settings

# def filter_history_days(df_feature_history, fulldata_settings, scaled_feature_name):
#     # m_history is the number of feature history
#     m_history = int(fulldata_settings["number_history"]*24*60*60/(fulldata_settings["history_resolution"]) + 1)

#     # calculate history of the solar wind driver and geomagentic indexes
#     feature_history_names = ["" for x in range(m_history)]
#     print(m_history)
#     ihf = 0
#     for k in range(m_history):
#         feature_history_names[ihf] = scaled_feature_name + '_' + str(k*2)+'h'
#         ihf = ihf + 1
        
#     return df_feature_history[feature_history_names]

def create_feature_history_names(fulldata_settings, scaled_feature_name):
    # m_history is the number of feature history
    m_history = int(fulldata_settings["number_history"]*24*60*60/(fulldata_settings["history_resolution"]) + 1)

    # calculate history of the solar wind driver and geomagentic indexes
    feature_history_names = ["" for x in range(m_history)]

    ihf = 0
    for k in range(m_history):
        feature_history_names[ihf] = scaled_feature_name + '_' + str(k*2)+'h'
        ihf = ihf + 1
        
    return feature_history_names

def plot_feature_data(df_feature, raw_feature_names,  scaled_raw_feature_names, datetime_name, filename):   
    
    plot_functions.view_data(df_feature, [raw_feature_names, scaled_raw_feature_names], [raw_feature_names, scaled_raw_feature_names], df_feature[datetime_name].astype('datetime64[ns]').reset_index(drop=True), figname = filename)

def scale_feature(df_feature, raw_feature_name, fulldata_settings, feature_filename, plot_data = True): 
    scaled_feature_name = "scaled_"+raw_feature_name
    
    df_feature[scaled_feature_name], fulldata_settings = scale_var(df_feature, raw_feature_name, fulldata_settings)
    
    if plot_data:
        plot_feature_data(df_feature, raw_feature_name, scaled_feature_name, fulldata_settings["datetime_name"], feature_filename)
        
    return df_feature, fulldata_settings

# def create_features_history(df_full, fulldata_settings):
#     # extract the length the parameters
#     # m_coor = len(fulldata_settings["coor_names"])
#     # m_y = len(fulldata_settings["y_name_log"])
#     m_feature = len(fulldata_settings["feature_names"])

#     # For each feature, we will add 2 hours earlier of the parametners: feature_1 no delay, feautre_2, 2 hours before the observing time, feature_3, 4 hours before the observation time.
#     # Time reslution is set to be two hours for each feature and 
#     n_history_total_days = fulldata_settings["number_history"]
#     n_history_total = n_history_total_days*24*60*60/fulldata_settings["average_time"]

#     m_history = int(n_history_total/(fulldata_settings["history_resolution"]/fulldata_settings["average_time"]) + 1)
    
#     # m is the total number of parameters including features and y
#     # m = m_coor + m_y + m_feature * m_history

#     # calculate history of the solar wind driver and geomagentic indexes
#     index_difference = fulldata_settings["history_resolution"]/fulldata_settings["average_time"]
#     feature_history_names = ["" for x in range(m_feature*m_history)]
#     ihf = 0
#     index0 = int(n_history_total)
#     index1 = df_full.index[-1]

#     df_history = np.zeros((index1-index0+1, len(feature_history_names)))
    
#     for feature_name in fulldata_settings["feature_names"]:
#         for k in range(m_history):
#             name = feature_name + '_' + str(k*2)+'h'
#             feature_history_names[ihf] = name
            
#             temp = np.array(df_full.loc[(index0 - index_difference*k):(index1-index_difference*k), feature_name])
#             df_history[:,ihf] = temp
#             # df_full.loc[index0:index1,name] = np.array(df_full.loc[(index0 - index_difference*k):(index1-index_difference*k), feature_name])  

#             ihf = ihf + 1

#     df_full.loc[index0:index1, feature_history_names] = df_history  

#     ## This method is slow but good if different history calculation is wanted
#     # def calculate_history(x, df_full, feature_name):
#     #     index = df_full['time'] == (x-fulldata_settings["history_resolution"])
#     #     return(float(df_full.loc[index,feature_name]))

#     # for feature_name in feature_names:
#     #     print(feature_name)
#     #     for k in range(m_history):
#     #         print(k)
#     #         name = feature_name + '_' + str(k*2)+'h'
#     #         df[name] = df.loc[:,'time'].swifter.apply(calculate_history, df_full = df_full,feature_name = feature_name)  

#     return df_full, fulldata_settings 

def create_feature_history(df_feature, fulldata_settings, raw_feature_name, scaled_feature_name, feature_history_filename, datetime_name, save_data = True):
    
    # Time reslution is set to be two hours for each feature. For each feature, we will add 2 hours earlier of the parametners: feature_0h no delay, feautre_2h, 2 hours before the observing time, feature_4h, 4 hours before the observation time
    n_history_total_days = fulldata_settings["number_history"]
    n_history_total = n_history_total_days*24*60*60/fulldata_settings["average_time"]

    # m_history is the number of feature history we are going to add
    m_history = int(n_history_total/(fulldata_settings["history_resolution"]/fulldata_settings["average_time"]) + 1)

    # calculate history of the solar wind driver and geomagentic indexes
    index_difference = fulldata_settings["history_resolution"]/fulldata_settings["average_time"]
    feature_history_names = ["" for x in range(m_history)]
    
    index1 = df_feature.index[-1]

    arr_history = np.zeros((df_feature.shape[0], m_history))
    
    ihf = 0
    for k in range(m_history):
        feature_history_names[ihf] = scaled_feature_name + '_' + str(k*2)+'h'
        if k == 0:
            arr_history[:,ihf] = np.array(df_feature.loc[:, scaled_feature_name]) 
        else:
            arr_history[:,ihf] = np.concatenate((np.full(int(index_difference*k), np.nan),  np.array(df_feature.loc[0:(index1-index_difference*k), scaled_feature_name])))
        
        ihf = ihf + 1
    
    df_history = pd.concat([df_feature[datetime_name], pd.DataFrame(arr_history, columns=feature_history_names)],axis=1)
    
    if save_data:
        df_history.to_csv(feature_history_filename+ ".csv", index=False)
        print("Writing csv data completed for " + feature_history_filename)
    fulldata_settings["feature_history_names"] = feature_history_names
    return df_history[feature_history_names], fulldata_settings

def load_features(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = False, plot_data = False, raw_feature_names = np.array(['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz' ])):
    df_features_history = pd.DataFrame()
    for raw_feature_name in raw_feature_names:
        print("start " + raw_feature_name)
        scaled_feature_name = "scaled_" + raw_feature_name
        feature_history_filename =  fulldataset_csv["df_feature"] + "_" + scaled_feature_name 
        
        feature_history_names = create_feature_history_names(fulldata_settings, scaled_feature_name) # feature names to extract

        if os.path.exists(feature_history_filename+'.csv') & (recalc != True):
            print("Reading from "+feature_history_filename+'.csv')
            
            idf_feature_history = pd.read_csv(feature_history_filename+'.csv', index_col=False, usecols=feature_history_names, low_memory=False, dtype = 'float')
        else:        
            print("Calculate the feature history of " + raw_feature_name)
            if len(df_full) == 0:
                df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
            idf_feature = df_full[[fulldata_settings["datetime_name"], raw_feature_name]]
            
            idf_feature, fulldata_settings = scale_feature(idf_feature, raw_feature_name, fulldata_settings, feature_history_filename, plot_data = plot_data)
            
            idf_feature_history, fulldata_settings = create_feature_history(idf_feature, fulldata_settings,raw_feature_name, "scaled_"+raw_feature_name, feature_history_filename, fulldata_settings["datetime_name"], save_data = save_data)
            
        df_features_history[feature_history_names] = idf_feature_history
        fulldata_settings["feature_history_names"] = fulldata_settings["feature_history_names"] + feature_history_names
    
    return df_features_history, df_full, fulldata_settings

def load_fulldata(energy =(np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237]) * 1000.).astype(int).astype(str), species = ['h','o'], recalc = False, release = 'rel05', average_time = 300, raw_coor_names = ["mlt","l","lat"], coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'], number_history = 30, history_resolution = 2*3600., save_data = False, plot_data = False, df_full = [], create_full_data = False):
    
    """ main function to load full data.

    Args:
        energy (_type_): energy channel selected. If input is []. Full energy channels are: (np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237, 834.421, 716.163, 614.578, 527.484, 452.702, 388.543, 333.459, 286.184, 245.592, 210.769, 180.870, 155.262, 133.243, 114.319, 98.138, 84.209, 72.320, 62.049, 53.255, 45.728, 39.185, 33.627, 28.914, 24.763, 21.246, 18.291, 15.688, 13.437, 11.537, 9.919, 8.512, 7.316, 6.261, 5.347, 4.643, 3.940, 3.377, 2.955, 2.533, 2.181, 1.829, 1.548, 1.337, 1.196, 0.985]) * 1000.).astype(int).astype(str)
        species (_type_): _description_
        recalc (bool, optional): _description_. Defaults to False.
        release (str, optional): _description_. Defaults to 'rel05'.
        average_time (int, optional): _description_. Defaults to 300.
        raw_coor_names (list, optional): _description_. Defaults to ["mlt","l","lat"].
        coor_names (list, optional): _description_. Defaults to ["cos0", 'sin0', 'scaled_lat','scaled_l'].
        raw_feature_names (list, optional): _description_. Defaults to ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'].
        number_history (int, optional): Days of history to calculate and store. Defaults to 30.
        history_resolution (_type_, optional): _description_. Defaults to 2*3600..
        save_data (bool, optional): _description_. Defaults to True.
        plot_data (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    if type(energy) == str:
        energy = [energy]
    if type(species) == str:
        species = [species]
        
    directories, fulldataset_csv, fulldata_settings = initialize_var.initialize_fulldata_var(release = release, average_time = average_time, raw_coor_names = raw_coor_names, coor_names = coor_names, raw_feature_names = raw_feature_names, number_history = number_history, history_resolution = history_resolution, energy = energy, species = species)

    df_coor, df_full, fulldata_settings = load_coor(directories, fulldataset_csv, fulldata_settings, recalc = recalc, df_full = df_full, save_data = save_data, plot_data = plot_data)
    
    df_y, df_full, fulldata_settings = load_y(directories, fulldataset_csv, fulldata_settings, recalc = recalc, df_full = df_full, save_data = save_data, plot_data = plot_data, energy_bins = energy, species_arr = species)

    df_features_history, df_full, fulldata_settings = load_features(directories, fulldataset_csv, fulldata_settings, recalc = recalc, df_full = df_full, save_data = save_data, plot_data = plot_data, raw_feature_names = raw_feature_names)   
    
    if create_full_data == True:
        print("You have calculated full data.")
        return pd.DataFrame(), directories, fulldataset_csv, fulldata_settings
    
    df_data = pd.concat([df_y, df_coor[fulldata_settings['coor_names']], df_features_history[fulldata_settings['feature_history_names']]], axis=1)
    
    return df_data, directories, fulldataset_csv, fulldata_settings

def prepare_fulldata_batch():
    save_data = True
    plot_data = True
    recalc = True
    create_full_data = True
    
    load_fulldata(save_data = save_data, plot_data = plot_data, recalc=recalc, create_full_data = create_full_data)

def __main__():
    if __name__ == "__name__":
      prepare_fulldata_batch()  

