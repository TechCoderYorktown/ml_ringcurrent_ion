"""
Read data

Examples
--------
python HopeData.py --save_csv True

"""


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # This is to disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

import numpy as np
import pandas as pd
import tensorflow as tf
import math
import swifter
# import ml_wrappers
from pyspedas import time_string
# import warnings
import fnmatch
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
np.set_printoptions(precision=4)

def view_data(df_full,ind_good_y, varnames, ylabels, time_array, figname ='temp'):

### attempt to use tplot, but since tplot can't do scatter/ dot
# pytplot.store_data(y_name_original, data={'x':time_array, 'y':df_full.loc[index_good, y_name_original]})
# pytplot.store_data(y_name, data={'x':time_array, 'y':df_full.loc[index_good, y_name]})
# # pytplot.options(y_name_original,'line_style','dot')
# pytplot.tplot([y_name_original, y_name])

    nvar = len(varnames)

    fig1, ax1 = plt.subplots(nvar,1, constrained_layout = True)
    fig1.set_size_inches(8, nvar*2)
    
    for ivar in range(len(varnames)):
        varname = varnames[ivar]
        ax1[ivar].scatter(time_array,df_full.loc[ind_good_y, varname],s = 0.1)
        ax1[ivar].set_ylabel(ylabels[ivar])
        
    plt.savefig(figname + ".png", format = "png", dpi = 300)

def scale_arr(arr):
    index = np.isfinite(arr)
    valid_arr = arr[index]
    max_value = max(valid_arr)
    min_value = min(valid_arr)
    mid_value = (max_value + min_value)/2
    scale_value = max_value - min_value
    scaled_arr = (arr - mid_value)/scale_value*2
    print(max_value, min_value, mid_value, scale_value)
    return(scaled_arr)

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

class HopeData:
    def __init__(self, energy, species, release = 'rel05', dL01=True, average_time = 300, coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], feature_names=[ 'scaled_symh', 'scaled_ae','scaled_asyh', 'scaled_asyd'], forecast = False, number_history = 7, history_resolution = 2*3600., test_ts='2017-01-01', test_te='2018-01-01'):
        
        self.path = "output_" + release + "/"
        
        self.directories = {
            "data_dir" : self.path + "data/", 
            "training_output_dir" : self.path + "training/", 
            "data_view_dir" : self.path + "data_view/",
            "model_setting_compare_dir" : self.path + "model_setting_compare/"     ,
            "ml_data" : self.path +"ml_data/"}
        
        self.data_settings = {
            "average_time" : average_time,
            "dL01" : dL01,
            "forecast" : forecast,
            "species" : species,
            "energy" : energy,
            "number_history" : number_history,
            "history_resolution" : history_resolution,
            "coor_names" : coor_names,
            "feature_names" : feature_names,
            "y_name" : species + '_flux_' + energy,
            "y_name_log" : 'log_' + species + '_flux_' + energy, 
            "test_ts" : test_ts,
            "test_te" : test_te   } 
        
        self.dataset_csv = {
            "x_train" : self.directories["ml_data"] + species+'_'+energy + '_' + str(number_history) + 'days_' + "x_train.csv",
            "y_train" : self.directories["ml_data"] + species+'_'+energy + '_' + str(number_history) + 'days_' + "y_train.csv",
            "x_valid" : self.directories["ml_data"] + species+'_'+energy + '_' + str(number_history) + 'days_' + "x_valid.csv",
            "y_valid" : self.directories["ml_data"] + species+'_'+energy + '_' + str(number_history) + 'days_' + "y_valid.csv",
            "x_test"  : self.directories["ml_data"] + species+'_'+energy + '_' + str(number_history) + 'days_' + "x_test.csv",
            "y_test"  : self.directories["ml_data"] + species+'_'+energy + '_' + str(number_history) + 'days_' + "y_test.csv"  }  
        
        
        self.df_full, self.index_good, self.data_settings["feature_history_names"] = self._create_df_full(release)
        
#set test set. Here we use one year (2017) of data for test set 
        self.index_train, self.index_valid, self.index_test = self._create_ml_indexes(self.df_full, '2017-01-01', '2018-01-01', self.index_good)
        
# Each round, one can only train one y. If train more than one y, need to  repeat from here
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = self._create_ml_data(self.df_full, self.index_train, self.index_valid, self.index_test, self.data_settings["y_name_log"], self.data_settings["coor_names"], self.data_settings["feature_history_names"])  

    def create_directories(self):
        for dir in self.directories.values():
            os.makedirs(dir, exist_ok = True)
        
    def _create_df_full(self, release):
        df_full = self._read_csv_data(self.directories["data_dir"], release)      

        index_good = self._create_good_index(df_full, self.data_settings["y_name"], self.data_settings["feature_names"], self.data_settings["dL01"])
                
        df_full = self._calculate_log_for_y(df_full, self.data_settings["y_name"], self.data_settings["y_name_log"], index_good )
        
        df_full = self._scale_corrdinates(df_full)

        df_full = self._scale_indexes(df_full)

        df_full = self._scale_sw(df_full)
        
        df_full, feature_history_names = self._create_feature_history(df_full, self.data_settings["feature_names"] , self.data_settings["number_history"] , self.data_settings["average_time"]  ,self.data_settings["history_resolution"]  )
        
        if self.data_settings["forecast"]:
            self._remove_features_by_time(feature_history_names, '*_0h')
            
        self.data_settings["feature_history_names"] = feature_history_names
            
        return df_full, index_good, feature_history_names
                   
    @staticmethod
    def _read_csv_data(data_dir, release):
        df_full = pd.DataFrame()
        probes = ['a','b']
        
        for iprobe in probes:
            print("Reading csv data for probe " + iprobe, end="\r")
            filename = data_dir + 'rbsp' + iprobe.capitalize() + '_data_' + release + '_fulldata.csv'
            print(filename)
            df = pd.read_csv(filename)  
            df['probe'] = iprobe
            if iprobe == probes[0]:
                df_full = df
            else:
                df_full = pd.concat([df_full, df], ignore_index=True)           

        df_full['Datetime'] = df_full['time'].swifter.apply(time_string)

        return df_full
    
    @staticmethod
    def _create_good_index(df_full, y_name, feature_names, dL01):
        """
        There are non valid data in the situ plasma, geomagnetic indexes data and solar wind data. Sometimes, the solar wind and index data are pre-processed (interpolated etc.) We need data with no NaN or Inf for the model. Indexes of valid data are created. 
        
        We have previousely reviewed that all coordinates data and all indexes data do not have NaN or Inf data. If solar wind parameters are used, we need to add index_good_sw into the final good index.

        """
        
        index_valid_coor = (df_full['l'] > 2) & (df_full['l'] < 7)
        index_good_index = (df_full['ae'] > 0) # & (df_full['al'] > 0)
        index_valid_rel05 = ((df_full['Datetime'] < '2017-10-29') | (df_full['Datetime'] > '2017-11-01')) 
        index_good_sw = np.isfinite(df_full['swp']) & np.isfinite(df_full['bz']) &  np.isfinite(df_full['swv'])
        index_good_y = np.isfinite(df_full[y_name]) & (df_full[y_name] > 0) # we take out 0 measurement data because we are using the log

        index_good = index_valid_coor & index_good_index & index_valid_rel05 & index_good_y
        
        if 'swp' in feature_names or 'swv' in feature_names or 'swn' in feature_names or 'by' in feature_names or 'bz' in feature_names:
            index_good = index_good & index_good_sw
            
        if dL01 == True:
            df_full["l10"] = df_full["l"].swifter.apply(lambda x: np.floor(x*10))
            df_full["l10_pre"] = np.append(0,np.array(df_full.loc[0:(df_full.shape[0]-2), "l10"]))
            df_full['isMask'] = df_full["l10"] ==  df_full["l10_pre"]
            index_good = index_good & ~df_full['isMask']
        
        return index_good
    
    @staticmethod 
    def _remove_features_by_time(feature_history_names, pattern):
        matching_feature_names = fnmatch.filter(feature_history_names, pattern)
        
        for ifeature_name in matching_feature_names:
            feature_history_names.remove(ifeature_name)
        return feature_history_names
            
    @staticmethod
    def _calculate_log_for_y(df_full, y_name, y_name_log, index_good ):
        
        df_full[y_name_log] = df_full.loc[index_good, y_name].swifter.apply(lambda x: np.log10(x*1e3*4*math.pi))
        
        return df_full
    
    @staticmethod   
    def _scale_corrdinates(df_full ):
        #Scale coordinates L, cos(theta),sin(theta),Lat. All are scaled linearly to [-1,1]
        df_full['cos0'] = df_full['mlt'].swifter.apply(lambda x: np.cos(x*np.pi/12.0))
        df_full['sin0'] = df_full['mlt'].swifter.apply(lambda x: np.sin(x*np.pi/12.0))
        df_full['scaled_l'] = scale_arr(df_full['l']) # here only scales the good data
        df_full['scaled_lat'] = scale_arr(df_full['lat'])# here only scales the good data
        
        return df_full
    
    @staticmethod
    def _scale_indexes(df_full):
        df_full['scaled_symh'] = scale_arr(df_full['symh'])
        df_full['scaled_asyh'] = scale_arr(df_full['asyh'])
        df_full['scaled_asyd'] = scale_arr(df_full['asyd'])
        df_full['scaled_ae'] = scale_arr(df_full['ae'])
        df_full['scaled_f107'] = scale_arr(df_full['f10.7'])
        df_full['scaled_kp'] = scale_arr(df_full['kp'])
    
        return df_full        
    
    @staticmethod
    def _scale_sw(df_full):
        df_full['scaled_swp'] = scale_arr(df_full['swp'])
        df_full['scaled_swn'] = scale_arr(df_full['swn'])
        df_full['scaled_swv'] = scale_arr(df_full['swv'])
        df_full['scaled_by'] = scale_arr(df_full['by'])
        df_full['scaled_bz'] = scale_arr(df_full['bz'])
        
        return df_full
    
    @staticmethod
    def _create_feature_history(df_full, feature_names, number_history, average_time, history_resolution):
        # extract the length the parameters
        # m_coor = len(coor_names)
        # m_y = len(y_name_log)
        m_feature = len(feature_names)

        # For each feature, we will add 2 hours earlier of the parametners: feature_1 no delay, feautre_2, 2 hours before the observing time, feature_3, 4 hours before the observation time.
        # Time reslution is set to be two hours for each feature and 
        n_history_total_days = number_history
        n_history_total = n_history_total_days*24*60*60/average_time

        m_history = int(n_history_total/24 + 1)

        # m is the total number of parameters including features and y
        # m = m_coor + m_y + m_feature * m_history

        # calculate history of the solar wind driver and geomagentic indexes
        index_difference = history_resolution/average_time
        feature_history_names = ["" for x in range(m_feature*m_history)]
        ihf = 0
        index0 = n_history_total
        index1 = df_full.index[-1]

        for feature_name in feature_names:
            for k in range(m_history):
                name = feature_name + '_' + str(k*2)+'h'
                print(df_full.columns   )
                feature_history_names[ihf] = name
                df_full.loc[index0:index1,name] = np.array(df_full.loc[(index0 - index_difference*k):(index1-index_difference*k), feature_name])  
                ihf = ihf + 1
                
        ## This method is slow but good if different history calculation is wanted
        # def calculate_history(x, df_full, feature_name):
        #     index = df_full['time'] == (x-data_settings["history_resolution"])
        #     return(float(df_full.loc[index,feature_name]))

        # for feature_name in feature_names:
        #     print(feature_name)
        #     for k in range(m_history):
        #         print(k)
        #         name = feature_name + '_' + str(k*2)+'h'
        #         df[name] = df.loc[:,'time'].swifter.apply(calculate_history, df_full = df_full,feature_name = feature_name)  
    
        return df_full, feature_history_names 
    
    @staticmethod
    def _create_ml_data(df_full, index_train, index_valid,index_test, y_name, coor_names, history_feature_names):
        y_train = np.array(df_full.loc[index_train, y_name],dtype='float32')
        y_valid = np.array(df_full.loc[index_valid, y_name],dtype='float32')
        y_test  = np.array(df_full.loc[index_test, y_name],dtype='float32')

        x_train = np.array(df_full.loc[index_train, coor_names + history_feature_names], dtype='float32')
        x_valid = np.array(df_full.loc[index_valid, coor_names + history_feature_names], dtype='float32')
        x_test  = np.array(df_full.loc[index_test,  coor_names + history_feature_names], dtype='float32')

        return x_train, x_valid, x_test, y_train, y_valid, y_test

    # functions for create train and validation data set with a given test data set
    # the function keeps the "episode time" to 2 hours
    @staticmethod
    def _create_ml_indexes(df_full, test_ts, test_te, index_good, train_size=0.8):
        index_test = (df_full['Datetime'] >= test_ts ) & (df_full['Datetime'] <= test_te ) & index_good

        #If the test set is randomly split
        #episode_train_full,episode_test=train_test_split(episodes, test_size=0.01, train_size=1.0, random_state=42)
        #episode_train,episode_valid=train_test_split(episode_train_full, test_size=0.2, train_size=0.8, random_state=42)
        
        t0 = min(df_full['time'])
        # t1 = max(df_full['time'])

        episode_time= 86400.0*2 # 2 days
        
        # N_episode = np.ceil((t1-t0)/episode_time).astype(int)
        
        df_full['episodes'] = df_full['time'].apply(lambda x: math.floor((x-t0)/episode_time))

        # episode_test = np.array(df_full.loc[index_test,'episodes'])

        episode_train, episode_valid = train_test_split(np.unique(df_full.loc[index_good & ~index_test,'episodes']), test_size=1-train_size, train_size=train_size, random_state=42)

        episode_train = np.array(episode_train)
        episode_valid = np.array(episode_valid)
        
        index_train = index_good & df_full.loc[:,'episodes'].apply(lambda x: x in episode_train) & ~index_test
        index_valid = index_good & df_full.loc[:,'episodes'].apply(lambda x: x in episode_valid) & ~index_test

        np.set_printoptions(precision=3,suppress=True)
        print(sum(index_train), sum(index_valid), sum(index_test))   
        
        return index_train, index_valid, index_test

    # def split_data(self):
    #     #set test set. Here we use one year (2017) of data for test set 
    #     self.index_train, self.index_valid, self.index_test =  ml_wrappers.create_ml_indexes(self.df_full, '2017-01-01', '2018-01-01', self.index_good, train_size=0.8)

    #     # Each round, one can only train one y. If train more than one y, need to  repeat from here
    #     self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = ml_wrappers.create_ml_data(self.df_full, self.index_train, self.index_valid, self.index_test, self.data_settings["y_name_log"], self.coor_names, self.feature_history_names)  
    
    def plot_plasma_data(self):        
        time_array = self.df_full.loc[self.index_good,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)
        
        view_data(self.df_full, self.index_good, [self.data_settings["y_name"],self.data_settings["y_name_log"]], [self.data_settings["y_name"],self.data_settings["y_name_log"]], time_array, figname = self.directories["data_view_dir"] + 'rbsp_'+self.data_settings["y_name"])
    
    def plot_index_data(self) :
        time_array = self.df_full.loc[self.index_good,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)

        view_data(self.df_full, self.index_good, ['mlt',"cos0",'sin0','l','scaled_l','lat','scaled_lat'], ['MLT','cos theta','sin theta','L','scaled L','LAT','scaled LAT'], time_array, figname = self.directories["data_view_dir"] + 'coor')
    
    def plot_sw_data(self):
        time_array = self.df_full.loc[self.index_good,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)

        view_data(self.df_full, self.index_good, ['swp',"scaled_swp",'swn','scaled_swn','swv','scaled_swv','by','scaled_by',"bz","scaled_bz"], ['SW P','scaled SW P','SW N','scaled SW N','SW V','scaled SW V','IMF By','scaled IMF By','IMF Bz','scaled IMF Bz'], time_array, figname = self.directories["data_view_dir"] + 'sw')
    
    def save_csv_data(self):
        pd.DataFrame(data=self.x_train).to_csv(self.dataset_csv["x_train"] ) 
        pd.DataFrame(data=self.y_train).to_csv(self.dataset_csv["y_train"] ) 
        pd.DataFrame(data=self.x_valid).to_csv(self.dataset_csv["x_valid"] ) 
        pd.DataFrame(data=self.y_valid).to_csv(self.dataset_csv["y_valid"] ) 
        pd.DataFrame(data=self.x_test).to_csv(self.dataset_csv["x_test"] ) 
        pd.DataFrame(data=self.y_test).to_csv(self.dataset_csv["y_test"] ) 
    
    @staticmethod
    def load_csv_data(dataset_csv):
        x_train = np.array(pd.read_csv(dataset_csv["x_train"]))
        y_train = np.array(pd.read_csv(dataset_csv["y_train"]))
        x_valid = np.array(pd.read_csv(dataset_csv["x_valid"]))
        y_valid = np.array(pd.read_csv(dataset_csv["y_valid"]))
        x_test = np.array(pd.read_csv(dataset_csv["x_test"] ))
        y_test = np.array(pd.read_csv(dataset_csv["y_test"] ))
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    
    def view_data(self):
        help, self.df_full.loc[self.index_good, :]
    
    def print_model(self):
        print(self.data_settings)

def read_hopedata(energy, species, release = 'rel05', dL01=True, average_time = 300, coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], feature_names=[ 'scaled_symh', 'scaled_ae','scaled_asyh', 'scaled_asyd'], forecast = False, number_history = 7, history_resolution = 2*3600. , test_ts = '2017-01-01', test_te = '2018-01-01'):
    return HopeData(energy, species, release = release, dL01=dL01, average_time = average_time, coor_names=coor_names, feature_names=feature_names, forecast = forecast, number_history = number_history, history_resolution = history_resolution, test_ts=test_ts, test_te=test_te)

def create_ml_dataset(energy, species, plot_plasma = True, plot_index = True, plot_sw = True, save_csv = True):
    hopedata = read_hopedata(energy, species)
    
    if plot_plasma:
        hopedata.plot_plasma_data()
    if plot_index:
        hopedata.plot_index_data()
    if plot_sw:
        hopedata.plot_sw_data()
    if save_csv:
        hopedata.save_csv_data()

    return hopedata

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some arguments for MyClass.")
    parser.add_argument('--plot_plasma', type=str, help='plot_plasma bool')
    parser.add_argument('--plot_index', type=str, help='plot_index bool')
    parser.add_argument('--plot_sw', type=str, help='plot_sw bool')
    parser.add_argument('--save_csv', type=str, help='save_csv bool')
    
    args = parser.parse_args()
    
    if args.plot_plasma is not None:
        plot_plasma = True 
    else:
        plot_plasma = False
        
    if args.plot_index is not None:
        plot_index = True
    else:
        plot_index = False
        
    if args.plot_sw is not None:
        plot_sw = True
    else:
        plot_sw = False
        
    if args.save_csv is not None:
        save_csv = True
    else:
        save_csv = False
    
    hope_data = create_ml_dataset('972237', 'h', plot_plasma = plot_plasma, plot_index = plot_index, plot_sw = plot_sw, save_csv = save_csv)        

 
        
        