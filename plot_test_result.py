"""
functions to make plots for machine learning project of ions in the magentosphere

Examples
--------



"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib as mpl
import math
import pytplot
from time_double import time_double
import datetime as datetime
import matplotlib.dates as mdates

import prepare_ml_dataset
import plot_functions
random_seed = 42

tf.random.set_seed(random_seed)
np.random.seed(random_seed)
np.set_printoptions(precision=4)

def plot_test_heatmap(energy, species, recalc = False, plot_data = False, save_data = True, release = 'rel05', dL01=True,coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], feature_names=[ 'scaled_symh', 'scaled_ae','scaled_asyh', 'scaled_asyd'], forecast = "none", number_history = 7):
    directories, dataset_csv, data_settings = prepare_ml_dataset.initializ_var(energy, species, release = release, dL01=dL01, feature_names=feature_names, forecast = forecast, number_history = number_history)
    
    x_test, y_test = prepare_ml_dataset.load_test_data(dataset_csv)
       
    model = (tf.keras.models.load_model(directories["model_dir" + dataset_csv["model_filename"]]))

    y_test_pred = model.predict(x_test)
    
    plot_functions.plot_correlation_heatmap(y_test, y_test_pred, xrange=[1,8], figname = directories["result_dir"] + data_settings["y_name_log"]+'_test_r2')

def plot_test_tplot(energy, species, release = 'rel05', dL01=True, feature_names=[ 'scaled_symh', 'scaled_ae','scaled_asyh', 'scaled_asyd'], forecast = "none", number_history = 7):
    """
        The following sections are to visulize the long-term variation of modeled proton flux
    """

    directories, dataset_csv, data_settings = prepare_ml_dataset.initializ_var(energy, species, release = release, dL01=dL01, feature_names=feature_names, forecast = forecast, number_history = number_history)

    df_full, index_good, data_settings["feature_history_names"] = plot_functions.create_df_full(release, data_settings["y_name"] , data_settings["y_name_log"] ,  data_settings["feature_names"] , data_settings["dL01"] , data_settings["number_history"] , data_settings["history_resolution"] , data_settings["average_time"], data_settings["forecast"] , directories["data_dir"] )
    
    index_train, index_valid, index_test = plot_functions.create_ml_indexes(df_full,  data_settings["test_ts"], data_settings["test_te"] , index_good)
    
    x_test, y_test = prepare_ml_dataset.load_test_data(dataset_csv)

    model = tf.keras.models.load_model(directories["model_dir"] + dataset_csv["model_filename"])

    y_test_pred = model.predict(x_test)

    omni_ts, y_data_ts, y_pred_ts, y_diff_ts = plot_functions.create_time_series_variables(df_full, index_test, data_settings["y_name_log"], y_test_pred, to_plot_omni_list=['symh'], to_plot_omni_label_list = ['SymH (nT)'])

    plot_functions.plot_test_tplot(omni_ts, y_data_ts, y_pred_ts, y_diff_ts, filename = directories["result_dir"] +'test_result')
    
    
    

    
