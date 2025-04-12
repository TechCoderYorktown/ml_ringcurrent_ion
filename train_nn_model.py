"""
functions to train and validate data

Examples
--------

from train_nn_model import train_nn_model 

train_nn_model('972237', 'h', number_history = 7)
train_nn_model('51767680', 'h', number_history = 7)

"""

import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
import os

import prepare_ml_dataset
import initialize_var
import plot_functions

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # This is to disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

random_seed = 42

tf.random.set_seed(random_seed)
np.random.seed(random_seed)
np.set_printoptions(precision=4)

def nn_model(x_train, y_train, x_valid, y_valid, y_name, output_dir = 'training/', model_fln = '', mse_fln = '', extra_str = '',n_neurons = 18, dropout_rate = 0.0, patience = 32,learning_rate = 1e-3, epochs = 200, batch_size = 32, dL01 = True, nlayer = 3):
    
    loss_function = "mean_squared_error"
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate) #Adam(learning_rate = learning_rate)) 
    
    tf.random.set_seed(42)
    
    callback  =  tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience)

    nlayer = nlayer

    if nlayer == 3:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(x_train.shape[1:]), 
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"), # , kernel_regularizer = tf.keras.regularizers.L2(0.01)),
            # Dropout(0.5),
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"),
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"),
    #       tf.keras.layers.Dense(units = n_neurons, activation = "relu"),
            tf.keras.layers.Dense(1)
        ])
    if nlayer == 4:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(x_train.shape[1:]), 
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"), # , kernel_regularizer = tf.keras.regularizers.L2(0.01)),
            # Dropout(0.5),
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"),
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"),
            tf.keras.layers.Dense(units = n_neurons, activation = "relu"),
            tf.keras.layers.Dense(1)
        ])
    
    print(model.summary())
    
    model.compile(loss = loss_function, optimizer = optimizer) 
    
    history  =  model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_valid, y_valid), callbacks = [callback])
    
    fln = output_dir + extra_str + y_name + '_dL01' + str(dL01) + '_' + str(n_neurons) + '_neurons_' + str(len(model.layers)) + '_layers_' + str(dropout_rate) + '_dropout_' + str(patience) + '_patience_' + str(learning_rate) + '_learning_rate_' + str(epochs) + '_epochs_' + str(batch_size) + '_batchsize_' # + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    if model_fln == '':
        model_fln = fln  + ".h5"
    if mse_fln == '':
        mse_fln = fln + '_mse'
    
    model.save(model_fln)
    print(history)

    # plot the mse loss function at each epoch
    plot_functions.plot_loss_function_history(history, figname = mse_fln, ylim = [0, 0.5]) 
    
    # calculate the accuracy of validation data
    y_valid_pred = model.predict(x_valid)

    plot_functions.plot_correlation_heatmap(y_valid.reshape([-1]), y_valid_pred.reshape([-1]), xrange=[0,10], figname = fln+'_validation_result')
    valid_r2 = r2_score(y_valid_pred.reshape([-1]), y_valid.reshape([-1]))

    return model, history,valid_r2

def train_nn_model(energy, species, recalc = False, plot_data = True, save_data = True, dL01=True, raw_feature_names=['symh','asyh','ae','asyd'], forecast = "none", number_history = 7, nlayer = 3, learning_rate = 1.e-3, n_neurons = 18, dropout_rate = 0.0, patience = 32, epochs = 2, batch_size = 8, release = 'rel05'):
        
    '''
    train_nn_model is routine to train a nn model.
    
    raw_feature_names: ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz']
    forecast: ["all", "index","none"]
    number_history : [7,8] # or any number
    energy : [51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237] * 1000
    species : 'h', 'o'
    dL01_arr = [True, False]
    
    '''

    
    np.set_printoptions(precision=4)
    
    data_directories, dataset_csv, data_settings = initialize_var.initialize_data_var(energy=energy, species=species, raw_feature_names = raw_feature_names, forecast = forecast, number_history = number_history, dL01=dL01)

    model_directories, modelset_csv, model_settings = initialize_var.initialize_model_var(nlayer, n_neurons, dropout_rate,patience, learning_rate, epochs,  batch_size, energy , species, raw_feature_names, number_history, dL01, forecast, release = release)

    print("start loading data")
    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_ml_dataset.load_ml_dataset(energy, species, recalc = recalc, plot_data = plot_data, save_data = save_data, dL01=dL01, raw_feature_names = raw_feature_names,  forecast = forecast, number_history = number_history)   
    
    has_nan = np.any(np.isnan(x_train)) | np.any(np.isnan(x_valid)) | np.any(np.isnan(x_test)) | np.any(np.isnan(y_train)) | np.any(np.isnan(y_valid)) | np.any(np.isnan(y_test)) 
    if has_nan:
        print("data has NaN")
        return False

    para_name = "learning_rate"
    para_set = [5.e-5, 1.5e-4, 5.e-4]

    final_train_loss = np.zeros(len(para_set))
    final_valid_loss = np.zeros(len(para_set))
    total_history = dict()
    valid_r2s = np.zeros(len(para_set))

    print("start to train")

    for ipara in range(len(para_set)):
        parameter = para_set[ipara]
        
        print(para_name + '=' + str(parameter))
        print(data_settings)
        model, history, valid_r2 = nn_model(x_train, y_train, x_valid, y_valid, data_settings["log_y_name"], output_dir = model_directories["training_output_dir"] , model_fln = '', mse_fln = '', n_neurons = n_neurons, dropout_rate = dropout_rate, patience = patience, learning_rate = parameter, epochs = epochs, batch_size = batch_size, dL01= dL01, nlayer= nlayer)        
        
        total_history[str(parameter)] = history.history
        final_train_loss[ipara] = history.history['loss'][-1]
        final_valid_loss[ipara] = history.history['val_loss'][-1]
        valid_r2s[ipara] = valid_r2
    
    print(para_set, final_valid_loss,valid_r2s)

    # plot_functions.plot_loss_function_historys(total_history, para_set, para_str = para_name, figname = directories["model_setting_compare_dir"] + data_settings["y_name"]+'_'+para_name, dataset_name='val_loss', ylim = [0,0.5])

    # plot_functions.plot_loss_function_historys(total_history, para_set, para_str = para_name, figname = directories["model_setting_compare_dir"] + data_settings["y_name"]+'_'+para_name, dataset_name='loss', ylim = [0,0.5])
    
    plot_functions.plot_training_comparisons(para_set, para_name, valid_r2s, data_settings, model_directories)
   
    return True

def __main__():
    if __name__ == "__name__":
        train_nn_model('51767680', 'h')
