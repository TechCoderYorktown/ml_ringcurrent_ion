"""
functions to train and validate data

Examples
--------

from train_nn_model import train_nn_model 

train_nn_model('972237', 'h', number_history = 7)
train_nn_model('51767680', 'h', number_history = 7)



"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib as mpl
import tensorflow as tf
# import plot_test_result
import os
# from datetime import datetime
import prepare_ml_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # This is to disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

def create_directories(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok = True)
        
def plot_loss_function_history(history ,figname="tmp.png",ylim=0):
    fig1 = plt.figure(figsize=(10, 8),facecolor='w')
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(pd.DataFrame(history.history))
    ax1.grid(True)
    if ylim != 0:
        plt.gca().set_ylim(ylim[0],ylim[1])
    fontsize1=20
    ax1.set_xlabel("Epoches",fontsize=fontsize1)
    ax1.set_ylabel("Loss",fontsize=fontsize1)
    
    ax1.legend(["Train","Validation"],fontsize=fontsize1)
    
    ax1.tick_params(axis='both', which='major', labelsize=fontsize1)
    ax1.tick_params(axis='both', which='minor', labelsize=fontsize1)
    
    plt.savefig(figname+".png", format="png", dpi=300)
    
    plt.show()

def plot_loss_function_historys(total_history,para_set,para_str = '', figname="tmp.png",ylim=0, dataset_name='val_loss'):
#     fig1 = plt.figure(figsize=(10, 8),facecolor='w')   
    for ihistory in range(len(total_history)):
        history = total_history[str(para_set[ihistory])]
        plt.plot(history[dataset_name], label =  para_str + '%s' % para_set[ihistory])
        
    plt.legend()
        
    plt.grid(True)
    if ylim != 0:
        plt.gca().set_ylim(ylim[0],ylim[1])

#     plt.set_xlabel("Epoches",fontsize=20)
#     plt.set_ylabel("Validation Loss",fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=20)
#     plt.tick_params(axis='both', which='minor', labelsize=20)
    
    plt.savefig(figname+'_'+dataset_name+".png", format="png", dpi=300)
    
    plt.show()
    
def factor_line_calculation(xrange, factor):
    yrange = xrange
    yrangeup = [xrange[0] + np.log10(factor),xrange[1] + np.log10(factor)]
    yrangelow = [xrange[0] - np.log10(factor),xrange[1] - np.log10(factor)]
    return(yrange, yrangeup, yrangelow)
    
def plot_correlation_heatmap(y_test_reshaped, y_test_pred_reshaped, xrange=[4,9],figname="tmp"):
    corr = r2_score(y_test_reshaped, y_test_pred_reshaped)
    mse_test1 = sum((y_test_pred_reshaped-y_test_reshaped)**2)/len(y_test_reshaped)

    #Plot data vs model predictioin
    grid=0.05 
    
    yrange, yrangeup, yrangelow = factor_line_calculation(xrange, 2)

    NX=int((xrange[1]-xrange[0])/grid)
    NY=int((yrange[1]-yrange[0])/grid)
    M_test=np.zeros([NX,NY],dtype=np.int16)

    for k in range(y_test_reshaped.size):
        xk=(y_test_reshaped[k]-xrange[0])/grid
        yk=(y_test_pred_reshaped[k]-yrange[0])/grid
        xk=min(xk,NX-1)
        yk=min(yk,NY-1)
        xk=int(xk)
        yk=int(yk)

        M_test[xk,yk]+=1

    extent = (xrange[0], xrange[1], yrange[0], yrange[1])

    # Boost the upper limit to avoid truncation errors.
    levels = np.arange(0, M_test.max(), 200.0)

    norm = mpl.cm.colors.Normalize(vmax=M_test.max(), vmin=M_test.min())

    fig2=plt.figure(figsize=(10, 8),facecolor='w')
    ax1=fig2.add_subplot(1,1,1)

    im = ax1.imshow(M_test.transpose(),  cmap=mpl.cm.plasma, interpolation='none',#'bilinear',
                origin='lower', extent=[xrange[0],xrange[1],yrange[0],yrange[1]],
                vmax=M_test.max(), vmin=-M_test.min())

    ax1.plot(xrange,yrange,'r')
    ax1.plot(xrange, yrangeup,'r',dashes=[3, 3])
    ax1.plot(xrange, yrangelow,'r', dashes = [3,3])

    ax1.set_title(figname)#,fontsize=10)
    ax1.set_xlabel("Measured flux in log10",fontsize=20)
    ax1.set_ylabel("Predicted flux in log10",fontsize=20)

    plt.text(5,3.6,('R2: %(corr)5.3f' %{"corr": corr}),color='w',fontsize=20)
    plt.text(5,2.5,('mse_test:%(mse_test)5.3f' %{"mse_test":mse_test1}),color='w',fontsize=20)

    # We change the fontsize of minor ticks label 
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    
    #plt.axis('equal')
    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])
    cbar=fig2.colorbar(im, ax=ax1)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('# of 5-minute data', fontsize=20)
    plt.savefig(figname+".png", format="png", dpi=300)
    plt.show()

def nn_model(x_train, y_train, x_valid, y_valid, y_name, output_dir = 'training/', model_fln = '', mse_fln = '', extra_str = '',n_neurons = 15, dropout_rate = 0.0, patience = 32,learning_rate = 1e-3, epochs = 200, batch_size = 32, dL01 = True):
    
    loss_function = "mean_squared_error"
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate) #Adam(learning_rate = learning_rate)) 
    
    tf.random.set_seed(42)
    
    callback  =  tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience)

    model  =  tf.keras.models.Sequential([
        tf.keras.layers.Input(x_train.shape[1:]), 
        tf.keras.layers.Dense(units = n_neurons, activation = "relu"), # , kernel_regularizer = tf.keras.regularizers.L2(0.01)),
        tf.keras.layers.Dense(units = n_neurons, activation = "relu", input_shape = (n_neurons + 1,0)),
        tf.keras.layers.Dense(units = n_neurons, activation = "relu", input_shape = (n_neurons + 1,0)),
#       tf.keras.layers.Dense(units = n_neurons, activation = "relu", input_shape = (n_neurons + 1,0)),
        tf.keras.layers.Dense(1)
    ])
    
    print(model.summary())
    
    model.compile(loss = loss_function, optimizer = optimizer) 
    
    history  =  model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_valid, y_valid), callbacks = [callback])
    
    fln = output_dir + extra_str + y_name + '_dL01' + str(dL01) + '_' + str(n_neurons) + '_neurons_' + str(len(model.layers)) + '_layers_' + str(dropout_rate) + '_dropout_' + str(patience) + '_patience_' + str(learning_rate) + '_learning_rate_' + str(epochs) + '_epochs_' + str(batch_size) + '_batchsize_'# + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if model_fln == '':
        model_fln = fln  + ".h5"
    if mse_fln == '':
        mse_fln = fln + '_mse.png'
    
    model.save(model_fln)
    print(history)

    # plot the mse loss function at each epoch
    plot_loss_function_history(history, figname = mse_fln, ylim = [0, 0.3]) 
   
    # calculate the accuracy of validation data
    y_valid_pred = model.predict(x_valid)

    plot_correlation_heatmap(y_valid, y_valid_pred.reshape([-1]), xrange=[1,8], figname = fln+'_validation_result')
    valid_r2 = r2_score(y_valid_pred.reshape([-1]), y_valid.reshape([-1]))

    return(model, history,valid_r2)

def train_nn_model(energy, species, recalc = False, plot_data = False, save_data = True, release = 'rel05', dL01=True,coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], feature_names=[ 'scaled_symh', 'scaled_ae','scaled_asyh', 'scaled_asyd'], forecast = False, number_history = 7):

    directories, dataset_csv, data_settings = prepare_ml_dataset.initializ_var(energy, species, release = release, dL01=dL01, feature_names=feature_names, forecast = forecast, number_history = number_history)

    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_ml_dataset.prepare_ml_dataset(energy, species, recalc = recalc, plot_data = plot_data, save_data = save_data, release = release, dL01=dL01, feature_names=feature_names, forecast = forecast, number_history = number_history)
    
    has_nan = np.any(np.isnan(x_train)) | np.any(np.isnan(x_valid)) | np.any(np.isnan(x_test)) | np.any(np.isnan(y_train)) | np.any(np.isnan(y_valid)) | np.any(np.isnan(y_test)) 
    
    if has_nan:
        print("data has nan")
        return False
    
    
    para_name = "learning_rate"
    para_set = [1.e-4, 1.5e-3, 1.e-3]

    final_train_loss = np.zeros(len(para_set))
    final_valid_loss = np.zeros(len(para_set))
    total_history = dict()
    valid_r2s = np.zeros(len(para_set))

    for ipara in range(len(para_set)):
        parameter = para_set[ipara]

        model, history, valid_r2 = nn_model(x_train, y_train, x_valid, y_valid, data_settings["y_name"], output_dir = directories["training_output_dir"] , model_fln = '', mse_fln = '', n_neurons = 18, dropout_rate = 0.0, patience = 32, learning_rate = parameter, epochs = 20, batch_size = 8, dL01= dL01)
        
        total_history[str(parameter)] = history.history
        final_train_loss[ipara] = history.history['loss'][-1]
        final_valid_loss[ipara] = history.history['val_loss'][-1]
        valid_r2s[ipara] = valid_r2
    
    print(para_set, final_valid_loss,valid_r2s)

    plot_loss_function_historys(total_history, para_set, para_str = para_name, figname = directories["model_setting_compare_dir"] + data_settings["y_name"]+'_'+para_name, dataset_name='val_loss', ylim = [0,0.5])

    plot_loss_function_historys(total_history, para_set, para_str = para_name, figname = directories["model_setting_compare_dir"] + data_settings["y_name"]+'_'+para_name, dataset_name='loss', ylim = [0,0.5])

    plt.plot(np.array(para_set), valid_r2s, marker = 'o')
    plt.title(para_name)
    plt.xlabel(data_settings["y_name"])
    plt.ylabel('Validation r2')
    plt.savefig(output_dir = directories["training_output_dir"]  + data_settings["y_name"]+'_'+para_name+'_r2.png', format="png", dpi=300)
    
    return True

def __main__():
    if __name__ == "__name__":
        prepare_ml_dataset('972237', 'h')
