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
from pyspedas import time_double, time_string
import datetime as datetime
import matplotlib.dates as mdates

def scale_arr_with_input(arr, mid_value, scale_value):
    scaled_arr = (arr - mid_value)/scale_value*2
#     print(mid_value, scale_value)
    return(scaled_arr)


def plot_global_distributions(df_omni, models, exmaple_start_time,exmaple_end_time, cut_times, y_names, coor_names, feature_history_names, vmax = 6, vmin = 3, lat_setting = 0, output_filename = 'global.png', v_label = ''):
    if v_label == '':
        v_label = y_names
    index_example = (df_omni['Datetime'] >=exmaple_start_time) &  (df_omni['Datetime'] <= exmaple_end_time) & (df_omni['ae'] > 0)
    time_arr_example = df_omni.loc[index_example,'Datetime'].astype('datetime64[ns]').reset_index(drop=True)

    pytplot.store_data("SymH", data={'x':time_arr_example, 'y':df_omni.loc[index_example, 'symh']})
    pytplot.store_data("AsyH", data={'x':time_arr_example, 'y':df_omni.loc[index_example, 'asyh']})
    pytplot.store_data("AsyD", data={'x':time_arr_example, 'y':df_omni.loc[index_example,'asyd']})
    pytplot.store_data("AE", data={'x':time_arr_example, 'y':df_omni.loc[index_example,'ae']})

    tplot_names = ["SymH","AE"]
    pytplot.options(tplot_names, 'thick', 1.2)
    pytplot.timebar(time_double(cut_times), color='green', dash=True)

    pytplot.tplot(tplot_names)

    mlt_example_grid = 0.1
    mlt_example_range = [0, 24]
    mlts_example = np.array(range(int(mlt_example_range[1]/mlt_example_grid)))*mlt_example_grid
    theta_example = mlts_example/12*math.pi
    n_theta = theta_example.shape[0]

    l_example_grid = 0.1
    l_example_range = [2, 6.5]
    l_example = np.array(range(int((l_example_range[1] - l_example_range[0])/l_example_grid)))*l_example_grid + l_example_range[0]
    n_l = l_example.shape[0]

    l_example_mesh, theta_example_mesh = np.meshgrid(l_example, theta_example)

    df_example = pd.DataFrame(data = {'theta': theta_example_mesh.reshape([-1]), 'l': l_example_mesh.reshape([-1])})

    df_example['lat'] = lat_setting
    df_example['scaled_lat'] = df_example['lat'].apply(lambda x: scale_arr_with_input(x, 44.388, 49.445))
    df_example['scaled_l'] = df_example['l'].apply(lambda x: scale_arr_with_input(x, 4.661, 6.408))

    df_example['sin0'] = df_example['theta'].apply(np.sin)
    df_example['cos0'] = df_example['theta'].apply(np.cos)
    df_example['x'] = -df_example['l']*df_example['cos0']
    df_example['y'] = -df_example['l']*df_example['sin0']

    coor_mesh = df_example.loc[:,coor_names]

    fig_global, ax = plt.subplots(2,len(cut_times), figsize=(14, 3.6))#, facecolor = 'white')

    m_feature_history = len(feature_history_names)

    for iy in [0, 1]:
        inner_model = tf.keras.models.load_model(models[iy])

        for icut in range(len(cut_times)):

            time_cut = cut_times[icut]

            cut = df_omni.loc[(df_omni['Datetime'] == time_cut) & index_example, :]
            cut = cut.iloc[0,:]

            feature_history_example_mesh = np.matmul(np.ones([n_theta*n_l,1]), (np.float32(cut[feature_history_names])).reshape([1,m_feature_history]))

            x_example_mesh = np.concatenate((coor_mesh, feature_history_example_mesh), axis = 1)

            y_example_pred_mesh = inner_model.predict(x_example_mesh, verbose = 0)
            y_example_pred = y_example_pred_mesh.reshape([n_theta,n_l])

            im = ax[iy,icut].contourf(np.array(df_example['x']).reshape([n_theta,n_l]), np.array(df_example['y']).reshape([n_theta,n_l]), y_example_pred, 256, cmap = mpl.cm.jet, vmax = vmax, vmin = vmin)

            #Add Earth
            theta=np.arange(0,2.01*np.pi,0.1)
            ax[iy,icut].plot(np.cos(theta),np.sin(theta),'k')
            r=np.arange(0,1.1,0.1)
            theta1=np.arange(0.5*np.pi,1.5*np.pi,0.02)
            R,THETA=np.meshgrid(r,theta1)
            X1=R*np.cos(THETA)
            Y1=R*np.sin(THETA)
            im_earth = ax[iy,icut].contourf(X1,Y1,X1*0+0.0,10,cmap=mpl.cm.cubehelix,vmax=1.0,vmin=0.0)
            #Add 0, 6, 12, 18 sectors and L=4,6
            ax[iy,icut].plot([-6,6],[0,0],':k',linewidth=1)
            ax[iy,icut].plot([0,0],[-6,6],':k',linewidth=1)
            ax[iy,icut].plot(6*np.cos(theta),6*np.sin(theta),':k')
            ax[iy,icut].plot(4*np.cos(theta),4*np.sin(theta),':k')
            ax[iy,icut].set_aspect(1)
            ax[iy,icut].invert_xaxis()
            ax[iy,icut].invert_yaxis()

            if iy == 0:
                ax[iy,icut].set_title(time_cut[0:19],fontsize=12)
        cbar = fig_global.colorbar(im, ticks = range(vmin-1, vmax, 1))
        cbar.set_label(v_label[iy], fontsize=11)

    plt.savefig(output_filename, format="png", dpi=300)


# -- plot global distribution in a given fig and ax --
def plot_global_distribution(fig_global, ax, df_omni, model, time_cut, y_name, coor_names, feature_history_names, vmax = 6, vmin = 3, lat_setting = 0, output_filename = 'global.png', v_label = '', location_x = 2, location_y=2):
    if v_label == '':
        v_label = y_name

    mlt_example_grid = 0.1
    mlt_example_range = [0, 24]
    mlts_example = np.array(range(int(mlt_example_range[1]/mlt_example_grid)))*mlt_example_grid
    theta_example = mlts_example/12*math.pi
    n_theta = theta_example.shape[0]

    l_example_grid = 0.1
    l_example_range = [2.5, 6.5]
    l_example = np.array(range(int((l_example_range[1] - l_example_range[0])/l_example_grid)))*l_example_grid + l_example_range[0]
    n_l = l_example.shape[0]

    l_example_mesh, theta_example_mesh = np.meshgrid(l_example, theta_example)

    df_example = pd.DataFrame(data = {'theta': theta_example_mesh.reshape([-1]), 'l': l_example_mesh.reshape([-1])})

    df_example['lat'] = lat_setting
    df_example['scaled_lat'] = df_example['lat'].apply(lambda x: scale_arr_with_input(x, 44.388, 49.445))
    df_example['scaled_l'] = df_example['l'].apply(lambda x: scale_arr_with_input(x, 4.661, 6.408))

    df_example['sin0'] = df_example['theta'].apply(np.sin)
    df_example['cos0'] = df_example['theta'].apply(np.cos)
    df_example['x'] = -df_example['l']*df_example['cos0']
    df_example['y'] = -df_example['l']*df_example['sin0']

    coor_mesh = df_example.loc[:,coor_names]

    fs_label = 10
    cmap = mpl.cm.jet
    m_feature_history = len(feature_history_names)

    inner_model = tf.keras.models.load_model(model)

    cut_feature_history = df_omni.loc[(df_omni['Datetime'] == time_cut) & (df_omni['ae'] > 0), feature_history_names]

    feature_history_example_mesh = np.matmul(np.ones([n_theta*n_l,1]), np.float32(cut_feature_history).reshape([1,m_feature_history]))

    x_example_mesh = np.concatenate((coor_mesh, feature_history_example_mesh),axis=1)

    y_example_pred_mesh = inner_model.predict(x_example_mesh,verbose = 0)
    y_example_pred = y_example_pred_mesh.reshape([n_theta,n_l])
    ax.set_position([location_x, location_y, 0.3, 0.4])

    im = ax.contourf(np.array(df_example['x']).reshape([n_theta,n_l]), np.array(df_example['y']).reshape([n_theta,n_l]), y_example_pred, 256, cmap = cmap, vmax = vmax, vmin = vmin)

    #Add Earth
    theta=np.arange(0,2.01*np.pi,0.1)
    ax.plot(np.cos(theta),np.sin(theta),'k')
    r=np.arange(0,1.1,0.1)
    theta1=np.arange(0.5*np.pi,1.5*np.pi,0.02)
    R,THETA=np.meshgrid(r,theta1)
    X1=R*np.cos(THETA)
    Y1=R*np.sin(THETA)
    im2=ax.contourf(X1,Y1,X1*0+0.0,10,cmap=mpl.cm.cubehelix,vmax=1.0,vmin=0.0)
    #Add 0, 6, 12, 18 sectors and L=4,6
    ax.plot([-6,6],[0,0],':k',linewidth=1)
    ax.plot([0,0],[-6,6],':k',linewidth=1)
    ax.plot(6*np.cos(theta),6*np.sin(theta),':k')
    ax.plot(4*np.cos(theta),4*np.sin(theta),':k')

    ax.axis('equal')
    ax.set_xlim([6.5,-6.5])
    ax.set_ylim([6.5,-6.5])
    ax.set_title(time_cut,fontsize=fs_label)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm = mpl.cm.colors.Normalize(vmax=vmax, vmin=vmin), cmap = cmap) ,cax=plt.axes([location_x + 0.31, location_y + 0.025, 0.01, 0.35]))
    cbar.ax.tick_params(labelsize = fs_label)
    cbar.set_label(v_label, fontsize=fs_label)
    if output_filename != '':
        plt.savefig(output_filename, format="png",pad_inches = 5)#, dpi=300)

# define time series class and creation
def create_time_series_class(x, y, v = [], xlabel = 'x', ylabel = 'y',vlabel = 'v', linecolors = 'k',plot_style = 'line',vrange=[],cmap='jet'):
    class time_series():
        x = []
        y = []
        v = []
        ylabel = 'y'
        xlabel = 'x'
        vlabel = 'v'
        linecolors = 'k'
        plot_style = 'line'
        vrange=[]
        cmap='jet'
    output = time_series()
    output.x = x
    output.y = y
    output.v = v
    output.ylabel = ylabel
    output.xlabel = xlabel
    output.vlable = vlabel
    output.linecolors = linecolors
    output.plot_style = plot_style
    output.vrange=vrange
    output.cmap=cmap

    return(output)

# -- function to create a time_series class variable --
def create_time_series_variables(df_full, index_plot, y_names, y_pred_reshaped, to_plot_omni_list=['symh'], to_plot_omni_label_list = ['SymH (nT)']) :
    if(len(to_plot_omni_list) != len(to_plot_omni_label_list)):
        print("to_plot_omni_list must have the same length as to_plot_omni_label")

    time_test = df_full['Datetime'].astype('datetime64[ns]')[index_plot]

    omni_ts = list()
    for i_to_plot_omni in range(len(to_plot_omni_list)):
        to_plot_omni = to_plot_omni_list[i_to_plot_omni]
        to_plot_omni_label = to_plot_omni_label_list[i_to_plot_omni]
        omni_ts.append(create_time_series_class(time_test, df_full[to_plot_omni][index_plot], ylabel=to_plot_omni_label))

    y_data_ts = list()
    y_pred_ts = list()
    y_diff_ts = list()

    for iy in range(len(y_names)):
        y_name = y_names[iy]
        y_data_ts.append(create_time_series_class(time_test, df_full.loc[index_plot,'l'], v = df_full.loc[index_plot,y_name], ylabel = 'L (data)', vlabel = y_name, plot_style = 'scatter', vrange=[1,6], cmap='jet'))
        y_pred_ts.append(create_time_series_class(time_test, df_full.loc[index_plot,'l'], v = y_pred_reshaped[iy], ylabel = 'L (model)', plot_style = 'scatter', vrange=[1,6], cmap='jet'))
        y_diff_ts.append(create_time_series_class(time_test, df_full.loc[index_plot,'l'], v = y_pred_reshaped[iy] - df_full.loc[index_plot,y_name], ylabel = 'L (model-data)', plot_style = 'scatter', vrange=[-2,2], cmap='bwr'))

    return(omni_ts, y_data_ts, y_pred_ts, y_diff_ts)


# --- function to draw panel in a time tplot using matplotlib  ---
def draw_panel_in_time_plot(fig, ax, obj0, last_panel = False, label_size = 20, time_cut = ''):
    ax.margins(x=0)
#     ax.autoscale_view('tight')
    if obj0.plot_style == 'line':
        s1 = ax.plot(obj0.x, obj0.y, c = obj0.linecolors)
    elif obj0.plot_style == 'scatter':
        if obj0.vrange:
            vmin = obj0.vrange[0]
            vmax = obj0.vrange[1]
        else:
            vmin = min(obj0.v)
            vmax = max(obj0.v)
        s1=ax.scatter(obj0.x, obj0.y, c=obj0.v, cmap = obj0.cmap, alpha = 0.9, vmin = vmin, vmax = vmax)

        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

        cbar = fig.colorbar(s1, ax=ax,fraction=0.02,pad=0.01)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(obj0.vlabel, fontsize=10)

                # draw y label
    ax.set_ylabel(obj0.ylabel,fontsize=label_size)

    # draw the bottom ticks
    if last_panel:
        ax.tick_params(axis='x',labelbottom=True) # labels along the bottom edge are on
        ax.set_xlabel('Time',fontsize=label_size)

    else:
        ax.tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off

    n_ts = obj0.x.shape[0]
    delta_t_days = obj0.x.reset_index(drop=True)[n_ts-1]-obj0.x.reset_index(drop=True)[0]

    if ( delta_t_days < datetime.timedelta(days=30) ):
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.tick_params(axis='both', which='minor', labelsize=label_size)

    if time_cut != '':
        ax.axvline(x = time_cut, color = 'c')


def plot_test_tplot(omni_ts, y_data_ts, y_pred_ts, y_diff_ts, filename = ''):

    n_panels = len(omni_ts) + 3*len(y_data_ts)
    
    fig, axs = plt.subplots(n_panels, 1, figsize=(10, 8), facecolor = 'white')

    last_panel = False
    for ipanel in range(len(omni_ts)):
        draw_panel_in_time_plot(fig, axs[ipanel], omni_ts[ipanel], last_panel = False)
    for iy in range(len(y_data_ts)):
        draw_panel_in_time_plot(fig,axs[len(omni_ts)+iy*3], y_data_ts[iy], last_panel = False)
        draw_panel_in_time_plot(fig,axs[len(omni_ts)+iy*3+1], y_pred_ts[iy], last_panel = False)

        if ipanel == len(y_data_ts)-1:
            last_panel = True

        draw_panel_in_time_plot(fig,axs[len(omni_ts)+iy*3+2], y_diff_ts[iy], last_panel = last_panel)
#     mpl.rcParams['pdf.fonttype']=42
#     mpl.rcParams['ps.fonttype']=42
    if filename != '':
        fig.savefig(filename+".png", format="png", dpi=300)

    plt.show()


def plot_tplot_and_global_distribution(df_omni, to_plot_omni_list, to_plot_omni_label_list, index_plot, models,  time_cut, y_names, coor_names, feature_history_names, output_filename = 'global.png', v_label = ''):

    fig_global, axs = plt.subplots(2+len(models), 1 , constrained_layout=True)

    time_plot = df_omni['Datetime'].astype('datetime64[ns]')[index_plot]

    omni_ts = list()
    for i_to_plot_omni in range(len(to_plot_omni_list)):
        omni_ts.append(create_time_series_class(time_plot, df_omni[to_plot_omni_list[i_to_plot_omni]][index_plot], ylabel = to_plot_omni_label_list[i_to_plot_omni]))

    n_panels = len(omni_ts)
    for ipanel in range(n_panels):
        if ipanel == n_panels-1:
            last_panel = True
        else:
            last_panel = False
        draw_panel_in_time_plot(fig_global, axs[ipanel], omni_ts[ipanel], last_panel = last_panel,label_size = 10,time_cut=[time_cut])

    for imodel in range(len(models)):
        model = models[imodel]
        y_name = y_names[imodel]
        plot_global_distribution(fig_global, axs[ipanel+imodel+1], df_omni, model, time_cut, y_name, coor_names, feature_history_names, output_filename = '', v_label = '', location_x = imodel*0.51+0.08,location_y = 0)

    if output_filename != '':
        fig_global.savefig(output_filename+".png", format="png", dpi=300)

    plt.show()


    
    
