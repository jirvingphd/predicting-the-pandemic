## to replace af.get_state_ts

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# pio.templates.default = "plotly_dark"
import pandas as pd

import numpy as np
# import kaggle


import datetime as dt

# def download_us_pop(data_folder = "./reference_data/",load=True):
#     """Downloads world pop zip from kaggle"""
#     import os,sys,shutil

#     # Download kaggle dataset
#     os.system("kaggle datasets download -d peretzcohen/2019-census-us-population-data-by-state")
# #     os.system('kaggle datasets download -d tanuprabhu/population-by-country-2020')

#     ## Specify file and target folder
#     file = '2019-census-us-population-data-by-state.zip'
#     target = os.path.join(data_folder,file)

#     ## Move zip file to target
#     shutil.move(file,target)
#     print(f'File saved to {target}')
    
#     ## Load csv 
#     if load:
#         df = pd.read_csv(target)
#     else:
#         df = target
#     return df




def plot_map_columns(DF,col='Cases-New',last_n_days=90,
               map_metric='sum',plot_map=True,return_map=False):
    """
    """
    import datetime as dt

    today = dt.date.today()
    end_date = today
    start_date = pd.Timestamp(today) - pd.Timedelta(f'{last_n_days} days')

    plot_df = get_states_to_plot(DF,state_list=None, plot_cols=col,
                             agg_func= 'mean',
                      rename_cols=True,fill_method='interpolate',
                      remove_outliers=True, state_first=True,
                      threshold_type=['0','%'], diagnose=False)

    plot_df.columns = [c.split(' - ')[0] for c in plot_df.columns]
    plot_df = plot_df.loc[start_date:end_date]
    
    
    
    agg_data = plot_df.agg(map_metric).reset_index()
    agg_data.columns= ['state',col]
    
    color_column = col
    map_ = px.choropleth(agg_data,color=color_column,locations='state',
                    locationmode="USA-states", scope='usa', 
                            title=f"{map_metric.title()} {col} by State (Last {last_n_days} Days)",#.format(color_column),
                    color_continuous_scale=px.colors.sequential.Reds)
    if plot_map:
        map_.show(config={'scrollZoom':False})

    if return_map:
            return map_
    else:
        return agg_data
    


def calc_perc_change(ts,periods=1,replace_inf_with_nan=True):
    """Calculated the pct_change with pandas and optionall replaces inf values"""
    ts_pct_change = ts.pct_change(periods=periods)#
    
    if replace_inf_with_nan:
        ts_pct_change = ts_pct_change.replace({np.inf:np.nan,
                                      -np.inf:np.nan})

    return ts_pct_change



### BEST VERSION YET! 09/12 ######
def get_state_df_ts_final(STATES, state_name,ts_col=None,group_col='state',# legacy compatible
                     freq='D', start_date='08-2020', agg_func='mean', #formerly sum
                    fill_nans=True, fill_method='interpolate',
                    rename_cols=True,name_sep=' - ',state_first=False,
                    remove_outliers=True,  n_diff=1, threshold_type='pct_change',
                      raw_thresh=100, pct_thresh=50, 
                          diagnose=True):
    """Take df_us and extracts state's data as then Freq/Aggregation provided
    
    
    Paramters:
    
        - DF
        - state_name
        - ts_col
        - group_col
        - freq 
        - start_date
        - agg_func
        -  fill_nans
        - fill_method
        
        
    Note on order:
    1. make state_df
    2. resample and aggregate
    3. slice start_date
    4. Select columns
    5. Remove Outliers
    6. Fill Null values
    7. Rename columns
    
    """
    ## Get default arguments for try/except
    import inspect
    args = inspect.signature(get_state_df_ts)
    args = {k:v.default for k,v in args.parameters.items()}  
    
    
    ## Slice state_df depending on which datatype 
    if isinstance(STATES,dict):
        state_df = STATES[state_name].copy()
        
    elif isinstance(STATES, pd.DataFrame):
        ## Slicing out state df for index dates 
        state_df = STATES.loc[state_name].copy()

    else:
        ## Get state_df group
        state_df = STATES.groupby(group_col).get_group(state_name)#.resample(freq).agg(agg)

        
        
    ## visualize pre-resampling
    if diagnose:
        pfig = px.line(state_df,title='Pre-Resampling')
        pfig.show()
    

    ## Resampling and Aggregating 
    if agg_func=='as_freq':
        try:
            state_df = state_df.resample(freq).asfreq(freq)
        except Exception as e:

            agg_func = args['agg_func']
            state_df = state_df.resample(freq).agg(agg_func)
            print(f"[!] Erorr using agg_func='as_freq'; Using default agg_func ('{agg_func}') instead.")
            print('\tError message below:')
            print("\t",e)        
            
    elif agg_func is None:
        pass
    
    else:
        state_df = state_df.resample(freq).agg(agg_func)
        
        
    ## Slice out time period desired.
    state_df = state_df.loc[start_date:]
    
    
    ## Return only columns containing ts_cols
    if ts_col is not None:

        if isinstance(ts_col,str):
            ts_col = [ts_col]

        # find cols that end with the column name
        selected_cols=[]
        for col in ts_col:
            selected_cols.extend([c for c in state_df.columns if c.endswith(col)])
            
        state_df = state_df[selected_cols]
        
        
    ## Remove Outleirs
    if remove_outliers:
        
        if isinstance(threshold_type,str):
            threshold_type= [threshold_type]
            
        for thresh_type in threshold_type:
            state_df = remove_outliers_ts(state_df,threshold_type=thresh_type,
                                         raw_thresh=raw_thresh,pct_thresh=pct_thresh,
                                         n_diff=n_diff,fill_method=fill_method)
        
   
    
    ## Deal with reamaining  null values:  (REMOVE??)
    if fill_method == 'interpolate':
        state_df = state_df.interpolate()
        
    elif fill_method == None:
        pass
    
    else:
        state_df = state_df.fillna(method=fill_method)
    
        

     ## Rename columns with state name
    if rename_cols == True:
    
        ## Get and Rename Sum Cols 
        orig_cols = state_df.columns

        for col in orig_cols:

            if state_first==True:
                new_col_name = f"{state_name}{name_sep}{col}"
            else:
                new_col_name = f"{col}{name_sep}{state_name}"

            state_df[new_col_name] = state_df[col].copy()

        ## Drop original cols
        state_df.drop(orig_cols,axis=1,inplace=True)
    

    ## Visualize post-resampling 
    if diagnose:
            pfig = px.line(state_df,title="post-Resampling")
            pfig.show()

    
    
    return state_df


def get_states_to_plot(DF,state_list=["NY",'MD','TX','PA', 'FL'],
                       plot_cols=None, 
                      agg_func= 'mean',
              rename_cols=True,fill_method='interpolate',
              remove_outliers=False, state_first=False,
              threshold_type=['0','%'], diagnose=False):
    
    get_states_kwargs = dict(rename_cols=rename_cols,
                          fill_method=fill_method,
                          state_first=state_first,
                           threshold_type=threshold_type,
                          diagnose=diagnose)
    
    if state_list is None:   
        if isinstance(DF,pd.DataFrame):
            state_list = list(DF.index.get_level_values(0).unique())
    
        elif isinstance(DF,dict):
            state_list = list(sorted(DF.keys()))
    
    ## Get each state
    dfs_to_concat = []
    for state in state_list:
        dfs = get_state_df_ts_final(DF,state,ts_col=plot_cols,**get_states_kwargs)
        dfs_to_concat.append(dfs)
        
        
     ## Concatenate final dfs
    try:
        plot_df = pd.concat(dfs_to_concat,axis=1)#[STATES[s] for s in plot_states],axis=1).iplot()
        new_order = sorted(plot_df.columns.to_list())
        plot_df = plot_df[new_order]
    except:
        print('[!] pd.concat failed, returning list...')
        plot_df = dfs_to_concat
    return plot_df


############################################################### PRE-09/21/21
def get_state_df_ts(STATES, state_name,ts_col=None,group_col='state',# legacy compatible
                     freq='D', start_date='08-2020', agg_func='mean', #formerly sum
                    fill_nans=True, fill_method='interpolate',
                    rename_cols=True,name_sep=' - ',state_first=True,diagnose=True):
    """Take df_us and extracts state's data as then Freq/Aggregation provided
    
    
    Paramters:
    
        - DF
        - state_name
        - ts_col
        - group_col
        - freq 
        - start_date
        - agg_func
        -  fill_nans
        - fill_method
    
    """
    import inspect
    args = inspect.signature(get_state_df_ts)
    args = {k:v.default for k,v in args.parameters.items()}  
    
    
    if isinstance(STATES,dict):
        state_df = STATES[state_name].copy()
        
    elif isinstance(STATES, pd.DataFrame):
        ## Slicing out state df for index dates 
        state_df = STATES.loc[state_name].copy()

    else:
        ## Get state_df group
        state_df = STATES.groupby(group_col).get_group(state_name)#.resample(freq).agg(agg)

        
    ## visualize pre-resampling
    if diagnose:
        pfig = px.line(state_df,title='Pre-Resampling')
        pfig.show()
    
    
    if agg_func=='as_freq':
        try:
            state_df = state_df.resample(freq).asfreq(freq)
        except Exception as e:

            agg_func = args['agg_func']
            state_df = state_df.resample(freq).agg(agg_func)
            print(f"[!] Erorr using agg_func='as_freq'; Using default agg_func ('{agg_func}') instead.")
            print('\tError message below:')
            print("\t",e)        
    elif agg_func is None:
        pass
    else:
        ## Resample and aggregate state data
        state_df = state_df.resample(freq).agg(agg_func)
        
    ## Slice out time period desired.
    state_df = state_df.loc[start_date:]
    

    
    
    ## Deal with null values:
    if fill_method == 'interpolate':
        state_df = state_df.interpolate()
        
    elif fill_method == None:
        pass
    
    else:
        state_df = state_df.fillna(method=fill_method)
        
        
        

     ## Renamed columns with state name
    if rename_cols == True:
    
        ## Get and Rename Sum Cols 
        orig_cols = state_df.columns

        for col in orig_cols:

            if state_first==True:
                new_col_name = f"{state_name}{name_sep}{col}"
            else:
                new_col_name = f"{col}{name_sep}{state_name}"

            state_df[new_col_name] = state_df[col].copy()

        ## Drop original cols
        state_df.drop(orig_cols,axis=1,inplace=True)
    
    
    
    ## Return only columns containing ts_cols
    if ts_col is not None:

        if isinstance(ts_col,str):
            ts_col = [ts_col]

            # find cols that end with the column name
        selected_cols=[]
        for col in ts_col:
            selected_cols.extend([c for c in state_df.columns if c.endswith(col)])
            
        state_df = state_df[selected_cols]
        
    ## Add outlier removal here:
        
        
        
    ## Visualize post-resampling 
    if diagnose:
            pfig = px.line(state_df,title="post-Resampling")
            pfig.show()
    
    
    return state_df





def calc_perc_change(ts,periods=1,replace_inf_with_nan=True):
    """Calculated the pct_change with pandas and optionall replaces inf values"""
    ts_pct_change = ts.pct_change(periods=periods)#
    
    if replace_inf_with_nan:
        ts_pct_change = ts_pct_change.replace({np.inf:np.nan,
                                      -np.inf:np.nan})

    return ts_pct_change


def remove_outliers_ts_series(ts,threshold_type='pct_change',raw_thresh=100,
                       pct_thresh=50, n_diff=1):
    """ Remove outliers from time series.
    
    Parameters:
        - ts 
        - threshold_type {'raw',('pct_change','%')}
        - raw_thresh 
        - pct_thresh
        - n_diff (1) - period for .diff or .pct_change
    """
    
    ## use threshold techniques to identify outleirs
    if threshold_type == 'raw':
        ## saving deltas 
        deltas  = ts.diff(n_diff)
        idx_outliers = deltas.abs()>raw_thresh

    elif (threshold_type == '%') | (threshold_type == 'pct_change'):
        deltas = calc_perc_change(ts,periods=n_diff)
        idx_outliers = deltas.abs() > pct_thresh
        
    else:
        raise Exception("Other threshold_kinds are not yet implemented.")
        
        
    ## SSaving outleirs
    outliers = deltas[idx_outliers]


    ## Filling in outliers
    ts_out = ts.copy()
    ts_out.loc[outliers.index] = np.nan
    ts_out = ts_out.interpolate()
    
    return ts_out
     
     

# def remove_outliers_ts(ts_,threshold_type='pct_change',raw_thresh=100,
#                        pct_thresh=50, n_diff=1):
#     """ Remove outliers from time series.
    
#     Parameters:
#         - ts_ (Series of DataFrame)
#         - threshold_type {'raw',('pct_change','%')}
#         - raw_thresh 
#         - pct_thresh
#         - n_diff (1) - period for .diff or .pct_change
#     """
    
#     if isinstance(ts_, pd.Series):
#         ts_df = ts_.to_frame(ts_.name)
#     else:
#         ts_df = ts_.copy()
    
#     ## sve copy to remove outleirs from
#     ts_out = ts_df.copy()
    
#     for col in ts_df.columns:
#         ts = ts_df[col].copy()
        
#         ## use threshold techniques to identify outleirs
#         if threshold_type == 'raw':
#             ## saving deltas 
#             deltas  = ts.diff(n_diff)
#             idx_outliers = deltas.abs()>raw_thresh

#         elif (threshold_type == '%') | (threshold_type == 'pct_change'):
#             deltas = calc_perc_change(ts,periods=n_diff)
#             idx_outliers = deltas.abs() > pct_thresh

#         else:
#             raise Exception("Other threshold_kinds are not yet implemented.")


#         ## SSaving outleirs
#         outliers = deltas[idx_outliers]


#         ## Filling in outliers
# #         ts_out = ts.copy()
#         ts_out[col].loc[outliers.index] = np.nan
#         ts_out[col] = ts_out[col].interpolate()
    
#     return ts_out

def remove_outliers_ts(ts_,threshold_type='pct_change',raw_thresh=100,
                       pct_thresh=50, n_diff=1,fill_method='interpolate'):
    """ Remove outliers from time series.
    
    Parameters:
        - ts 
        - threshold_type {'raw',('pct_change','%'),('zero','0')}
        - raw_thresh 
        - pct_thresh
        - n_diff (1) - period for .diff or .pct_change
    """
    
    if isinstance(ts_, pd.Series):
        ts_df = ts_.to_frame(ts_.name)
    else:
        ts_df = ts_.copy()
    
    ## sve copy to remove outleirs from
    ts_out = ts_df.copy()
    
    for col in ts_df.columns:
        ts = ts_df[col].copy()
        
        ## use threshold techniques to identify outleirs
        if threshold_type == 'raw':
            ## saving deltas 
            deltas  = ts.diff(n_diff)
            idx_outliers = deltas.abs()>raw_thresh

        elif (threshold_type == '%') | (threshold_type == 'pct_change'):
            deltas = calc_perc_change(ts,periods=n_diff)
            idx_outliers = deltas.abs() > pct_thresh


        elif (threshold_type == 'zero') | (threshold_type == '0'): 
            deltas = ts.copy()
            idx_outliers = ts == 0
        else:
            raise Exception("Other threshold_kinds are not yet implemented.")


        ## SSaving outleirs
        outliers = deltas[idx_outliers]


        ## Filling in outliers
#         ts_out = ts.copy()
        ts_out[col].loc[outliers.index] = np.nan
    
    
    

        ## Deal with null values:
        if fill_method == 'interpolate':
            ts_out[col] = ts_out[col].interpolate()

        elif fill_method == None:
            pass

        else:
            ts_out[col] = ts_out[col].fillna(method=fill_method)

    
    return ts_out
     
    

    

    








############
def plot_states(df, state_list, plot_cols = ['Confirmed'],df_only=False,
                new_only=False,plot_scatter=True,show=False):
    """Plots the plot_cols for every state in state_list.
    Returns plotly figure
    New as of 06/21"""
    import pandas as pd 
    import numpy as np
    ## Get state dataframes
    concat_dfs = []  
    STATES = {}
    
    ## Get each state
    for state in state_list:

        # Grab each state's df and save to STATES
        dfs = get_state_ts(df,state)
        STATES[state] = dfs

        ## for each plot_cols, find all columns that contain that col name
        for plot_col in plot_cols:
            concat_dfs.append(dfs[[col for col in dfs.columns if col.endswith(plot_col)]])#plot_col in col]])

    ## Concatenate final dfs
    plot_df = pd.concat(concat_dfs,axis=1)#[STATES[s] for s in plot_states],axis=1).iplot()
    
    
    ## Set title and df if new_only
    if new_only:
        plot_df = plot_df.diff()
        title = "Coronavirus Cases by State - New Cases"
    else:
        title = 'Coronavirus Cases by State - Cumulative'
    
    ## Reset Indes
    plot_df.reset_index(inplace=True)
    
    
    ## Return Df or plot
    if df_only==False:

        if np.any(['per capita' in x.lower() for x in plot_cols]):
            value_name = "# of Cases - Per Capita"
        else:
            value_name='# of Cases'


        pfig_df_melt = plot_df.melt(id_vars=['Date'],var_name='State',
                                    value_name=value_name)
        
        if plot_scatter:
            plot_func = px.scatter
        else:
            plot_func = px.line
            
            
        # Plot concatenated dfs
        pfig = plot_func(pfig_df_melt,x='Date',y=value_name,color='State',
                      title=title,template='plotly_dark',width=1000,height=700)        
#         pfig.update_xaxes(rangeslider_visible=True)

                # Add range slider
        pfig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7,
                             label="1week",
                             step="day",
                             stepmode="backward"),
                        dict(count=14,
                             label="2weeks",
                             step="day",
                             stepmode="backward"),
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),

                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        
        if show:
            pfig.show()
                
        return pfig
    
    else:
        return plot_df#.reset_index()



def get_group_ts(df,group_name,group_col='state',
                     ts_col=None, freq='D', agg_func='sum'):
        """Take df_us and extracts state's data as then Freq/Aggregation provided"""
        from IPython.display import display
        try:
            ## Get state_df group
            group_df = df.groupby(group_col).get_group(group_name).copy()
            
        except Exception:
            print("[!] ERROR!")
#             display(df.head())
            return None
        
        ## Resample and aggregate state data
        group_df = set_freq_resample(group_df,freq=freq,agg_func=agg_func)
#         group_df = group_df.resample(freq).agg(agg_func)


        ## Get and Rename Sum Cols 
        orig_cols = group_df.columns

        ## Create Renamed Sum columns
        for col in orig_cols:
            # Group - Column 
            group_df[f"{group_name} - {col}"] = group_df[col]

        ## Drop original cols
        group_df.drop(orig_cols,axis=1,inplace=True)

        ## Return on columns containing ts_cols
        if ts_col is not None:

            if isinstance(ts_col,str):
                ts_col = [ts_col]

            
            ts_cols_selected=[]
            for column in ts_col:
                ts_cols_selected.extend([col for col in group_df.columns if column in col])

            group_df = group_df[ts_cols_selected]

        return group_df 
    
    
    
def plot_group_ts(df, group_list,group_col, plot_cols = ['Confirmed'],
                df_only=False,
                new_only=False,plot_scatter=True,show=False,
                 width=1000,height=700):
    """Plots all columns conatining the words in plot_cols for every group in group_list.
    Returns plotly figure
    New as of 06/21"""
    import pandas as pd 
    import numpy as np
    
    ## Get state dataframes
    concat_dfs = []  
    GROUPS = {}
    
    ## Get each state
    for group in group_list:

        # Grab each state's df and save to STATES
        dfs = get_group_ts(df,group,group_col)
        GROUPS[group] = dfs

        ## for each plot_cols, find all columns that contain that col name
        for plot_col in plot_cols:
            concat_dfs.append(dfs[[col for col in dfs.columns if col.endswith(plot_col)]])

    ## Concatenate final dfs
    plot_df = pd.concat(concat_dfs,axis=1)
    
    
    ## Set title and df if new_only
    if new_only:
        plot_df = plot_df.diff()
        title = f"New Coronavirus Cases by {group_col}"
    else:
        title = f'Cumulative Coronavirus Cases by {group_col}'
    
    ## Reset Indes
    plot_df.reset_index(inplace=True)
  
    ## Return Df or plot
    if df_only:
         return plot_df#.reset_index()
    
    else:
        ## If any columns are per capita, change titleß
        if np.any(['per capita' in x.lower() for x in plot_cols]):
            value_name = "# of Cases - Per Capita"
        else:
            value_name='# of Cases'
            
            
        ## Melt Data for plotting
        pfig_df_melt = plot_df.melt(id_vars=['Date'],var_name='Group',
                                    value_name=value_name)
        
        ## Set plotting function
        if plot_scatter:
            plot_func = px.scatter
        else:
            plot_func = px.line
    
        # Plot concatenated dfs
        pfig = plot_func(pfig_df_melt,x='Date',y=value_name,color='Group',
                      title=title,template='plotly_dark',width=width,height=height)     
        
        ## Add range slider
        pfig.update_xaxes(rangeslider_visible=True)
        
        ## Display?
        if show:
            pfig.show()
                
        return pfig
    