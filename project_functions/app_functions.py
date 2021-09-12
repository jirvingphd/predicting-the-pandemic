## to replace af.get_state_ts

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# pio.templates.default = "plotly_dark"
import pandas as pd

import numpy as np

### BEST VERSION YET! 09/11 ######
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
     
     

def remove_outliers_ts(ts_,threshold_type='pct_change',raw_thresh=100,
                       pct_thresh=50, n_diff=1):
    """ Remove outliers from time series.
    
    Parameters:
        - ts_ (Series of DataFrame)
        - threshold_type {'raw',('pct_change','%')}
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

        else:
            raise Exception("Other threshold_kinds are not yet implemented.")


        ## SSaving outleirs
        outliers = deltas[idx_outliers]


        ## Filling in outliers
#         ts_out = ts.copy()
        ts_out[col].loc[outliers.index] = np.nan
        ts_out[col] = ts_out[col].interpolate()
    
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
    