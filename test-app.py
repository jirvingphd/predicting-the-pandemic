import streamlit as st

# from fsds.imports import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os,glob,sys,joblib,zipfile,json
import datetime as dt
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

plt.rcParams['figure.figsize'] = (12,6)
pd.set_option('display.max_columns',0)
# fs.check_package_versions(['statsmodels'],fpath=True)



### FORECAST SPECIFIC FUNCTIONS
import statsmodels.api as sms
import statsmodels.tsa.api as tsa
from pmdarima import auto_arima
import project_functions as fn
# from fsds import pandemic as fn
import os,json,glob

with open("FILE_DIRECTORY.json") as f:
    FPATHS = json.load(f)


### TItle
st.markdown('# Planning for the Pandemic')
st.markdown("""
- James M. Irving, PhD.
    - james.irving.phd@gmail.com
    - [LinkedIn](https://www.linkedin.com/in/james-irving-4246b571/)
    - [GitHub Repo](https://github.com/jirvingphd/predicting-the-pandemic)

___""")

st.markdown("""## ***Goal***
- Covid-19 and the various strains that have since emerged has upended modern life and fundamentally changed how we function as a society.
- Part of what has made it difficult to tackle the pandemic is the differences between states, state laws/policies, and a lack of public understanding about the predictability of the surges in cases. 
- The goal of this dashboard is to find the provide easy access state-level coronavirus and hospital capacity statistics.
    - Furthermore, I wanted to provide on-demand timeseries forecasts into the near future for all/any of these statistics.
""")



st.markdown('## ***The Data***')
st.markdown('- This dashboard uses data from several APIs and kaggle datasets. To fetch the lateast data, click the button below.')
WORKFLOW_BUTTON = st.button("Fetch new data.",)

st.markdown('> Note: it can take up to 2 minutes to download the data.')

st.markdown("""### Sources
- Coronavirus Data by State- # of Cases/Deaths by State
    - [Kaggle Dataset: "COVID-19 data from John Hopkins University"](https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university) 
    - Repackaged version of the data from the [official Johns Hopkins Repository](https://github.com/CSSEGISandData/COVID-19)
- Hospital Hospital & ICU Occupancy Data:
    - [HealthData.gob Api: "COVID-19 Reported Patient Impact and Hospital Capacity by State Timeseries API"](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh)
""")
# RUN_FULL_WORKFLOW=False


def load_data(WORKFLOW_BUTTON=False):
    if WORKFLOW_BUTTON == True:
        df_states,STATES = fn.data_acquisition.FULL_WORKFLOW(merge_hospital_data=True)
        ## renaming since merge_hofspital_data=True
    #     DF = df_states.copy()
    #     print(STATES.keys())    
        
    else:
        # print(f"[i] Using previously downloaded data...")
        # df_states = pd.read_pickle(FPATHS['fpath_final_df_pickle'])
        df_states =  pd.read_csv(FPATHS['fpath_final_df_csv'],compression='gzip',parse_dates=['Date'],
                    index_col=[0,1])

    #     with open(FPATHS['fpath_final_states']) as f:
        STATES = joblib.load(FPATHS['fpath_final_states'])
    return df_states,STATES

## load data and save options
df, STATES = load_data(WORKFLOW_BUTTON)
options_stats= df.drop(['Deaths','Cases'],axis=1).columns.tolist()



st.markdown("___")
st.markdown("## ***Overview - Comparing All States***")

########## MAP CODE ##########
# calc dates for map
# today = dt.date.today()
# end_state = today
# start_date = pd.Timestamp(today) - pd.Timedelta(f'{str(n_days)} days')
latest_date = df.droplevel(0).index.max()
end_date = latest_date

## plot state map
n_days = st.slider(f"PAST N # OF DAYS BEFORE {latest_date.strftime('%m/%d/%Y')}",value=30,min_value=7,max_value=180)
col = st.selectbox("Which statistic to map?", options_stats)

start_date = pd.Timestamp(latest_date) - pd.Timedelta(f'{str(n_days)} days')


## get map
map = fn.app_functions.plot_map_columns(df,col=col, last_n_days=n_days,
plot_map=False,return_map=True)

# get just df
df_rank= fn.app_functions.plot_map_columns(df,col=col, last_n_days=n_days,
plot_map=False,return_map=False)

# show map
st.plotly_chart(map)


########## COMPARISON LINEPLOT CODE ##########

## Download pop data
pop_df = fn.data_acquisition.get_state_pop_data()#get_state_pop_data()
pop_df = pop_df.set_index('abbr')

st.markdown("___")
st.markdown('## ***Comparing Selected States***')


## select states and stats
stat_to_compare = st.multiselect("Which statistic to compare?",options_stats,
default=["Cases-New"])
states_to_compare = st.multiselect("Which states to compare?",list(STATES.keys()),
default=["NY",'MD','FL','CA','TX'])


## Adding rolling average
plot_rolling =  st.checkbox('Plot 7 day rolling average instead of daily data.')

## Pop-normalized data
per_capita =  st.checkbox('Plot population-adjusted metrics.')


## get and show plot
plot_df = fn.app_functions.get_states_to_plot(df,state_list=states_to_compare,
            plot_cols=stat_to_compare,
                            agg_func= 'mean',
                  rename_cols=True,fill_method='interpolate',
                  remove_outliers=False, state_first=True,
                  threshold_type=['0','%'], diagnose=False)
                  
if per_capita:
    ## get the corresponding  population estimate
    orig_cols = plot_df.columns
    for state in states_to_compare:
        state_cols = [c for c in orig_cols if state in c]
        
        for col in state_cols:
            pop_adj = (plot_df[col] / pop_df.loc[state, "POP_2021"]) * 100_000
            
            plot_df[f"{col} (per 100K)"] = pop_adj
    plot_df = plot_df.drop(columns=orig_cols)
    
    

                  

                  
if plot_rolling==True:
    plot_df = plot_df.rolling(7).mean()
    title='7-Day Rolling Average'
else:
    title='Raw Daily Data'
    
st.plotly_chart(px.line(plot_df,title=title))


st.markdown("___")

# ############################## PRIOR TO  09/21 ###########################
st.markdown('## ***Timeseries Forecasting by State/Statistic***')


default_model_start = latest_date - pd.to_timedelta('365 days')
state_name = st.selectbox('Select State', list(STATES.keys()))
col = st.selectbox("Select statistic",options_stats)
start_date = st.date_input('Start Date for Training Data',
 value=default_model_start)#pd.to_datetime('06-2020'))


df_state = STATES[state_name].loc[start_date:].copy()

# # col = 'Cases-New'
ts = df_state[col].copy()
ax = ts.plot(title=f"{state_name}-{col}");
ax.set_ylabel(col)



st.pyplot(ax.get_figure())# plt.show()


st.markdown("""> **Click "`Run model`" below to start the modeling process for the selected state and statistic.**
-  [!] Warning: the gridsearch process may take several minutes. Try selecting a more recent start date to increase performance.""")


model_q = st.button('Run model.', 
on_click= fn.modeling.make_timeseries_model,args=(STATES,state_name,col))



# st.button('Hit me')
# st.checkbox('Check me out')
# st.radio('Radio', [1,2,3])
# st.multiselect('Multiselect', [1,2,3])
# st.slider('Slide me', min_value=0, max_value=10)
# st.select_slider('Slide to select', options=[1,'2'])
# st.text_input('Enter some text')
# st.number_input('Enter a number')
# st.text_area('Area for textual entry')
# st.date_input('Date input')
# st.time_input('Time entry')
# st.file_uploader('File uploader')
# st.color_picker('Pick a color')