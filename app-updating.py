import streamlit as st

from fsds.imports import *
import pandas as pd

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

st.markdown('- This dashboard uses data from several APIs and kaggle datasets. To fetch the lateast data, click the button below.')
WORKFLOW_BUTTON = st.button("Fetch new data.",)

st.markdown('> Note: it can take up to 2 minutes to download the data.')
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

df, STATES = load_data(WORKFLOW_BUTTON)

options_stats= df.drop(['Deaths','Cases'],axis=1).columns.tolist()

st.markdown("___")

st.markdown("## Overview - Comparing All States")
## plot state map
n_days = st.slider("PAST N # OF DAYS",value=30,min_value=7,max_value=180)
col = st.selectbox("Which statistic to map?", options_stats)

today = dt.date.today()
end_state = today
start_date = pd.Timestamp(today) - pd.Timedelta(f'{str(n_days)} days')

map = fn.app_functions.plot_map_columns(df,col=col, last_n_days=n_days,
plot_map=False,return_map=True)

df_rank= fn.app_functions.plot_map_columns(df,col=col, last_n_days=n_days,
plot_map=False,return_map=False)

st.plotly_chart(map)

st.markdown("___")


### Plot same stat for different states
st.markdown('## Comparing Selected States')
stat_to_compare = st.multiselect("Which statistic to compare?",options_stats,
default=["Cases-New"])
states_to_compare = st.multiselect("Which states to compare?",list(STATES.keys()),
default=["NY",'MD','FL','CA','TX'])

plot_df = fn.app_functions.get_states_to_plot(df,state_list=states_to_compare,
            plot_cols=stat_to_compare,
                            agg_func= 'mean',
                  rename_cols=True,fill_method='interpolate',
                  remove_outliers=False, state_first=True,
                  threshold_type=['0','%'], diagnose=False)
st.plotly_chart(px.line(plot_df))


st.markdown("___")

# ############################## PRIOR TO  09/21 ###########################
st.markdown('## Timeseries Forecasting by State/Statistic ')

state_name = st.selectbox('Select State', list(STATES.keys()))
col = st.selectbox("Select statistic",options_stats)
start_date = st.date_input('Start Date for Training Data',
 value=pd.to_datetime('06-2020'))


df_state = STATES[state_name].loc[start_date:].copy()

# # col = 'Cases-New'
ts = df_state[col].copy()
ax = ts.plot(title=f"{state_name}-{col}");
ax.set_ylabel(col)


model_q = st.button('Run model?', 
on_click= fn.modeling.make_timeseries_model,args=(STATES,state_name,col))


st.pyplot(ax.get_figure())# plt.show()





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