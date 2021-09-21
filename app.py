from zlib import compress
import streamlit as st

from fsds.imports import *
import pandas as pd

import os,glob,sys,joblib,zipfile,json
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

plt.rcParams['figure.figsize'] = (12,6)
pd.set_option('display.max_columns',0)
# fs.check_package_versions(['statsmodels'],fpath=True)

import joblib

### FORECAST SPECIFIC FUNCTIONS
import statsmodels.api as sms
import statsmodels.tsa.api as tsa
from pmdarima import auto_arima
import project_functions as fn
# from fsds import pandemic as fn
import os,json,glob

with open("FILE_DIRECTORY.json") as f:
    FPATHS = json.load(f)



# RUN_FULL_WORKFLOW=False
WORKFLOW_BUTTON = st.button("Fetch new data.",)


def load_data(WORKFLOW_BUTTON=False):
    if WORKFLOW_BUTTON == True:
        df_states,STATES = fn.data_acquisition.FULL_WORKFLOW(merge_hospital_data=True)
        ## renaming since merge_hofspital_data=True
    #     DF = df_states.copy()
    #     print(STATES.keys())    
        
    else:
        print(f"[i] Using previously downloaded data...")
        # df_states = pd.read_pickle(FPATHS['fpath_final_df_pickle'])
        df_states = pd.read_csv(FPATHS['fpath_final_df_csv'],compression='gzip')#joblib.load(FPATHS['final')
        
    #     with open(FPATHS['fpath_final_states']) as f:
        STATES = joblib.load(FPATHS['fpath_final_states'])
    return df_states,STATES

df, STATES = load_data(WORKFLOW_BUTTON)



st.write('Planning for the Pandemic')


state_name = st.sidebar.selectbox('Select State', list(STATES.keys()))
col = st.sidebar.selectbox("Select statistic",STATES["NY"].columns.to_list())
df_state = STATES[state_name].copy()

# col = 'Cases-New'
ts = df_state[col].copy()
ax = ts.plot(title=f"{state_name}-{col}");
ax.set_ylabel(col)
st.pyplot(ax.get_figure())# plt.show()


model_q = st.sidebar.button('Run model?', on_click= fn.modeling.make_timeseries_model,args=(STATES,state_name,col))




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