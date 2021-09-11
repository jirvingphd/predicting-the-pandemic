# try:
#     from jupyter_dash import JupyterDash
# except:
#     import os
#     os.system("conda install -c conda-forge -c plotly jupyter-dash")
#     from jupyter_dash import JupyterDash

## PLOTLY IMPORTS/PARAMS
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

## Acitvating Cufflinks
import cufflinks as cf
cf.go_offline()
cf.set_config_file(sharing='public',theme='solar',offline=True)

## Importing Dash
import dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import dcc
from dash import html

from dash.dependencies import Input, Output


## Load Functions and Data
from functions import CoronaData,plot_states,get_state_ts,plot_group_ts

## LOAD DATA AND SAVE DFs
corona_data = CoronaData(verbose=False,run_workflow=True)
df = corona_data.df_us.copy()
df_world = corona_data.df.copy()


## Map of total cases by state
max_corona = df.groupby('state').max().reset_index()
color_column = 'Confirmed'
map = px.choropleth(max_corona,color=color_column,locations='state',
              hover_data=['Confirmed','Deaths','Recovered'], 
              hover_name='state',
              locationmode="USA-states", scope='usa',
              title=f"Total {color_column} Cases by State",
              color_continuous_scale=px.colors.sequential.Reds)



def make_options(menu_choices):
    """Returns list of dictionary with {'label':menu_choice,'value':menu_choice}"""
    options = []
    for choice in menu_choices:
        options.append({'label':choice,'value':choice})
    return options

## Make Plot Cols list for options
stat_cols = ['Confirmed','Deaths','Recovered']
plot_cols = []
for column in stat_cols:
    plot_cols.extend([col for col in df.columns if column in col])
    
## Columns for the world
plot_cols_world=[]
for column in stat_cols:
    plot_cols.extend([col for col in df_world.columns if column in col])

## Make Case-Type Options
new_options = [{'label':'New Cases Only','value':1},
{'label':'Cumulative Cases','value':0}]



# Build App
# app = JupyterDash()
app = dash.Dash(__name__)
server = app.server




app.layout = html.Div(id='outerbox',children=[
    html.H1("Coronavirus Cases - By State"),
        dcc.Graph(id='map',figure=map),         
        
        ## State App
        html.Div(id='app',
        children=[
            html.Div(id="menu",
                    children=[
                        html.H2("Select Case Types and States"),

                        html.Div(id='case_type_menu', className='case_menu_class',
                                children=[
                                    dcc.RadioItems(id='choose_new',className='case_menu_class',
                                                    options=new_options,
                                                    value=0),
                                    dcc.Dropdown(id='choose_cases',multi=False,className='case_menu_class',
                                                placeholder='Select Case Type', 
                                                options=make_options(plot_cols),
                                                value='Confirmed'),#]),
                        dcc.Dropdown(id='choose_states',className='case_menu_class',
                                    multi=True,
                                    placeholder='Select States', 
                                    options= make_options(df['state'].sort_values().unique( )),
                                    value=['MD','NY','TX','CA','AZ'])])
                    ]),
            dcc.Graph(id='graph')
        ]),
        html.H1("Total Coronavirus Cases - By Country",id='world-section'),

        html.Div(id='app-world',
        children=[
            html.Div(id="menu-world",className='case_menu_class',
                    children=[
                        html.H2("Select Case Types and Countries"),

                        html.Div(id='case_type_menu-world', className='case_menu_class',
                                children=[
                                    dcc.RadioItems(id='choose_new-world',className='case_menu_class',
                                                    options=new_options,
                                                    value=1),
                                    dcc.Dropdown(id='choose_cases-world',className='case_menu_class',
                                                 multi=False,
                                                placeholder='Select Case Type', 
                                                options=make_options(plot_cols_world),
                                                value='Confirmed'),#]),
                        dcc.Dropdown(id='choose_countries',className='case_menu_class',
                                    multi=True,
                                    placeholder='Select Countries', 
                                    options= make_options(df_world['Country/Region'].sort_values().unique( )),
                                    value=['US','Italy','France','Canada','Mainland China'])])
                    ]),
            dcc.Graph(id='graph-world')
        ])
        
        ])


@app.callback(Output('graph','figure'),[Input('choose_states','value'),
                                       Input('choose_cases','value'),
                                       Input('choose_new','value')])
def update_output_div(states,cases,new_only):
    if isinstance(states,list)==False:
        states = [states]
    if isinstance(cases,list)==False:
        cases = [cases]

    pfig = plot_states(df,states,plot_cols=cases,new_only=new_only)
    return pfig


## world callback
@app.callback(Output('graph-world','figure'),[Input('choose_countries','value'),
                                       Input('choose_cases-world','value'),
                                       Input('choose_new-world','value')])
def update_output_div(countries,cases,new_only):
    if isinstance(countries,list)==False:
        countries = [countries]
    if isinstance(cases,list)==False:
        cases = [cases]
        
    pfig=plot_group_ts(df_world,group_list=countries,plot_cols=cases,
                       group_col='Country/Region',
                     new_only=new_only,plot_scatter=True,width=900,height=600)

    # pfig = plot_states(df,countries,plot_cols=cases,new_only=new_only)
    return pfig


FLASK_APP=app

# if __name__=='main':
# app.run_server(debug=True)
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.server.run(port=8000,debug=True, host='127.0.0.1')
    
