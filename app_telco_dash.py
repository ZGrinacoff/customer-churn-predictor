# Import libraries.
from __future__ import division
from datetime import datetime, timedelta, date
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from functions import mean_group_df, count_group_df

import dash_auth

import logging
import boto3
from botocore.exceptions import ClientError

USERNAME_PASSWORD_PAIRS = [
    ['ZGrinacoff', 'Rangers123!']
]


# external_stylesheets = [
#     'https://codepen.io/chriddyp/pen/bWLwgP.css',
#     {
#         'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
#         'rel': 'stylesheet',
#         'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
#         'crossorigin': 'anonymous'
#     }
# ]

app = dash.Dash()
auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)
server = app.server

s3 = boto3.client('s3')
response = s3.list_buckets()
bucketName='telco-customers'
s3.download_file(bucketName, 'data/final_telco_df.csv', 'data/final_telco_df.csv')
final_telco_df = pd.read_csv('data/final_telco_df.csv', encoding = 'unicode_escape')

s3.download_file(bucketName, 'data/future_retention_df.csv', 'data/future_retention_df.csv')
future_retention_df = pd.read_csv('data/future_retention_df.csv', encoding = 'unicode_escape')

telco_features = final_telco_df.columns

cat_features = telco_features.drop(['customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TenureCluster', 'MonthlyChargesCluster', 'TotalChargesCluster', 'proba'])

num_features = telco_features.drop(['customerID', 'Churn', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'TenureCluster', 'MonthlyChargesCluster', 'TotalChargesCluster', 'proba'])

cluster_features = telco_features.drop(['customerID', 'Churn', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges', 'proba'])

retention_count = future_retention_df.groupby('ProbabilityCluster').customerID.count().reset_index()

app.layout = html.Div([
    html.Div([
        html.H1('Telco Customer Churn Dashboard: Which factors have the greatest impact on Churn and Retention?',style={'color':'blue', 'border':'2px blue solid', 'borderRadius':5,
        'padding':10, 'width':500}),
            html.H3('Our historical data shows that out of 7,043 total customers, 1,869 have churned. With an actual churn rate of 26.54%, it is important for us to understand which features most affect our potential for retaining existing and future customers. '),
        html.Div([
            html.Label('Categorical Feature Selection'),
            dcc.Dropdown(
                id='cat_xaxis',
                options=[{'label': i.title(), 'value': i} for i in cat_features],
                value='InternetService'
            ),
            html.Label('Numerical Feature Selection'),
            dcc.Dropdown(
                id='num_xaxis',
                options=[{'label': i.title(), 'value': i} for i in num_features],
                value='tenure'
            ),
            html.Label('Numerical Cluster Feature Selection'),
            dcc.Dropdown(
                id='cluster_xaxis',
                options=[{'label': i.title(), 'value': i} for i in cluster_features],
                value='TenureCluster'
            )
        ], style={'width': '15%', 'display': 'inline-block', 'padding': 10}),
        html.Div([
        dcc.Graph(id='cat-graphic', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='num-graphic', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='cluster-graphic', style={'width': '33%', 'display': 'inline-block'})
                ]),
    ]),
    html.Div([
        dcc.Graph(id='probacluster-graphic',
        figure={
            'data': [
                {'x': ['Low (<1 - 9.92%)', 'Mid-Low (9.95 - 25.90%)', 'Mid-High (25.93 - 48.10%)', 'High (48.17 - 86.97%)'],
                'y': [2672, 1302, 749, 440],
                'type': 'bar',
                'hovertemplate': "<b>Churn Probability Cluster: %{x}<br>Customer Count: %{y}<br><br>",
                'name': ""}
            ],
            'layout': {
                    'margin': {'l':'50', 'r':'50', 't':'50', 'b':'50'},
                    'yaxes': {'tickfont': '6'},
                    'xaxis': {'title': 'Churn Probability Cluster', 'titlefont': {'size': '15'}},
                    'yaxis': {'title': 'Customer Count', 'titlefont': {'size': '15'}},
                    # 'text': {'x'},
                    'title': 'Customer Count by Churn Probability Cluster',
                    'colorway': ['green', 'slategray', 'purple', 'blue']
                }
        }, style={'width': '50%', 'display': 'inline-block'}
        ),
        dcc.Graph(id='probpie-graphic',
        figure={
            'data': [
            {'labels': ['Low (<1 - 9.92%)', 'Mid-Low (9.95 - 25.90%)', 'Mid-High (25.93 - 48.10%)', 'High (48.17 - 86.97%)'],
            'values': [2672, 1302, 749, 440],
            'type': 'pie',
            'hole': '.3',
            'marker': {'colors':['slategray', 'purple', 'blue', 'green']}}
            ],
            'layout': [
                {'title_text': 'Percent Share by Probability Cluster'}
            ]
        }, style={'width': '50%', 'display': 'inline-block'}
        )
    ]),
    html.Div([
            dcc.Graph(id='feature-graphic',
            figure={
                'data': [
                    {'x': [1, 1, 2, 4, 5, 9, 10, 13, 14, 15, 17, 18, 18, 18, 18, 18, 19, 22, 30, 32, 37, 40, 42, 44, 46, 52, 53, 58, 75, 83, 109, 313, 540, 579],
                    'y': ['OnlineSecurity_Yes', 'TechSupport_Yes', 'TenureCluster_Mid', 'MonthlyChargesCluster_Mid', 'OnlineBackup_Yes', 'TotalChargesCluster_Mid',
                    'MultipleLines_Yes', 'InternetService_DSL', 'StreamingTV_Yes', 'InternetService_Fiber_optic', 'DeviceProtection_No',
                    'StreamingTV_No', 'StreamingMovies_Yes', 'StreamingMovies_No', 'PaymentMethod_Mailed_check', 'PaymentMethod_Bank_transfer_automatic_',
                    'Partner', 'PhoneService', 'Contract_Two_year', 'MultipleLines_No', 'PaymentMethod_Credit_card_automatic_', 'Dependents',
                    'Contract_Month_to_month', 'Contract_One_Year', 'TechSupport_No', 'PaperlessBilling', 'OnlineBackup_No', 'OnlineSecurity_No',
                    'SeniorCitizen', 'gender', 'PaymentMethod_Electronic_check', 'tenure', 'MonthlyCharges','TotalCharges'],
                    'type': 'bar', 'orientation': 'h',
                    'marker': 
                    {'color': ['slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green',
                  'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 
                  'slategray', 'purple', 'green','slategray', 'purple', 'green', 'slategray']}
                }
                ],
                'layout': {
                    'margin': {'l':'250', 'r':'50', 't':'50', 'b':'50'},
                    'yaxes': {'tickfont': '6'},
                    'xaxis': {'title': 'F-Score', 'titlefont': {'size': '15'}},
                    'yaxis': {'title': 'Feature', 'titlefont': {'size': '15'}},
                    # 'text': {'x'},
                    'title': 'Feature Importance Based on XGBoost Model'
                }
            }
            )
    ])
        ])

@app.callback(
    Output('cat-graphic', 'figure'),
    [Input('cat_xaxis', 'value')])

def update_cat_graph(cat_xaxis_name):
    cat_plot_df=mean_group_df(final_telco_df, cat_xaxis_name)
    hovertemplate = "<b>Feature Type: %{x}<br>Average Churn Rate: %{y}<br><br>"
    return {
        'data': [go.Bar(
            x=cat_plot_df[cat_xaxis_name],
            y=cat_plot_df['Churn'],
            hovertemplate=hovertemplate,            
            name=""
        )],
        'layout': go.Layout(
            xaxis={'title': cat_xaxis_name.title()},
            yaxis={'title': 'Churn'},
            title='Categorical Feature vs. Average Churn',
            margin={'l': 80, 'b': 40, 't': 40, 'r': 80},
            hovermode='closest',
            colorway= ['green']
        )
    }

@app.callback(
    Output('num-graphic', 'figure'),
    [Input('num_xaxis', 'value')])

def update_num_graph(num_xaxis_name):
    num_plot_df=mean_group_df(final_telco_df, num_xaxis_name)
    hovertemplate = "<b>Feature Type: %{x}<br>Average Churn Rate: %{y}<br><br>"
    return {
        'data': [go.Scatter(
            x=num_plot_df[num_xaxis_name],
            y=num_plot_df['Churn'],
            mode='markers',
            hovertemplate=hovertemplate,            
            name=""
        )],
        'layout': go.Layout(
            xaxis={'title': num_xaxis_name.title()},
            yaxis={'title': 'Churn'},
            title='Numerical Feature vs Average Churn',
            margin={'l': 80, 'b': 40, 't': 40, 'r': 80},
            hovermode='closest',
            colorway= ['blue']
        )
    }

@app.callback(
    Output('cluster-graphic', 'figure'),
    [Input('cluster_xaxis', 'value')])

def update_cluster_graph(cluster_xaxis_name):
    cluster_plot_df=mean_group_df(final_telco_df, cluster_xaxis_name)
    hovertemplate = "<b>Feature Type: %{x}<br>Average Churn Rate: %{y}<br><br>"
    return {
        'data': [go.Bar(
            x=cluster_plot_df[cluster_xaxis_name],
            y=cluster_plot_df['Churn'],
            hovertemplate=hovertemplate,            
            name=""
        )],
        'layout': go.Layout(
            xaxis={'title': cluster_xaxis_name.title(), "type": "category", "categoryarray": ['Low', 'Mid', 'High']},
            yaxis={'title': 'Churn'},
            title='Numerical Cluster vs Average Churn',
            margin={'l': 80, 'b': 40, 't': 40, 'r': 80},
            hovermode='closest',
            colorway= ['purple']
        )
    }

if __name__ == '__main__':
    app.run_server()