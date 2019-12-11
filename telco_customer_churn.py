# Import libraries.
from __future__ import division
from datetime import datetime, timedelta, date
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# Initiate visualization library for jupyter notebook.
pyoff.init_notebook_mode()

## Pushing Data to AWS S3

# Import boto3 to access s3 bucket.
# Retrieve the list of existing buckets.
import logging
import boto3
from botocore.exceptions import ClientError
s3 = boto3.client('s3')
response = s3.list_buckets()

print('Existing buckets:')
for bucket in response['Buckets']:
    print(f' {bucket["Name"]}')

# Function to upload a file.
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket
    
    :param file_name: File to upload
    :param bucket: Bucket to upload file to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False"""
    
    # If S3 object_name was not specified, use file_name.
    if object_name is None:
        object_name = file_name
        
    # Upload the file.
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

# Define bucket to upload files to.
bucketName='telco-customers'

# # Upload Churn file to S3 bucket.
# fileName='data/Telco-Customer-Churn.csv'
# upload_file(fileName, bucketName)

## Part 4: Churn Prediction

# Since we know our best customers by segmentation and lifetime value prediction, we should also work hard on retaining them. That’s what makes Retention Rate is one of the most critical metrics.

# Retention Rate is an indication of how good is your product market fit (PMF). If your PMF is not satisfactory, you should see your customers churning very soon. One of the powerful tools to improve Retention Rate (hence the PMF) is Churn Prediction. By using this technique, you can easily find out who is likely to churn in the given period.

# Steps to develop a churn prediction model:

# * Exploratory data analysis

# * Feature engineering

# * Investigating how the features affect Retention by using Logistic Regression

# * Building a classification model with XGBoost

### Exploratory Data Analysis:

# Download the OnlineRetail.csv file.
s3.download_file(bucketName, 'data/Telco-Customer-Churn.csv', 'data/Telco-Customer-Churn.csv')

# Read CSV into Pandas DF and display first 10 rows.
telco_df = pd.read_csv('data/Telco-Customer-Churn.csv', encoding = 'unicode_escape')
telco_df.head(10)

telco_df.info()

# The data falls under two categories:

# * Categorical Features: gender, streaming tv, payment method, etc.

# * Numerical Features: tenure, monthly charges, and total charges.

# Before ananlysing each of our features to identify how helpful they might be in determining customer churn, lets first convert encode Yes/No as an integer (Yes=1, No=0).

# Convert Churn Yes/No to binary integers.
telco_df.loc[telco_df.Churn=='No', 'Churn'] = 0
telco_df.loc[telco_df.Churn=='Yes', 'Churn'] = 1

telco_df.Churn.value_counts()

# Actual Churn Rate.
(telco_df.query("Churn == 1")['Churn'].count() / telco_df.Churn.count()) * 100

### Categorical Features:

#### Gender

gender_plot = telco_df.groupby('gender').Churn.mean().reset_index()
gender_plot

(float(gender_plot.query("gender=='Female'")['Churn']) -
 float(gender_plot.query("gender=='Male'")['Churn'])) * 100

gender_plot.Churn

# The difference between Customer Churn by gender is minimal, let's visualize.

plot_data_gender = [
    go.Bar(
        x=gender_plot['gender'],
        y=gender_plot['Churn'],
        width=[0.5, 0.5],
        text=gender_plot['Churn'],
        name="Gender",
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Gender</b>: %{x}<br>',
        marker= dict(color=['green', 'purple'])
    )
]

plot_layout_gender = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Gender',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_gender = go.Figure(data=plot_data_gender, layout=plot_layout_gender)
pyoff.iplot(fig_gender)

#### Partner:

# Whether the customer has a partner, or not.

partner_plot = telco_df.groupby('Partner').Churn.mean().reset_index()
partner_plot

(float(partner_plot.query("Partner=='No'")['Churn']) -
 float(partner_plot.query("Partner=='Yes'")['Churn'])) * 100

# There is 13% difference between customers without partners to those with partners that have churned.

plot_data_partner = [
    go.Bar(
        x=partner_plot['Partner'],
        y=partner_plot['Churn'],
        width=[0.5, 0.5],
        name='Partner',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Partner</b>: %{x}<br>',
        marker= dict(color=['green', 'purple'])
    )
]

plot_layout_partner = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Partner',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_partner = go.Figure(data=plot_data_partner, layout=plot_layout_partner)
pyoff.iplot(fig_partner)

#### Phone Service:

# Whether the customer has a phone service or not (Yes, No)

phoneService_plot = telco_df.groupby('PhoneService').Churn.mean().reset_index()
phoneService_plot

(float(phoneService_plot.query("PhoneService=='Yes'")['Churn']) -
 float(phoneService_plot.query("PhoneService=='No'")['Churn'])) * 100

# There is only a slight difference in churn between customers with or without phone service.

plot_data_phoneService = [
    go.Bar(
        x=phoneService_plot['PhoneService'],
        y=phoneService_plot['Churn'],
        width=[0.5, 0.5],
        name='Phone Service',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Phone Service</b>: %{x}<br>',
        marker= dict(color=['green', 'purple'])
    )
]

plot_layout_phoneService = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Phone Service',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_phoneService = go.Figure(data=plot_data_phoneService, layout=plot_layout_phoneService)
pyoff.iplot(fig_phoneService)

#### Multiple Lines

# Whether the customer has multiple lines or not (Yes, No, No phone service)

multipleLines_plot = telco_df.groupby('MultipleLines').Churn.mean().reset_index()
multipleLines_plot

plot_data_multipleLines = [
    go.Bar(
        x=multipleLines_plot['MultipleLines'],
        y=multipleLines_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Multiple Lines',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Multiple Lines</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_multipleLines = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Multiple Lines',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_multipleLines = go.Figure(data=plot_data_multipleLines, layout=plot_layout_multipleLines)
pyoff.iplot(fig_multipleLines)

#### Internet Service

# Customer’s internet service provider (DSL, Fiber optic, No)

internetService_plot = telco_df.groupby('InternetService').Churn.mean().reset_index()
internetService_plot

(float(internetService_plot.query("InternetService=='Fiber optic'")['Churn']) -
 float(internetService_plot.query("InternetService=='DSL'")['Churn'])) * 100

# There is a substantial difference between Churn rates for customers that used Fiber Optic as opposed to DSL services.

# This is quite surprising because we normally expect Fiber Optic customers to churn less due to the fact that they use a more premium service. But this can happen due to high prices, competition, customer service, and many other reasons.

plot_data_internetService = [
    go.Bar(
        x=internetService_plot['InternetService'],
        y=internetService_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Internet Service',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Internet Service</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_internetService = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Internet Service',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_internetService = go.Figure(data=plot_data_internetService, layout=plot_layout_internetService)
pyoff.iplot(fig_internetService)

#### Online Security

# Whether the customer has online security or not (Yes, No, No internet service)

onlineSecurity_plot = telco_df.groupby('OnlineSecurity').Churn.mean().reset_index()
onlineSecurity_plot

(float(onlineSecurity_plot.query("OnlineSecurity=='No'")['Churn']) -
 float(onlineSecurity_plot.query("OnlineSecurity=='Yes'")['Churn'])) * 100

# There is a substantial difference between Churn rates for customers that do and don't have Online Security. Customers without are roughly 27% more likely to churn.

plot_data_onlineSecurity = [
    go.Bar(
        x=onlineSecurity_plot['OnlineSecurity'],
        y=onlineSecurity_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Online Security',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Online Security</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_onlineSecurity = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Online Security',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_onlineSecurity = go.Figure(data=plot_data_onlineSecurity, layout=plot_layout_onlineSecurity)
pyoff.iplot(fig_onlineSecurity)

#### Online Backup

# Whether the customer has online backup or not (Yes, No, No internet service)

onlineBackup_plot = telco_df.groupby('OnlineBackup').Churn.mean().reset_index()
onlineBackup_plot

(float(onlineBackup_plot.query("OnlineBackup=='No'")['Churn']) -
 float(onlineBackup_plot.query("OnlineBackup=='Yes'")['Churn'])) * 100

# There is a  difference between Churn rates for customers that do and don't have Online Backup. Customers without are roughly 18% more likely to churn.

plot_data_onlineBackup = [
    go.Bar(
        x=onlineBackup_plot['OnlineBackup'],
        y=onlineBackup_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Online Backup',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Online Backup</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_onlineBackup = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Online Backup',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_onlineBackup = go.Figure(data=plot_data_onlineBackup, layout=plot_layout_onlineBackup)
pyoff.iplot(fig_onlineBackup)

#### Device Protection

# Whether the customer has device protection or not (Yes, No, No internet service)

deviceProtection_plot = telco_df.groupby('DeviceProtection').Churn.mean().reset_index()
deviceProtection_plot

(float(deviceProtection_plot.query("DeviceProtection=='No'")['Churn']) -
 float(deviceProtection_plot.query("DeviceProtection=='Yes'")['Churn'])) * 100

plot_data_deviceProtection = [
    go.Bar(
        x=deviceProtection_plot['DeviceProtection'],
        y=deviceProtection_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Device Protection',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Device Protection</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_deviceProtection = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Device Protection',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_deviceProtection = go.Figure(data=plot_data_deviceProtection, layout=plot_layout_deviceProtection)
pyoff.iplot(fig_deviceProtection)

#### Tech Support

# Whether the customer has tech support or not (Yes, No, No internet service)

techSupport_plot = telco_df.groupby('TechSupport').Churn.mean().reset_index()
techSupport_plot

(float(techSupport_plot.query("TechSupport=='No'")['Churn']) -
 float(techSupport_plot.query("TechSupport=='Yes'")['Churn'])) * 100

# Customers that don't use tech support are roughly 26% more likely to churn.

plot_data_techSupport = [
    go.Bar(
        x=techSupport_plot['TechSupport'],
        y=techSupport_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Tech Support',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Tech Support</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_techSupport = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Tech Support',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_techSupport = go.Figure(data=plot_data_techSupport, layout=plot_layout_techSupport)
pyoff.iplot(fig_techSupport)

#### StreamingTV

# Whether the customer has streaming TV or not (Yes, No, No internet service)

streamingTV_plot = telco_df.groupby('StreamingTV').Churn.mean().reset_index()
streamingTV_plot

(float(streamingTV_plot.query("StreamingTV=='No'")['Churn']) -
 float(streamingTV_plot.query("StreamingTV=='Yes'")['Churn'])) * 100

# Although not substational, customers that don't stream TV are roughly 3% more likely to churn.

plot_data_streamingTV = [
    go.Bar(
        x=streamingTV_plot['StreamingTV'],
        y=streamingTV_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Streaming TV',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Streaming TV</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_streamingTV = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Streaming TV',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_streamingTV = go.Figure(data=plot_data_streamingTV, layout=plot_layout_streamingTV)
pyoff.iplot(fig_streamingTV)

#### StreamingMovies 

# Whether the customer has streaming movies or not (Yes, No, No internet service)

streamingMovies_plot = telco_df.groupby('StreamingMovies').Churn.mean().reset_index()
streamingMovies_plot

(float(streamingMovies_plot.query("StreamingMovies=='No'")['Churn']) -
 float(streamingMovies_plot.query("StreamingMovies=='Yes'")['Churn'])) * 100

plot_data_streamingMovies = [
    go.Bar(
        x=streamingMovies_plot['StreamingMovies'],
        y=streamingMovies_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Streaming Movies',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Streaming Movies</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_streamingMovies = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Streaming Movies',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_streamingMovies = go.Figure(data=plot_data_streamingMovies, layout=plot_layout_streamingMovies)
pyoff.iplot(fig_streamingMovies)

#### Contract

# The contract term of the customer (Month-to-month, One year, Two year)

contract_plot = telco_df.groupby('Contract').Churn.mean().reset_index()
contract_plot

monthly_year1 = (float(contract_plot.query("Contract=='Month-to-month'")['Churn']) -
 float(contract_plot.query("Contract=='One year'")['Churn'])) * 100

monthly_year2 = (float(contract_plot.query("Contract=='Month-to-month'")['Churn']) -
 float(contract_plot.query("Contract=='Two year'")['Churn'])) * 100

year1_year2 = (float(contract_plot.query("Contract=='One year'")['Churn']) -
 float(contract_plot.query("Contract=='Two year'")['Churn'])) * 100

print(f'Churn Rate Difference Between Monthly & 1 Year = {round(monthly_year1, 2)}%\nChurn Rate Difference Between Monthly & 2 Year = {round(monthly_year2, 2)}%\nChurn Rate Difference Between 1 Year & 2 Year = {round(year1_year2, 2)}%')

# Customers with a monthly contract are substantially more likely to churn than those 1 year or 2 year contracts.

plot_data_contract = [
    go.Bar(
        x=contract_plot['Contract'],
        y=contract_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Contract',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Contract</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple'])
    )
]

plot_layout_contract = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Contract Type',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_contract = go.Figure(data=plot_data_contract, layout=plot_layout_contract)
pyoff.iplot(fig_contract)

#### PaperlessBilling

# Whether the customer has paperless billing or not (Yes, No)

paperlessBilling_plot = telco_df.groupby('PaperlessBilling').Churn.mean().reset_index()
paperlessBilling_plot

paperlessBilling = (float(paperlessBilling_plot.query("PaperlessBilling=='Yes'")['Churn']) -
 float(paperlessBilling_plot.query("PaperlessBilling=='No'")['Churn'])) * 100
print(f'Churn Rate Difference Between Paperless Billing = {round(paperlessBilling, 2)}%')

# Customers with Paperless Billing are roughly 17% more likely to churn than those without.

plot_data_paperlessBilling = [
    go.Bar(
        x=paperlessBilling_plot['PaperlessBilling'],
        y=paperlessBilling_plot['Churn'],
        width=[0.5, 0.5],
        name='Paperless Billing',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Paperless Billing</b>: %{x}<br>',
        marker= dict(color=['green', 'purple'])
    )
]

plot_layout_paperlessBilling = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Paperless Billing',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_paperlessBilling = go.Figure(data=plot_data_paperlessBilling, layout=plot_layout_paperlessBilling)
pyoff.iplot(fig_paperlessBilling)

#### Payment Method

# The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))

paymentMethod_plot = telco_df.groupby('PaymentMethod').Churn.mean().reset_index()
paymentMethod_plot

eCheck_Bank = (float(paymentMethod_plot.query("PaymentMethod=='Electronic check'")['Churn']) -
 float(paymentMethod_plot.query("PaymentMethod=='Bank transfer (automatic)'")['Churn'])) * 100

eCheck_CC = (float(paymentMethod_plot.query("PaymentMethod=='Electronic check'")['Churn']) -
 float(paymentMethod_plot.query("PaymentMethod=='Credit card (automatic)'")['Churn'])) * 100

print(f'Churn Rate Difference Between Electronic Check & Bank Transfer (automatic) = {round(eCheck_Bank, 2)}%\nChurn Rate Difference Between Electronic Check & Credit Card (automatic) = {round(eCheck_CC, 2)}%')

# Automating the payment makes the customer more likely to retain in your platform by roughly 30%.

plot_data_paymentMethod = [
    go.Bar(
        x=paymentMethod_plot['PaymentMethod'],
        y=paymentMethod_plot['Churn'],
        width=[0.5, 0.5, 0.5, 0.5],
        name='Payment Method',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Payment Method</b>: %{x}<br>',
        marker= dict(color=['green', 'slategray', 'purple', 'navy'])
    )
]

plot_layout_paymentMethod = go.Layout(
        xaxis={'type': "category"},
        yaxis={'title': "Churn Rate"},
        title='Churn Rate by Payment Method',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)'
)

fig_paymentMethod = go.Figure(data=plot_data_paymentMethod, layout=plot_layout_paymentMethod)
pyoff.iplot(fig_paymentMethod)

### Numerical Features:

#### Tenure:

# Number of months the customer has stayed with the company.

telco_df.tenure.describe()

tenure_plot = telco_df.groupby('tenure').Churn.mean().reset_index()
# Drop 0 value.
tenure_plot.loc[~(tenure_plot==0).all(axis=1)]
tenure_plot

# It appears that Churn rate decreases as tenure increases, let's visualize as Scatterplot.

plot_data_tenure = [
    go.Scatter(
        x=tenure_plot['tenure'],
        y=tenure_plot['Churn'],
        mode='markers',
        name='Tenure',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Tenure</b>: %{x}<br>',
        marker= dict(size=7,
            line= dict(width=1),
            color= 'green',
            opacity= 0.8
            )
    )
]

plot_layout_tenure = go.Layout(
        yaxis= {'title': "Churn Rate"},
        xaxis= {'title': 'Tenure'},
        title='Tenure Based Churn Rate',
        hovermode ='closest',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)"
    )

fig_tenure = go.Figure(data=plot_data_tenure, layout=plot_layout_tenure)
pyoff.iplot(fig_tenure)

#### Monthly Charges:

# The amount charged to the customer monthly.

telco_df.MonthlyCharges.describe()

monthlyCharges_plot = telco_df.copy()
monthlyCharges_plot['MonthlyCharges'] = monthlyCharges_plot['MonthlyCharges'].astype(int)
monthlyCharges_plot = monthlyCharges_plot.groupby('MonthlyCharges').Churn.mean().reset_index()
monthlyCharges_plot

# It's difficult to see any relationships from the DF table, let's visualize.

plot_data_monthlyCharges = [
    go.Scatter(
        x=monthlyCharges_plot['MonthlyCharges'],
        y=monthlyCharges_plot['Churn'],
        mode='markers',
        name='Monthly Charges',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Monthly Charges</b>: $%{x}<br>',
        marker= dict(size=7,
            line= dict(width=1),
            color= 'green',
            opacity= 0.8
            )
    )
]

plot_layout_monthlyCharges = go.Layout(
        yaxis= {'title': "Churn Rate"},
        xaxis= {'title': 'Monthly Charges'},
        title='Monthly Charges Based Churn Rate',
        hovermode ='closest',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)"
    )

fig_monthlyCharges = go.Figure(data=plot_data_monthlyCharges, layout=plot_layout_monthlyCharges)
pyoff.iplot(fig_monthlyCharges)

#### Total Charges:

# The total amount charged to the customer.

# Check telco_df to see if there are any missing, or Nan values.

telco_df[pd.to_numeric(telco_df['TotalCharges'], errors='coerce').isnull()]
# There does appear to be a few.

print(f'There are {len(telco_df[pd.to_numeric(telco_df["TotalCharges"], errors="coerce").isnull()])} missing values.')

# Convert missing values to Nan, then dropna.
telco_df.loc[pd.to_numeric(telco_df['TotalCharges'], errors='coerce').isnull(), 'TotalCharges'] = np.nan
telco_df = telco_df.dropna()

telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')

totalCharges_plot = telco_df.copy()
totalCharges_plot['TotalCharges'] = totalCharges_plot['TotalCharges'].astype(int)
totalCharges_plot = totalCharges_plot.groupby('TotalCharges').Churn.mean().reset_index()

plot_data_totalCharges = [
    go.Scatter(
        x=totalCharges_plot['TotalCharges'],
        y=totalCharges_plot['Churn'],
        mode='markers',
        name='Total Charges',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Total Charges</b>: $%{x}<br>',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'green',
            opacity= 0.8
           ),
    )
]

plot_layout_totalCharges = go.Layout(
        yaxis= {'title': "Churn Rate"},
        xaxis= {'title': "Total Charges"},
        title='Total Charge vs Churn rate',
        hovermode ='closest',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)",
    )
fig_totalCharges = go.Figure(data=plot_data_totalCharges, layout=plot_layout_totalCharges)
pyoff.iplot(fig_totalCharges)

# Unfortunately, there is no clear trend between Churn Rate and Monthly/Total Charges.

### Feature Engineering:

# In this section, we are going to transform our raw features to extract more information from them. Our strategy is as follows:

# 1. Group the numerical columns by using clustering techniques

# 2. Apply Label Encoder to categorical features which are binary

# 3. Apply get_dummies() to categorical features which have multiple values

# ##### Numerical Columns

# As we know from the EDA section, We have three numerical columns:

# * Tenure

# * Monthly Charges

# * Total Charges

# We are going to apply the following steps to create groups:

# 1. Using Elbow Method to identify the appropriate number of clusters.

# 2. Applying K-means logic to the selected column and change the naming.

# 3. Observe the profile of clusters.

# Let’s check how this works for Tenure in practice:

##### Cluster profiles:

# Function for ordering cluster numbers.
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

#### Tenure Cluster:

# We should tell how many clusters we need to K-means algorithm.
# To find it out, we will apply Elbow Method.
# Elbow Method simply tells the optimal cluster number for optimal inertia.

# Empty dictionary for SSE: sum of the squared differences between each observation and its group's mean.
sse={}
df_cluster = telco_df[['tenure']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)
    df_cluster['clusters'] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of Clusters")
plt.show()

# Apply KMeans to predict Tenure Clusters.
kmeans = KMeans(n_clusters=3)
kmeans.fit(telco_df[['tenure']])
telco_df['TenureCluster'] = kmeans.predict(telco_df[['tenure']])

# Apply order_cluster function to Tenure Cluster.
telco_df = order_cluster('TenureCluster', 'tenure', telco_df, True)

telco_df.groupby('TenureCluster').tenure.describe()

# Apply Low, Mid, High labels to each Tenure Cluster.
telco_df['TenureCluster'] = telco_df['TenureCluster'].replace({0:'Low', 1:'Mid', 2:'High'})

# Group by each Tenure Cluster and calculate the average churn rate for each.
tenureCluster_plot = telco_df.groupby('TenureCluster').Churn.mean().reset_index()
tenureCluster_plot

# Plot Churn Rate for each Tenure Cluster.
plot_data_tenureCluster = [
    go.Bar(
        x=tenureCluster_plot['TenureCluster'],
        y=tenureCluster_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Tenure Cluster',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Tenure Cluster</b>: %{x}<br>',
        marker= dict(
            color=['green', 'slategray', 'purple']
        )
    )
]

plot_layout_tenureCluster = go.Layout(
        xaxis={"type": "category", "categoryarray": ['Low', 'Mid', 'High']},
        title='Tenure Cluster vs. Churn Rate',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)"
)

fig_tenureCluster = go.Figure(data=plot_data_tenureCluster, layout=plot_layout_tenureCluster)
pyoff.iplot(fig_tenureCluster)

#### Monthly Charges Cluster:

# Apply Elbow Method for Monthly Charges to determine optimal cluster number.
sse={}
df_cluster = telco_df[['MonthlyCharges']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)
    df_cluster["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of Clusters")
plt.show()

# Apply KMeans to predict Monthly Charges Clusters.
kmeans = KMeans(n_clusters=3)
kmeans.fit(telco_df[['MonthlyCharges']])
telco_df['MonthlyChargesCluster'] = kmeans.predict(telco_df[['MonthlyCharges']])

# Apply order_cluster function to Monthly Charges Cluster.
telco_df = order_cluster('MonthlyChargesCluster', 'MonthlyCharges', telco_df, True)

telco_df.groupby('MonthlyChargesCluster').MonthlyCharges.describe()

# Apply Low, Mid, High labels to each Monthly Charges Cluster.
telco_df['MonthlyChargesCluster'] = telco_df['MonthlyChargesCluster'].replace({0: 'Low', 1: 'Mid', 2: 'High'})

# Group by each Monthly Charges Cluster and calculate the average churn rate for each.
monthlyChargesCluster_plot = telco_df.groupby('MonthlyChargesCluster').Churn.mean().reset_index()
monthlyChargesCluster_plot

# Plot Churn Rate for each Monthly Charges Cluster.
plot_data_monthlyChargesCluster = [
    go.Bar(
        x=monthlyChargesCluster_plot['MonthlyChargesCluster'],
        y=monthlyChargesCluster_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Monthly Charges Cluster',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Monthly Charges Cluster</b>: %{x}<br>',
        marker= dict(
            color=['green', 'slategray', 'purple']
        )
    )
]

plot_layout_monthlyChargesCluster = go.Layout(
        xaxis={"type": "category", "categoryarray": ['Low', 'Mid', 'High']},
        title='Monthly Charges Cluster vs. Churn Rate',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)"
)

fig_monthlyChargesCluster = go.Figure(data=plot_data_monthlyChargesCluster, layout=plot_layout_monthlyChargesCluster)
pyoff.iplot(fig_monthlyChargesCluster)

#### Total Charges Cluster:

# Apply Elbow Method for Total Charges to determine optimal cluster number.
sse={}
df_cluster = telco_df[['TotalCharges']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)
    df_cluster["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of Clusters")
plt.show()

# Apply KMeans to predict Total Charges Clusters.
kmeans = KMeans(n_clusters=3)
kmeans.fit(telco_df[['TotalCharges']])
telco_df['TotalChargesCluster'] = kmeans.predict(telco_df[['TotalCharges']])

# Apply order_cluster function to Total Charges Cluster.
telco_df = order_cluster('TotalChargesCluster', 'TotalCharges', telco_df, True)

telco_df.groupby('TotalChargesCluster').TotalCharges.describe()

# Apply Low, Mid, High labels to each Total Charges Cluster.
telco_df['TotalChargesCluster'] = telco_df['TotalChargesCluster'].replace({0: 'Low', 1: 'Mid', 2: 'High'})

# Group by each Total Charges Cluster and calculate the average churn rate for each.
totalChargesCluster_plot = telco_df.groupby('TotalChargesCluster').Churn.mean().reset_index()
totalChargesCluster_plot

# Plot Churn Rate for each Total Charges Cluster.
plot_data_totalChargesCluster = [
    go.Bar(
        x=totalChargesCluster_plot['TotalChargesCluster'],
        y=totalChargesCluster_plot['Churn'],
        width=[0.5, 0.5, 0.5],
        name='Total Charges Cluster',
        hovertemplate =
    '<b>Churn Rate</b>: %{y:.2%}'+
    '<br><b>Total Charges Cluster</b>: %{x}<br>',
        marker= dict(
            color=['green', 'slategray', 'purple']
        )
    )
]

plot_layout_totalChargesCluster = go.Layout(
        xaxis={"type": "category", "categoryarray": ['Low', 'Mid', 'High']},
        title='Total Charges Cluster vs. Churn Rate',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)"
)

fig_totalChargesCluster = go.Figure(data=plot_data_totalChargesCluster, layout=plot_layout_totalChargesCluster)
pyoff.iplot(fig_totalChargesCluster)

### Copy Dataframe for Final Table in Dashboard

final_telco_df = telco_df.copy()

#### Categorical Features:

telco_df.info()

# Use label encoder to convert categorical coumns to numerical.
le = LabelEncoder()

# Array for multiple value columns.
dummy_columns = []

for column in telco_df.columns:
    if telco_df[column].dtype == object and column != 'customerID':
        if telco_df[column].nunique()  == 2:
            # Apply label encoder for binary Object columns.
            telco_df[column] = le.fit_transform(telco_df[column])
        else:
            dummy_columns.append(column)
            


print(dummy_columns)

telco_df.head()

# Apply get dummies for selected columns that did not meet binary label encoder parameters.
telco_df = pd.get_dummies(data=telco_df, columns=dummy_columns)
telco_df.head()

telco_df.info()

### Logistic Regression:

# Predicting churn is a binary classification problem. Customers either churn or retain in a given period. Along with being a robust model, Logistic Regression provides interpretable outcomes too.

# Our steps are as follows:

# 1. Prepare the data (inputs for the model).

# 2. Fit the model and see the summary.

# We now need to clean up column names and replace the following characters; " ", "(", ")", "-", with "_", for consistency.

# Empty array to append to once each column name has been transformed.
all_columns = []

# Iterate through list of columns, transform column names and then append each back to all_columns array. 
for column in telco_df.columns:
    column = column.replace(" ", "_").replace("(", "_").replace("-", "_").replace(")", "")
    all_columns.append(column)
    
# Set DF column names to transformed names.
telco_df.columns = all_columns

telco_df.info()

# Now we will prepare the columns that will be utilized as inputs for the Generalized Linear Regression Model.
glm_columns = 'gender'

for column in telco_df.columns:
    if column not in ['Churn', 'customerID', 'gender']:
        glm_columns = glm_columns + ' + ' + column

glm_columns

# Let's fit the model and then visualize the results.
import statsmodels.api as sm
import statsmodels.formula.api as smf

glm_model = smf.glm(formula='Churn ~ {}'.format(glm_columns), data=telco_df, family=sm.families.Binomial())
results = glm_model.fit()
print(results.summary())

# We have two important outcomes from this report. When we prepare a Churn Prediction model, we will be faced with the questions below:

# 1. Which characteristics make customers churn or retain?

# 2. What are the most critical ones? What should we focus on?

# For which characteristics make customers churn or retain, we should look at (P>|z|). If the absolute **p-value** is smaller than **0.05**, then that feature affects churn in a statistically significant way.

# That includes:

# * Tenure

# * PaperlessBilling

# * OnlineSecurity_No

# * TechSupport_No

# * Contract_Month_to_Month

# * Contract_Two_Year

# * PaymentMethod_Electronic_check

# * InternetService_DSL

# * SeniorCitizen

# For our second question, we want to reduce the Churn Rate, and increase Customer Retention.

# In other words, *which feature will bring the best ROI if I increase/decrease it by one unit?*

# That question can be answered by looking at the coef column. Exponential coef gives us the expected change in Churn Rate if we change it by one unit. For example, one unit change in Monthly Charge means ~3.4% improvement in the odds for churning if we keep everything else constant. If we apply the code below, we will see the transformed version of all coefficients:

np.exp(results.params)

### Binary Classification Model with XGBoost

# To fit our XGBoost to our data, let's first prepare features (X) and label (y) sets and do the train test split.

# Create feature sset and lables.
X = telco_df.drop(['Churn', 'customerID'], axis=1)
y = telco_df.Churn

# Train/Test Split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

# Build the model and print the score.
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08,
                             objective= 'binary:logistic', n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))

# Make predictions on test set.
y_pred = xgb_model.predict(X_test)

# Print Classification Report.
print(classification_report(y_test, y_pred))

# We can interpret the report above as if our model tells us, 100 customers will churn, 67 of it will churn (0.67 precision). And actually, there are around 220 customers who will churn (0.45 recall). Especially recall is the main problem here, and we can improve our model’s overall performance by:

# * Adding more data (we have around 2k rows for this example)

# * Adding more features

# * More feature engineering

# * Trying other models

# * Hyper-parameter tuning

# Moving forward, let’s see how our model works in detail. First off, we want to know which features our model exactly used from the dataset. Also, which were the most important ones?

# For addressing this question, we can use the code below:

from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(xgb_model, ax=ax)

x = [1, 1, 2, 4, 5, 9, 10, 13, 14, 15, 17, 18, 18, 18, 18, 18, 19, 22, 30, 32, 37, 40, 42, 44, 46, 52, 53, 58, 75, 83, 109, 313, 540, 579]
y = ['OnlineSecurity_Yes', 'TechSupport_Yes', 'TenureCluster_Mid', 'MonthlyChargesCluster_Mid', 'OnlineBackup_Yes', 'TotalChargesCluster_Mid',
                'MultipleLines_Yes', 'InternetService_DSL', 'StreamingTV_Yes', 'InternetService_Fiber_optic', 'DeviceProtection_No',
                 'StreamingTV_No', 'StreamingMovies_Yes', 'StreamingMovies_No', 'PaymentMethod_Mailed_check', 'PaymentMethod_Bank_transfer_automatic_',
                 'Partner', 'PhoneService', 'Contract_Two_year', 'MultipleLines_No', 'PaymentMethod_Credit_card_automatic_', 'Dependents',
                 'Contract_Month_to_month', 'Contract_One_Year', 'TechSupport_No', 'PaperlessBilling', 'OnlineBackup_No', 'OnlineSecurity_No',
                 'SeniorCitizen', 'gender', 'PaymentMethod_Electronic_check', 'tenure', 'MonthlyCharges','TotalCharges']
color = ['slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green',
                  'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 'slategray', 'purple', 'green', 
                  'slategray', 'purple', 'green','slategray', 'purple', 'green', 'slategray']
# Plot Feature Importance for Plotly.
plot_data_featureImportance = [
    go.Bar(
        x=x,
        y=y,
        name='Feature Importance',
        hovertemplate =
    '<b>Feature</b>: %{y}'+
    '<br><b>F Score</b>: %{x}<br>',
        marker= dict(
            color=color
        ),
        orientation='h'
    )
]

plot_layout_featureImportance = go.Layout(
        yaxis={"type": "category"},
        title='Feature Importance',
        margin=dict(l=250, r=50, t=50, b=50),
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)"
)

fig_featureImportance = go.Figure(data=plot_data_featureImportance, layout=plot_layout_featureImportance)
fig_featureImportance.update_xaxes(title_text='F-Score')
fig_featureImportance.update_yaxes(title_text='Feature', tickfont=dict(size=9), )
pyoff.iplot(fig_featureImportance)

# We can see that our model assigned more importance to TotalCharges and MonthlyCharges compared to others.

# Finally, the best way to use this model is assigning Churn Probability for each customer, create segments, and build strategies on top of that.

telco_df['proba'] = xgb_model.predict_proba(telco_df[X_train.columns])[:,1]

telco_df[['customerID', 'proba']].head(10)

# Now that we know which customers are most likely to churn, we can go ahead and build actions based on our results.

# Create probability df and merge with final df. 
prob_telco_df = telco_df[['customerID', 'proba']]

# Merge both DFs back together.
final_telco_df = final_telco_df.merge(prob_telco_df, on='customerID')

# Convert Churn & SeniorCitizen binary integers back to Yes/No.
final_telco_df.loc[final_telco_df.Churn==0, 'Churn'] = 'No'
final_telco_df.loc[final_telco_df.Churn==1, 'Churn'] = 'Yes'
final_telco_df.loc[final_telco_df.SeniorCitizen==0, 'SeniorCitizen'] = 'No'
final_telco_df.loc[final_telco_df.SeniorCitizen==1, 'SeniorCitizen'] = 'Yes'
final_telco_df.head()


final_telco_df.columns

rowEvenColor = 'green'
rowOddColor = 'purple'
figProba = go.Figure(data=[go.Table(
#     columnwidth = [1000,400],
    header=dict(values=final_telco_df.columns,
                fill_color='slategray',
                align='left',
                font=dict(color='white', size=12)),
    cells=dict(values=[final_telco_df.customerID, final_telco_df.gender, final_telco_df.SeniorCitizen, final_telco_df.Partner,
                       final_telco_df.Dependents, final_telco_df.tenure, final_telco_df.PhoneService, final_telco_df.MultipleLines,
                       final_telco_df.InternetService, final_telco_df.OnlineSecurity, final_telco_df.OnlineBackup, final_telco_df.DeviceProtection,
                       final_telco_df.TechSupport, final_telco_df.StreamingTV, final_telco_df.StreamingMovies, final_telco_df.Contract,
                       final_telco_df.PaperlessBilling, final_telco_df.PaymentMethod, final_telco_df.MonthlyCharges, final_telco_df.TotalCharges,
                       final_telco_df.Churn, final_telco_df.TenureCluster, final_telco_df.MonthlyChargesCluster, final_telco_df.TotalChargesCluster,
                       (final_telco_df.proba)*100],
               fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor]*1758],
               align='left',
               font= dict(color='white', size=12)))
])

pyoff.iplot(figProba)