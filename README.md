# Telco Telecoms Customer Churn Predictor

## Overview
Retention Rate is an indication of how good your product market fit (PMF). If your PMF is not satisfactory, you should see your customers churning very soon. One of the powerful tools to improve Retention Rate (hence the PMF) is Churn Prediction. By using this technique, you can easily find out who is likely to churn in the given period.

Steps to develop this churn prediction model:

* Exploratory data analysis

  * The data falls under two categories:
  
    * Categorical Features: gender, streaming tv, payment method, etc.
    * Numerical Features: tenure, monthly charges, and total charges.
    
* Feature engineering

  * **In this section, we transform our raw features to extract more information from them. Strategy is as follows:**
  
    1. Group the numerical columns by using clustering techniques
    2. Apply Label Encoder to categorical features which are binary
    3. Apply get_dummies() to categorical features which have multiple values
    
  * Numerical Columns
  
    1) Tenure
    2) Monthly Charges
    3) Total Charges
    
        * We are going to apply the following steps to create groups:
        * Using Elbow Method to identify the appropriate number of clusters.
        * Applying K-means logic to the selected column and change the naming.
        * Observe the profile of clusters.
        
* Investigating how the features affect Retention by using Logistic Regression

  * Predicting churn is a binary classification problem. Customers either churn or retain in a given period. Along with being a robust model, Logistic Regression provides interpretable outcomes too.
    
    **When we prepare a Churn Prediction model, we will be faced with the important questions:**
    
    1. Which characteristics make customers churn or retain?
    2. What are the most critical ones? What should we focus on?
    
    **Steps Taken**
    
        1. Prepare the data (inputs for the model).
        2. Fit the model and see the summary.
* Building a Binary Classification Model with XGBoost

  * First prepare features (X) and label (y) sets and do the train test split.
  
  * Determine which of our features our model used from the dataset and which were the most important ones for predicting customer churn.
    * Our model helps us to determine that the most important features are:
        * TotalCharges
        * TotalMonthlyCharges
        * CustomerTenure
        * Payment_Method_ECheck
    * Finally, the best way to use this model is to assign Churn Probability for each customer, create segments and build strategies on top of that: **ultimately, we can try to convert customers to pay on an annual contract, which will increase tenure and goodwill, and finally convert customer to Electronic Payments.**
    
## Telco Customer Churn Plotly-Dash Dashboard:

* Please reach out to **zgrinacoff@gmail.com** for access to the [Plotly-Dash Dashboard Application](https://churn-app.herokuapp.com/ "Application") as User Name and Password are required.

## Directory:

* Root:
  * app_telco_dash.py
  * functions.py
  * telco_customer_churn.ipynb
  * Procfile
  * README.md
  * data: **All data used for this project is located on [Kaggle](https://www.kaggle.com/lampubhutia/telecomcustomer-churn/data).**
    * Telco-Customer-Churn.csv
    * final_telco_df.csv
    * future_retention_df.csv
    * monthlyChargesCluster.csv
    * tenureCluster.csv
    * totalChargesCluster.csv
