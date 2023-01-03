import pandas as pd
import numpy as np
import os
from env import get_connection
from sklearn.model_selection import train_test_split

def prep_telco_data(telco_churn):
    telco_churn = telco_churn.drop(columns=['internet_service_type_id', 'contract_type_id','payment_type_id'])

    telco_churn['gender_encoded'] = telco_churn.gender.map({'Female': 1, 'Male': 0})
    telco_churn['partner_encoded'] = telco_churn.partner.map({'Yes': 1, 'No': 0})
    telco_churn['dependents_encoded'] = telco_churn.dependents.map({'Yes': 1, 'No': 0})
    telco_churn['phone_service_encoded'] = telco_churn.phone_service.map({'Yes': 1, 'No': 0})
    telco_churn['paperless_billing_encoded'] = telco_churn.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_churn['churn_encoded'] = telco_churn.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(telco_churn[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    telco_churn = pd.concat( [telco_churn, dummy_df], axis=1 )
    
    return telco_churn
 
   
    
def split_data(df, target=''):
        train, test = train_test_split(df, 
                               train_size = 0.8,
                               random_state=1349,
                              stratify=df[target])
        train, val = train_test_split(train,
                             train_size = 0.7,
                             random_state=1349,
                             stratify=train[target])
        return train, val, test
    
    
    





