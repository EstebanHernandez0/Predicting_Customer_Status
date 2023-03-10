# Telco Churn 
------
# Project Description 

This project will be based on the Telco Churn data. Telco is a company that provides multiple services to its customers.
Some of these services include, but not limited to: phone service, device protection, movie/tv streaming, and online security.
The company has noticed a spike in the amount of customers leaving and have started to ask what is causing this spike. I have decided to take a deeper look
to attempt to find the main cause for customers leaving.

------
# Project Goals
+ Find significant causes of churn
+ Take the found drivers and develop a machine learning that can help predict if a customer is at risk of churrning
+ Make code reproducable

-----
# Data Dictionary


| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| customer_id | Unique id for each customer| string |
| senior_citizen| 1 if customer is a senior citizen, 0 if not | int |
| tenure | Months of tenure as a customer| int |
| monthly_charges| The customer's monthly bill| float |
| total_charges| The customer's total bills since they have been a customer| float|
| gender_encoded | 1 if the customer is male, 0 if not | int |
| partner_encoded | 1 if the customer has a partner, 0 if not  | int |
| dependents_encoded| 1 if the customer has dependents, 0 if not| int |
| phone_service_encoded | 1 if the customer has phone service, 0 if not | int |
| paperless_billing_encoded | 1 if the customer has paperliess billing, 0 if not | int |
| multiple_lines_yes | 1 if the customer has multiple phone lines, 0 if not | int |
| online_security_no | 1 if the customer has internet but no online security, 0 if not | int |
| online_security_yes | 1 if the customer has online security add-on, 0 if not | int |
| online_backup_no | 1 if the customer has internet but no online backup, 0 if not | int |
| online_backup_yes | 1 if the customer has online backup, 0 if not | int |
| device_protection_no | 1 if the customer has internet but no device protection, 0 if not | int |
| device_protection_yes | 1 if the customer has device protection, 0 if not | int |
| tech_support_no | 1 if the customer has internet but no tech support, 0 if not | int |
| tech_support_yes | 1 if the customer has tech_support, 0 if not | int |
| streaming_tv_no | 1 if the customer has internet but no streaming tv, 0 if not | int |
| streaming_tv_yes | 1 if the customer has streaming tv, 0 if not | int |
| streaming_movies_no | 1 if the customer has internet but no streaming movies, 0 if not | int |
| streaming_movies_yes | 1 if the customer has streaming movies, 0 if not | int |
| contract_type_one_year | 1 if the customer has a one year contract, 0 if not | int |
| contract_type_two_year | 1 if the customer has a two year contract, 0 if not | int |
| payment_type_bank_transfer_auto | 1 if the customer pays by automatic bank transfer, 0 if not | int
| payment_type_credit_card_auto | 1 if the customer pays automatically by credit card, 0 if not | int
| payment_type_electronic_check | 1 if the customer pays manually by electronic check, 0 if not | int
| payment_type_mailed_check | 1 if the customer pays manually by mailed check, 0 if not | int
| internet_type_dsl  | 1 if the customer has DSL internet service, 0 if not |  int
| internet_type_fiber_optic | 1 if the customer has fiber optic internet service, 0 if not | int
| internet_type_none | 1 if the customer has no internet | int
| num_addons | sum of how many internet service add-ons the customer has | int
 


------
# Initial Hypothesis

My initial hypothesis is that the main reason for churn will be realted to how much a customer has to pay per month.

------
# My Plan of Action

+ Aquire data from Codeup
  - Use Sequel Ace to obtain, and filter for the needed data
  
+ Prepare the acquired data
  - Download and turn Telco data into a .csv
  - Bring in data into our jupyter notebook
  - Use pandas library
    - to clean data 
    - turn any columns from that need to be `object` to `int64`
    - turn any columns from that need to be `int64` to `object`
    - make `object` columns into numerical columns (No's and Yes's to  0's and 1's respectively)
  - Split our data into three parts 
    - `Train` which will have the largest amount of the data so we can use it to make our model
    - `Validate` which is used to make sure our model is the best it can before finally moving on
    - `Test` which is the final determinator to see if the model is good enough, the last part of modeling

Exploration
  
 - Are senior citizens more likely to churn than those who are not senior citizens?
 - The mean monthly charges of churned customers is higher than the overall average
 - Do month-to-month customers churn more than the rest?
 - Do customers who have device protection churn more than customes without device protection?
 
Modeling

- Use or make functions to help build our models 

- Build models 
  - Create our object
  - fit the model to find the most useful model
  - Use the best model on the in-sample data
  - Use the best model on the out-of-sample data
  
Delivery
  
  - Visualze to help the audiance easily see the findings of the work that was done
  - Model to help predict for a specified outcome
  
  ----
  # How to Reproduce the work
  
  - Clone this repo
  - Put data in the same file as the repo
  - Run the notebook
 

























