#!/usr/bin/env python
# coding: utf-8

# # Assignment 6 and 7 Analysis Script

# Import libraries.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import data from Kaggle.

# In[3]:


data = pd.read_csv('Metro_Nashville_Police_Department_Incidents.csv')
data = data.dropna(subset=['Incident Reported']).reset_index(drop=True)
data.head()


# In[188]:


data = data[(data['Incident Reported'].str.contains('2021')) | (data['Incident Reported'].str.contains('2020'))]
data.to_csv('data_reduced.csv')


# Start building the dataset. 
# 
# 1) Drop columns that will not be needed and reset the indices
# 
# 2) The "count column" for davidson_data should include the cumulative counts at each day.

# In[4]:


davidson_data = pd.read_csv('COVID.csv')
davidson_data = davidson_data.drop(columns=['NEVER','RARELY','SOMETIMES','FREQUENTLY','ALWAYS'])
davidson_data = davidson_data.T
davidson_data = davidson_data.rename(columns={'0':'count'})
davidson_data['date'] = davidson_data.index
davidson_data.reset_index(drop=False)
davidson_data = davidson_data.reset_index().drop(columns=['index'])
davidson_data = davidson_data.rename(columns={0:'count'})
davidson_data = davidson_data.iloc[12:].reset_index(drop=True)
davidson_data.head()


# Write a function to parse the dates and times of the incidents.

# In[5]:


from datetime import datetime
def parse_date_time(date_time):
    return datetime.strptime(str(date_time),'%m/%d/%Y %I:%M:%S %p')


# Loop through the "incident reported" column and pick out only the years we are researching (2020 and 2021)

# In[6]:


indices = []
for i in range(0,len(data['Incident Reported'])):
    if str(parse_date_time(data['Incident Reported'][i]).year) == '2020':
        indices.append(i)
    elif str(parse_date_time(data['Incident Reported'][i]).year) == '2021':
        indices.append(i)


# Reset the indices for the data frame.

# In[7]:


covid_crime_data = data.iloc[indices]
covid_crime_data = covid_crime_data.reset_index(drop=True)


# Parse the date-time value in the "Incident Reported" column to separate it into month, day, and year values.

# In[8]:


year = []
month = [] 
day = []
for i in range(0,len(covid_crime_data['Incident Reported'])):
   year.append(parse_date_time(covid_crime_data['Incident Reported'][i]).year)
   month.append(parse_date_time(covid_crime_data['Incident Reported'][i]).month)
   day.append(parse_date_time(covid_crime_data['Incident Reported'][i]).day)
covid_crime_data['year'] = year
covid_crime_data['month'] = month
covid_crime_data['day'] = day
covid_crime_data = covid_crime_data.sort_values(['year', 'month', 'day'], ascending=(True))


# Reset the indices for the COVID cumulative cases data frame, rename the columns, and obtain only the dates of COVID cases that we need.

# In[9]:


davidson_data.reset_index(drop=False)
davidson_data = davidson_data.reset_index().drop(columns=['index'])
davidson_data = davidson_data.rename(columns={0:'count'})
davidson_data = davidson_data.iloc[12:].reset_index(drop=True)
davidson_data


# Use the parse_date_time function above to parse through the police incident data.

# In[10]:


year = []
month = []
day = []
for i in range(0,len(davidson_data['date'])):
    year.append(datetime.strptime(davidson_data['date'][i], '%m/%d/%y').year)
    month.append(datetime.strptime(davidson_data['date'][i], '%m/%d/%y').month)
    day.append(datetime.strptime(davidson_data['date'][i], '%m/%d/%y').day)
davidson_data['year'] = year
davidson_data['month'] = month
davidson_data['day'] = day


# ## Plots

# Create a few elementary plots to understand the trends over time for both the COVID case counts and police incident counts.
# 
# 1) Plot the number of police incidents by day in Davidson County, TN.
# 
# 2) Plot the number of cumulative COVID cases over time in Davidson County, TN.
# 
# 3) Plot the difference in COVID cases per day in Davidson County, TN.
# 
# 4) Plot the difference in police incident cases per day in Davidson County, TN. 

# In[176]:


plt.plot(range(0,625),covid_crime_data.groupby(['year','month','day']).size().values)
plt.xlabel('Date number')
plt.ylabel('Case count')


# In[177]:


plt.plot(davidson_data.index,davidson_data['count'])
plt.xlabel('Date number')
plt.ylabel('Cumulative case count')


# ## Plot the first derivative 

# In[13]:


covid_case_differences = [y-x for x, y in zip(davidson_data['count'][:-1], davidson_data['count'][1:])]
covid_case_differences


# In[14]:


vals = covid_crime_data.groupby(['year','month','day']).size().values
covid_crime_differences = [y-x for x, y in zip(vals[:-1], vals[1:])]
covid_crime_differences


# In[179]:


plt.plot(range(0,637),covid_case_differences)
plt.xlabel("Date Number")
plt.ylabel('Case difference day by day')


# In[180]:


plt.figure(figsize=(60,20))
plt.plot(range(0,624),covid_crime_differences)
plt.xlabel("Date Number")
plt.ylabel("Difference in counts from previous day")


# ## Determine key time periods and run statistical tests

# Create the data frame that will tell us the change in police incident counts per day.

# In[18]:


df_covid_crime_diff = pd.DataFrame(covid_crime_data.groupby(['year','month','day']).size().index[1:],covid_crime_differences)
df_covid_crime_diff['difference'] = df_covid_crime_diff.index
df_covid_crime_diff = df_covid_crime_diff.reset_index(drop=True)
df_covid_crime_diff = df_covid_crime_diff.rename(columns={0:'date'})
df_covid_crime_diff.head()


# Calculate the column that will tell us the differences between COVID-19 case counts per day.

# In[19]:


df_covid_case_diff = pd.DataFrame(davidson_data.groupby(['year','month','day']).size().index[1:],covid_case_differences)
df_covid_case_diff['difference'] = df_covid_case_diff.index
df_covid_case_diff = df_covid_case_diff.reset_index(drop=True)
df_covid_case_diff = df_covid_case_diff.rename(columns={0:'date'})
df_covid_case_diff.head()


# Add the columns in both data frames that represent the COVID-19 case counts and police incident counts per day.

# In[20]:


df_covid_case_diff['case count'] = davidson_data['count'][1:].reset_index(drop=True)
df_covid_crime_diff['incident count'] = covid_crime_data.groupby(['year','month','day']).size().values[1:]


# Set up the column in both data frames that will include the dates in M/DD/YYYY.

# In[23]:


date_slash = []
for i in range(0,len(df_covid_case_diff)):
    var = df_covid_case_diff['date'][i]
    date_slash.append(str(var[0]) + '/' + str(var[1]) + '/' + str(var[2]))
df_covid_case_diff['date_slash'] = date_slash

date_slash = []
for i in range(0,len(df_covid_crime_diff)):
    var = df_covid_crime_diff['date'][i]
    date_slash.append(str(var[0]) + '/' + str(var[1]) + '/' + str(var[2]))
df_covid_crime_diff['date_slash'] = date_slash


# Rename the columns accordingly.

# In[24]:


df_covid_crime_diff = df_covid_crime_diff.rename(columns={'case difference':'incident difference'})
df_covid_case_diff = df_covid_case_diff.rename(columns={'incident difference':'case difference'})


# Merge the two data frames so that we only have counts and differences for both COVID cases and police incidents for common days (February 2, 2020 to September 16, 2021).

# In[26]:


combined = pd.merge(df_covid_crime_diff, df_covid_case_diff, on='date_slash',how='left').dropna().reset_index(drop=True).drop(columns=['date_y']).rename(columns={'date_x':'date'})


# Rename the columns accordingly after the merge.

# In[29]:


combined = combined.rename(columns={'difference_x': 'incident difference', 'difference_y': 'case difference'})


# ## Statistical Testing (two sample T test) 

# Validate the assumptions that exist for two-sample t-testing:
# 
# 1) The data points of the incident counts are continuous.
# 
# 2) The data points of the incident counts follow a relatively normal distribution. 

# In[67]:


plt.hist(combined['incident count'])
plt.xlabel('Incidents per day')
plt.ylabel('Frequency')


# Find the indices of the dates and time periods we will analyze.

# In[30]:


idx_mask_mandate_start = combined[combined['date_slash']=='2020/7/3'].index.values[0]
idx_mask_mandate_end = combined[combined['date_slash']=='2021/5/14'].index.values[0]
idx_pres_election_start = combined[combined['date_slash']=='2020/11/7'].index.values[0]
idx_pres_election_end = combined[combined['date_slash']=='2021/1/20'].index.values[0]
idx_covid_vacc_start = combined[combined['date_slash']=='2021/3/8'].index.values[0]
idx_covid_vacc_end = combined[combined['date_slash']=='2021/9/16'].index.values[0]


# Print the means of the case counts during the time period we are analyzing.

# In[31]:


print(np.mean(combined['case difference'][idx_mask_mandate_start:idx_mask_mandate_end]))
print(np.mean(combined['case difference'][idx_pres_election_start:idx_pres_election_end]))
print(np.mean(combined['case difference'][idx_covid_vacc_start:idx_covid_vacc_end]))


# Run the two-sample t-test for the police incident counts during the time periods we decided to analyze.
# 
# 1) Test 1: Case counts from the start to the end of the COVID vaccine availability vs. all other days not in that category
# 
# 2) Test 2: Case counts from the start to the end of the presidential election/lame-duck period vs. all other days not in that category
# 
# 3) Test 3: Case counts from the start to the end of the county mask mandate vs. all other days not in that category

# In[152]:


import scipy.stats
print(scipy.stats.ttest_ind(combined['case difference'][idx_covid_vacc_start:idx_covid_vacc_end],combined['case difference'].iloc[np.r_[0:idx_covid_vacc_start, idx_covid_vacc_end:590]]).pvalue)
print(scipy.stats.ttest_ind(combined['case difference'][idx_pres_election_start:idx_pres_election_end],combined['case difference'].iloc[np.r_[0:idx_pres_election_start, idx_pres_election_end:590]]).pvalue)
print(scipy.stats.ttest_ind(combined['case difference'][idx_mask_mandate_start:idx_mask_mandate_end],combined['case difference'].iloc[np.r_[0:idx_mask_mandate_start, idx_mask_mandate_end:590]]).pvalue)


# Run the two-sample t-test for the police incident counts during the time periods we decided to analyze.
# 
# 1) Test 1: Incident counts from the start to the end of the COVID vaccine availability vs. all other days not in that category
# 
# 2) Test 2: Incident counts from the start to the end of the presidential election/lame-duck period vs. all other days not in that category
# 
# 3) Test 3: Incident counts from the start to the end of the county mask mandate vs. all other days not in that category

# In[64]:


import scipy.stats
print(scipy.stats.ttest_ind(combined['incident count'][idx_covid_vacc_start:idx_covid_vacc_end],combined['incident count'].iloc[np.r_[0:idx_covid_vacc_start, idx_covid_vacc_end:590]]).pvalue)
print(scipy.stats.ttest_ind(combined['incident count'][idx_pres_election_start:idx_pres_election_end],combined['incident count'].iloc[np.r_[0:idx_pres_election_start, idx_pres_election_end:590]]).pvalue)
print(scipy.stats.ttest_ind(combined['incident count'][idx_mask_mandate_start:idx_mask_mandate_end],combined['incident count'].iloc[np.r_[0:idx_mask_mandate_start, idx_mask_mandate_end:590]]).pvalue)


# # Linear Regression Model Build

# Build the linear regression model to see if we can predict the police incident counts based on the COVID-19 case counts in Davidson County, TN. 

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(combined['case difference'], combined['incident count'], test_size = 0.2, random_state=42)
model = LinearRegression()
model.fit([X_train],[y_train])


# Predict police incident counts based on the COVID-19 case counts and change the data shape into an one-dimensional array. 

# In[89]:


predictions = model.predict([X_test])
preds = []
for i in range(0,len(predictions[0])):
    preds.append(predictions[0][i])


# Plot the 119 predicted police incident counts (in blue) and the 119 actual test set police incident counts (in orange).

# In[175]:


plt.figure(figsize=(40,10))
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Police Incidents')
plt.title('Prediction of Police Incidents')
plt.plot(np.arange(0,119),preds)
plt.plot(np.arange(0,119),y_test)
plt.scatter(np.arange(0,119),preds)
plt.scatter(np.arange(0,119),y_test)
plt.show()


# Print the mean difference between the prediction set and the test set.
# 
# Calculate the number of predicted values whose difference between it and the true value were less than the mean difference.

# In[189]:


mean_diff = np.mean(y_test - preds)
differences = np.array(preds-y_test)
print(mean_diff)
print(len(differences[np.where((np.absolute(differences) <= mean_diff))]))


# Calculate the RMSE (root-mean-squared-error) for the predictions against the test set.

# In[155]:


from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(preds,y_test)))

