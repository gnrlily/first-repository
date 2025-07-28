#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


data = pd.read_csv('data/trip.csv')


# In[9]:


data.head()


# In[11]:


data.info()


# In[29]:


data.describe()


# In[26]:


# data.duplicated().sum()


# In[25]:


#


# In[24]:


# data_cl.duplicated().sum()


# In[31]:


data


# In[33]:


data.isna().sum()


# In[35]:


data.isnull().sum() / len(data) * 100


# In[37]:


data = data.dropna()


# In[39]:


data.isna().sum()


# In[41]:


data.isnull().mean()


# In[43]:


data['passenger_count'].sort_values()


# In[45]:


# passenger_count 값의 scatter plot을 그립니다.

sns.scatterplot(x = data.index, y = data['passenger_count'])


# In[48]:


data = data[data['passenger_count'] <= 6]
len(data[data['passenger_count'] == 0])


# In[51]:


data = data[data['passenger_count'] != 0]
sns.scatterplot(x = data.index, y = data['passenger_count'])


# In[53]:


data['trip_distance'].sort_values()


# In[58]:


Q1 = data['trip_distance'].quantile(0.25)
Q3 = data['trip_distance'].quantile(0.75)
IQR = Q3 - Q1


# In[65]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[63]:


lower_bound


# In[66]:


upper_bound


# In[68]:


data = data[(data['trip_distance'] >= lower_bound) & (data['trip_distance'] <= upper_bound)]


# In[70]:


data = {'trip_distance': np.random.lognormal(mean=1.5, sigma=1.0, size=1000)}
data = pd.DataFrame(data)


# In[72]:


sns.set_style('whitegrid')


# In[75]:


plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='trip_distance', bins=50, kde=True)


# In[81]:


data = pd.read_csv('data/trip.csv')


# In[104]:


outlier_count = (data['fare_amount'] <= 0).sum()


# In[85]:


data.describe()


# In[89]:


data = data[data['fare_amount'] > 0]


# In[91]:


data.sort_values('fare_amount')


# In[101]:


sns.scatterplot(x = data.index, y = data['fare_amount'])


# In[97]:


def fare_func(x):
    if x > 150:
        return 150
    else:
        return x
data['fare_amount'].apply(fare_func)


# In[99]:


data['fare_amount'] = data['fare_amount'].apply(lambda x: 150 if x > 150 else x)


# In[102]:


data.sort_values('fare_amount')


# In[105]:


# Q. tip_amount의 scatter plot을 그립니다.

sns.scatterplot(x = data.index, y = data['tip_amount'])


# In[108]:


outlier_count = (data['tip_amount'] > 100).sum()


# In[111]:


data = data[data['tip_amount'] > 100]


# In[113]:


len(data)


# In[116]:


sns.scatterplot(x = data.index, y = data['tolls_amount'])


# In[118]:


data.head(30)


# In[120]:


data['payment_method'].unique()


# In[122]:


data['payment_method'].nunique()


# In[124]:


data['payment_method'] = data['payment_method'].replace(
    ['Debit Card', 'Credit Card'], 'Card')


# In[125]:


data['payment_method'].value_counts()


# In[127]:


example = 'Susan Robinson'


# In[129]:


example.split()


# In[131]:


data['passenger_first_name'] = data['passenger_name'].str.split(' ').str[0]


# In[133]:


data.head()


# In[135]:


data.info()


# In[137]:


data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])


# In[139]:


data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])


# In[141]:


data.info()


# In[143]:


data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])
data['travel_time'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']


# In[145]:


data.head()


# In[147]:


data.info()


# In[149]:


data['travel_time_seconds'] = data['travel_time'].dt.total_seconds()


# In[150]:


data.head()


# In[152]:


data['total_amount'] = data['fare_amount']+data['tip_amount']+data['tolls_amount']


# In[154]:


sns.scatterplot(x=data['trip_distance'], y= data['fare_amount'])


# In[155]:


sns.scatterplot(x =data['travel_time'], y= data['fare_amount'])


# In[157]:


plt.scatter(x= data['trip_distance'], y = data['travel_time'])


# In[158]:


Q1 = data['travel_time'].quantile(0.25)
Q3 = data['travel_time'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_filtered = data[(data['travel_time'] >= lower_bound) & (data['travel_time'] <= upper_bound)]

