#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[122]:


df1 = pd.read_csv("C:/Python/Hackathon/27-08-2019/train_1.csv")
df1.head()


# In[123]:


df1.columns


# In[124]:


plt.hist(df1.fare_amount)


# <b>From the above graph we can infer that above graph is positively skewed.

# In[125]:


# df1.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False, legend=True, 
#         figsize=(15,10))
# plt.show()


# In[126]:


z=pd.DataFrame(df1.isnull().sum())
z


# In[127]:


# Radius of earth = 6371 km


# In[128]:


df1


# In[129]:


df2 = df1.fillna({
                  'dropoff_longitude':int(df1['dropoff_longitude'].mean()),
                  'dropoff_latitude':int(df1['dropoff_latitude'].mean())
                })


# In[130]:


df2.head()


# In[131]:


z1=pd.DataFrame(df2.isnull().sum())
z1


# In[132]:


df2.drop("key", axis = 1, inplace=True)
df2


# In[133]:


def zero(cell):
    if cell==0:
        return 'NaN'
    else:
        return cell

def negative1(cell):
    if float(cell)<=0:
        return 'NaN'
    else:
        return cell    


# In[134]:


df3 = pd.read_csv("C:/Python/Hackathon/27-08-2019/train_1.csv",
                    converters = {'fare_amount':negative1, 
                                  'fare_amount':zero,
                                  'passenger_count':zero})
df3.drop("key", axis = 1, inplace=True)
df3


# In[135]:


# df3 = df3.fillna({
#                   'fare_amount':int(df3['fare_amount'].mean()),
#                   'passenger_count':int(df3['passenger_count'].mean())
#                 })


# In[136]:


df3_describe = df3.describe()

df3 = df2[~(df2.fare_amount<=0) & ~(df2.passenger_count==0) & ~(df2.dropoff_latitude == 0) 
          & ~(df2.dropoff_longitude == 0) & ~(df2.pickup_longitude==0) & ~(df2.pickup_latitude==0)]


# In[137]:


import datetime

lyear = []
lmonth = []
lday = []

df3[df3.pickup_longitude==0]

for i in range(0,df3.shape[0]):
    lyear.append(int(df3.iloc[i,1][0:4]))
    lmonth.append(int(df3.iloc[i,1][5:7]))
    lday.append(int(df3.iloc[i,1][8:10]))


# In[138]:


df3['year']=lyear
df3['month']=lmonth
df3['day']=lday


# In[19]:


plt.scatter(df3.year, df3.fare_amount)


# In[20]:


plt.scatter(df3.month, df3.fare_amount)


# In[21]:


plt.scatter(df3.day, df3.fare_amount)


# In[139]:


from math import radians, cos, sin, asin, sqrt

# dlon = df3.dropoff_longitude - df3.pickup_longitude
# dlat = df3.dropoff_latitude - df3.pickup_latitude

# R = 6371

# for i in range(0,df3.shape[0]):
#     a = (sin(dlat/2))**2 + cos(df3.pickup_latitude)*cos(df3.dropoff_latitude)*(sin(dlon/2))**2
#     c = 2*asin2(sqrt(a), sqrt(1-a)) 
#     d = R*c


# In[140]:


#### ------ Handling longitude and latitude data ----- #######

#Checking latitude == 0
df3 = df3[df3['pickup_longitude'] != 0]
df3 = df3[df3['pickup_latitude'] != 0]
df3 = df3[df3['dropoff_latitude'] != 0]
df3 = df3[df3['dropoff_longitude'] != 0]

# Calculate Distance

dlatitude = df3['dropoff_latitude'] - df3['pickup_latitude']
dlongitude = df3['dropoff_longitude'] - df3['pickup_longitude']

df3['dlatitude'] = dlatitude
df3['dlongitude'] = dlongitude


a = [((sin(dlat/2))**2 + cos(x)*cos(y)*(sin(dlon/2))**2)  for dlat, dlon, x, y 
     in zip(dlatitude, dlongitude, df3['pickup_latitude'], df3['dropoff_latitude'])]
#((sin(-0.009040999999996302/2))**2 + cos(1.707957)*cos(0)*(sin(00.00270100000000184/2))**2) 
df3['a'] = a

df3.isnull().sum()

#df_a.fillna(df_a.median(), inplace = True)
c = [ (2 * asin(sqrt(x))) for x in df3['a']]
df3['c'] = c

radius_of_earth = 6371
df3['distance'] = [(radius_of_earth * x) for x in df3['c']]


# In[141]:


# df3 =df3[~((df3['distance']<200) & (df3['fare_amount']>100))]
# df3 =df3[~((df3['distance']>2500) & (df3['fare_amount']<50))]


# In[142]:


plt.scatter(df3.distance, df3.fare_amount)


# In[143]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[144]:


df3.columns


# In[145]:


x = df3[['passenger_count','distance']]
y = df3['fare_amount']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


# In[146]:


linearR = LinearRegression().fit(x_train, y_train)
pred = linearR.predict(x_test)
mean_squared_error(y_test, pred)


# In[147]:


rmse = np.sqrt(mean_squared_error(np.array(y_test).reshape(-1,1), pred))
rmse


# In[148]:


r2_score(y_test,pred)


# In[149]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_leaf_nodes=10)
DecisionR = model.fit(x_train, y_train)
DecisionR


# In[150]:


from sklearn.metrics import mean_squared_error, r2_score

pred = DecisionR.predict(x_test)
mean_squared_error(y_test, pred)


# In[151]:


r2_score(y_test,pred)


# In[152]:


pred = DecisionR.predict(x_train)
mean_squared_error(y_train, pred)


# <b>For training data

# In[153]:


r2_score(y_train,pred)


# ## Test Data

# In[154]:


df_1 = pd.read_csv("C:/Python/Hackathon/27-08-2019/test.csv")
df_1.head()


# In[155]:


df_1.drop("key", axis = 1, inplace=True)


# In[156]:


df_1.head(2)


# In[157]:


z3 = df_1.isnull().sum()
z3


# In[158]:


df_1 = df_1[df_1['pickup_longitude'] != 0]
df_1 = df_1[df_1['pickup_latitude'] != 0]
df_1 = df_1[df_1['dropoff_latitude'] != 0]
df_1 = df_1[df_1['dropoff_longitude'] != 0]

# Calculate Distance

dlatitude = df_1['dropoff_latitude'] - df_1['pickup_latitude']
dlongitude = df_1['dropoff_longitude'] - df_1['pickup_longitude']

df_1['dlatitude'] = dlatitude
df_1['dlongitude'] = dlongitude


a = [((sin(dlat/2))**2 + cos(x)*cos(y)*(sin(dlon/2))**2)  for dlat, dlon, x, y 
     in zip(dlatitude, dlongitude, df_1['pickup_latitude'], df_1['dropoff_latitude'])]
#((sin(-0.009040999999996302/2))**2 + cos(1.707957)*cos(0)*(sin(00.00270100000000184/2))**2) 
df_1['a'] = a

df_1.isnull().sum()

#df_a.fillna(df_a.median(), inplace = True)
c = [ (2 * asin(sqrt(x))) for x in df_1['a']]
df_1['c'] = c

radius_of_earth = 6371
df_1['distance'] = [(radius_of_earth * x) for x in df_1['c']]


# In[159]:


df_1.head(2)


# In[160]:


#df_2=df_1.drop(["dlatitude", "dlongitude", "a", "c"], axis = 1, inplace=True)


# In[162]:


## Predicting result ##
Xt = df_1[['passenger_count', 'distance']]
yt_pred = model.predict(Xt)
df_final = pd.DataFrame(yt_pred)

with pd.ExcelWriter('Final.xlsx') as writer :
    df_final.to_excel(writer, sheet_name = 'Final')    


# In[ ]:




