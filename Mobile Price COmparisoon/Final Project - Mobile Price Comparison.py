#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
import bs4,requests
import pandas as pd
import numpy as np


# In[3]:


mobile_name=input('please enter a mobile model: ')
driver = webdriver.Firefox(executable_path='/Users/kinchal/PYTHON/geckodriver')
driver.get('https://www.flipkart.com/');
search_box = driver.find_element_by_name('q')
search_box.send_keys(mobile_name)
search_box.submit()
mobile_url=driver.current_url
data=requests.get(mobile_url)
page_soup=bs4.BeautifulSoup(data.text,'html.parser')
containers = page_soup.findAll('div', {'class': '_1UoZlX'})
l = len(containers)
price = page_soup.findAll('div', {'class': '_1vC4OE _2rQ-NK'})
#f_ratings = page_soup.find('div',{"class": "hGSR34"})
#print(f_ratings.text)

fl_price=[]
for i in range(1):
    pr1=(price[i].text)
    fl_price.append(pr1)
    
fl_price1=[]
for i in fl_price:
    fl_price1.append(i.replace("₹",""))

    


# In[7]:


#mobile_name=input('please enter a mobile model: ')
driver = webdriver.Firefox(executable_path='/Users/kinchal/PYTHON/geckodriver')
driver.get('https://www.amazon.in/');
search_box = driver.find_element_by_name('field-keywords')
search_box.send_keys(mobile_name)
search_box.submit()
mobile_url=driver.current_url
data=requests.get(mobile_url)
page_soup=bs4.BeautifulSoup(data.text,'html.parser')
containers =  driver.find_elements_by_class_name("sg-col-inner")
l = len(containers)
price = driver.find_elements_by_class_name("a-price-whole")
a_ratings =page_soup.find("span",{"class","a-icon-alt"}).text
print(a_ratings)
am_price=[]
for i in range(i+1):
    pr=(price[i].text)
    am_price.append(pr)
Al_price1=[]
for i in am_price:
    Al_price1.append(i.replace("₹|,",""))


# In[8]:


d = {'Model Name': [mobile_name],'Flipkart price':fl_price1, 'Amazon price':am_price}
df = pd.DataFrame(d)
df


# In[9]:


df['Flipkart price'] = df['Flipkart price'].str.replace(',', '')
df['Amazon price'] = df['Amazon price'].str.replace(',', '')
print(df)


# In[10]:


f = df['Flipkart price'].astype(float)
a = df['Amazon price'].astype(float) 
df['Diffrence'] = f - a


# In[11]:


df


# In[12]:


if f.all() > a.all():
    print ('Flipkart is giving the BEST PRICE "\U0001F603"')
else :
    print ('Amazon is giving the BEST PRICE "\U0001F603"')

