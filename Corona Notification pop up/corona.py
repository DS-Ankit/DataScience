#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from plyer import notification
import requests 
from bs4 import BeautifulSoup 
import time

def notifyme(title,message):
    notification.notify(title=title,message=message,app_icon="E:\\python\\images_.ico",timeout=10)
def getdata(url):
    r=requests.get(url)
    return r.text
    
if __name__=="__main__":
    while True:
        data=getdata("https://www.mohfw.gov.in/")
        soup=BeautifulSoup(data,"html.parser")
        mydatas=""
        for tr in soup.find_all("tbody")[0].find_all("tr"):
            mydatas += tr.get_text()
            mydatas = mydatas[1:] 
            itmlist=(mydatas.split("\n"))
            for item in itmlist:
                item.split("\n")
        total=itmlist[201:-14]
        total=list(filter(None, total)) 
        #print(total)
        nTitle="Total Indian Corona cases"
        nText=f"confirmed cases:{total[0]}\n Cured:{total[1]}\n Deaths:{total[2]}"
        notifyme(nTitle,nText)
        time.sleep(3600)


# In[ ]:




