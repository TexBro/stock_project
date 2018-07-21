#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 00:06:11 2018

@author: dsl
"""

import os
import pandas as pd
import datetime
import numpy as np
import save_kospi

save_kospi.save_kospi(29)

files = os.listdir('./stock/kospi200')
for file in files[1:]:
    kospi=pd.read_csv('./stock/kospi200/'+file,usecols=['date','open','high','low','close','volume','adjclose'])
    date=kospi['date'].values
    after_5days=[]
    for i in range(1,8):
        #last_date=date[-1]
        last_date=datetime.datetime.strptime(date[-1],"%Y-%m-%d")
        predict_date=last_date+datetime.timedelta(days=i)
        if predict_date.weekday() != 5 and predict_date.weekday()!=6:
            after_5days.append(predict_date.strftime("%Y-%m-%d"))
            df1=pd.DataFrame([predict_date.strftime("%Y-%m-%d")],columns=['date'])
            kospi=pd.concat([kospi,df1])
            #kospi.append([predict_date.strftime("%Y-%m-%d")],ignore_index=True)
    #kospi.loc[]
    kospi[['date','open','high','low','close','volume','adjclose']].to_csv('./stock/kospi200/'+file,index=False)