# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:44:45 2018

@author: Donghyun Kim
"""
import pandas as pd 

def back_testing(i_begin_date,i_end_date,i_stock,i_weight):
    sum_rate=0
    open_arr=[]
    end_arr=[]
    rate_arr=[]
    for i in range(len(i_stock)):
        
        #print('stock/kospi200/'+i_stock[i])
        df_temp = pd.read_csv('stock/kospi200/'+i_stock[i], index_col=None,parse_dates=True, usecols=['date','open','close'], na_values=['nan'])
        begin=df_temp.loc[df_temp['date'] == i_end_date].values[0][1]
        end=df_temp.loc[df_temp['date'] == i_end_date].values[0][2]
        rate=(end/begin)
        rate_arr.append(rate-1)
        open_arr.append(begin)
        end_arr.append(end)
        rate=rate*i_weight[i]
        sum_rate+=rate
        #p_Val = (i_weight)
        #print(" : "+i_stock+" : ","평가손익", " : ", p_Val )
    print("individual return rate:",[round(i,3) for i in rate_arr])
    #print("Open : ",open_arr)
    #print("Close: ",end_arr)
    return sum_rate

#back_testing('2018-05-15','2018-05-16',['003410.csv','001800.csv'],[0.5,0.5])   
        
        