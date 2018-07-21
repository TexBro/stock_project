import sys
import random
import pandas as pd
import numpy as np
import stockUtils as StockUtils
import os
import select_predict_by_tsne
import back_test
DATE=['2018-06-01','2018-06-04','2018-06-05','2018-06-07','2018-06-08','2018-06-11','2018-06-12','2018-06-14','2018-06-15','2018-06-18','2018-06-18','2018-06-19','2018-06-20','2018-06-21','2018-06-22','2018-06-25','2018-06-26']
#DATE=['2018-04-30','2018-05-02','2018-05-03','2018-05-08','2018-05-09','2018-05-10','2018-05-11','2018-05-14','2018-05-15','2018-05-16','2018-05-17','2018-05-18','2018-05-21','2018-05-23','2018-05-24','2018-05-25','2018-05-28','2018-05-29','2018-05-30','2018-05-31']

if (__name__ == "__main__"):
    total_rate2=1
    for i in range(len(DATE)-1):
        B_DATE=DATE[i]
        E_DATE=DATE[i+1]
        print(B_DATE)
        symbols = []
        
        #get 20 tickers which are the most likely to go up
        symbols,sub_symbols=select_predict_by_tsne.get_max_daily_return(B_DATE)
        symbols=symbols[1:6]
    
        print(symbols)
        #get close data [60 days of history + 3 days of predict ]
        adj_close = StockUtils.get_data_with_predict(symbols,B_DATE)
        #adj_close = StockUtils.get_data(symbols)
        
        daily_returns = StockUtils.get_daily_returns(adj_close)
        StockUtils.init_portfolio(len(symbols), daily_returns)
        result = StockUtils.optimize_portfolio()
        
        
        print ('Minimum Variance Portfolio (Red Star){0}'.format(result[1]))
        #print ('Expected Returns: {0} \nExpected Volatility: {1} \nSharpe Ratio: {2}'.format( StockUtils.statistics(result[1])[0], StockUtils.statistics(result[1])[1],StockUtils.statistics(result[1])[2]))
        actual_return=back_test.back_testing(B_DATE,E_DATE,symbols,result[1])
        print("Daily return rate: ",actual_return,'\n')
        total_rate2=total_rate2*(actual_return)
        """
        print ('Minimum Variance Portfolio (Yellow Star){0}'.format(result[0]))
        print ('Expected Returns: {0} \nExpected Volatility: {1} \nSharpe Ratio: {2}'.format( StockUtils.statistics(result[0])[0], StockUtils.statistics(result[0])[1],StockUtils.statistics(result[0])[2]))
        actual_return=back_test.back_testing(B_DATE,E_DATE,symbols,result[0])
        print("Daily return rate: ",actual_return,'\n')
        total_rate2=total_rate2*(actual_return)
        """
    print("Total rate :" ,total_rate2)