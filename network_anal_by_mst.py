# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 23:41:32 2018

@author: Donghyun Kim
"""

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import random
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer;

STOCK_PATH='stock/kospi200_3'
B_date="2011-07-01"
S_date="2011-12-29"

def init_portfolio(num_assets, df_daily_returns):
	global noa
	global daily_returns
	noa = num_assets
	daily_returns = df_daily_returns

def get_data(symbols):
    df = pd.DataFrame()
    
    standard_df=pd.read_csv(STOCK_PATH+'/005930.csv', index_col='date' ,parse_dates=True, usecols=['date', 'adjclose'], na_values=['nan'])
    begin=standard_df.index.get_loc(B_date)
    standard_df=standard_df[begin-120:begin]
    stnadard_date=standard_df.index[0]
    
    for symbol in symbols:
        try:
            df_temp = pd.read_csv(STOCK_PATH+'/'+symbol, index_col='date' ,parse_dates=True, usecols=['date', 'adjclose'], na_values=['nan'])
            #df_temp=df_temp[::-1]
            begin=df_temp.index.get_loc(B_date)
            #anal_begin=df_temp.index.get_loc(H_date)
            df_temp=df_temp[begin-120:begin]
            df_temp = df_temp.rename(columns={'adjclose': symbol})
            if (df_temp.index[0] == stnadard_date):
                df = df.join(df_temp, how='outer')
            else:
                print("missing : ",symbol)
                
        except:
            print("nan : ",symbol)
            pass
    return df

def get_daily_returns(df):
	daily_returns = np.log(df.shift(1) / df)
	daily_returns = daily_returns[1:]
	return daily_returns



if (__name__ == "__main__"):
    
    stock_path=STOCK_PATH
    stocklists=os.listdir(stock_path)

    adj_close = get_data(stocklists)
        
    daily_returns = get_daily_returns(adj_close)

    init_portfolio(daily_returns.shape[1], daily_returns)
    
    corr=daily_returns.corr(method='pearson')
    
    distance= np.sqrt(2*(1-corr))
    
    distnace_matrix=distance.values.tolist()
    
    index_labels=distance.index.values.tolist()
    
    
    
    X=csr_matrix(distnace_matrix)
    tc_org=X.toarray().astype(float)
    Tcsr=minimum_spanning_tree(X)
    MST=Tcsr.toarray().astype(float)
    
    colors = [(random.random(), random.random(), random.random()) for _i in range(noa)]

    G = nx.Graph(MST)
    #G.add_nodes_from(index_labels)
        
    nx.draw(G,node_color=colors ,with_labels=True)
    #plt.show()
    
    np_MST=np.array(MST)
    line_count=[]
    for i in range(noa):
        line=np.count_nonzero(np_MST[i])
        line+=np.count_nonzero(np_MST[:,i])
        line_count.append(line)
    '''   
    add_all=0        
    for a in line_count:
        add_all+=a
    '''
    kmedoids_instance=kmedoids(distnace_matrix,[random.randrange(0,noa) for i in range(10)],data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
    
    
    one_edge_in_each_cluster=[]
    
    for cluster in clusters:
        one_edge=[]
        for i in cluster:
            if line_count[i] == 1:
                one_edge.append(i)
        one_edge_in_each_cluster.append(one_edge)
        
            
    final_10_stocks=[]
    
    for i in one_edge_in_each_cluster:
        final_10_stocks.append(random.choice(i))
    print("final list : ",final_10_stocks)
    
    return_rate=0
    total_buy=0
    total_sell=0
    
    for stock_num in final_10_stocks:
        df_temp = pd.read_csv(STOCK_PATH +'/'+index_labels[stock_num], index_col='date',na_values=['nan'])
        df_temp.fillna(method='ffill')
        buy_price = df_temp['open'][df_temp.index.get_loc(B_date)]
        sell_price=df_temp['close'][df_temp.index.get_loc(S_date)]
        total_buy+=buy_price
        total_sell+=sell_price
    
    print(round(((total_sell-total_buy)/total_sell) *100,2) )