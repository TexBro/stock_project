from time import time
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
#from sklearn import (manifold, datasets, decomposition, ensemble,cluster,
#                     discriminant_analysis, random_projection)

NUMBER_OF_LIST=40

def quotes_historical(symbol):

    data=pd.read_csv(symbol,index_col=None,dtype={'close': np.float64})
    
    data=data.fillna(method='pad')
    data=data['close'].pct_change(1)[1:]
    data=data.values.tolist()

    return data

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    '''
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(X[:,0],X[:,1],'o')


    if title is not None:
        plt.title(title)
    ''' 
def get_max_daily_return(DATE):
    X = []
    X_3days_mean=[]
    filelist=[]
    stocklists=os.listdir('./stock/kospi200')
    first_day_rate=[]
    second_day_rate=[]
    for stocklist in stocklists:
        try:
            selected_file='./stock/result200/'+ stocklist[:-4]+'-'+DATE+'.csv'
            #print('Fetching quote history for %r' % selected_file)
            rate=quotes_historical(selected_file)
            X.append(rate)
            first_day_rate.append(rate[0])
            second_day_rate.append(rate[1])
            filelist.append(stocklist)
        except:
            #print('########################### for %r')
            pass
    '''
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca,
                   "Principal Components projection of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()
    X_tsne=X_pca
    '''
    '''
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    
    #plot_embedding(X_tsne,"t-SNE embedding of the digits (time %.2fs)" %(time() - t0))
    #plt.show()
    
    #from IPython import get_ipython
    #get_ipython().run_line_magic('matplotlib', 'qt')
    
    
    kmeans=cluster.KMeans(n_clusters=4).fit(X_tsne)
    #fig ,ax=plt.subplots()
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans.labels_.astype(float))
    #for i, txt in  enumerate(filelist):
    #    if(i%2==0):
    #        ax.annotate(txt[:-4],(X_tsne[i,0],X_tsne[i,1]))
    #plt.title(DATE+' 5days prediction embedding')
    #plt.show()
    '''
    max_filename=[]
    above_predict=[]
    max_rate=np.sort(first_day_rate)
    for i in range(NUMBER_OF_LIST):
        index=np.where(first_day_rate==max_rate[-i-1])[0][0]
        symbol=filelist[index]
        max_filename.append(symbol)
        df_temp = pd.read_csv('stock/kospi200/'+symbol, index_col='date',parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        df_pred = pd.read_csv('stock/result200/'+symbol[:-4]+'-'+DATE+'.csv',index_col='date',parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        if df_temp.loc[DATE].values[0] <= df_pred.values[0]:
            above_predict.append(symbol)
        
    '''
    sum_X_tsne=X_tsne[:,0]

    chosen_list=[]
    chosen_max_list=[]
    sorted_list=np.sort(sum_X_tsne)
    sorted_max_list=np.sort(X_3days_mean)
    
    for i in range(NUMBER_OF_LIST):
        a= np.where(sum_X_tsne==sorted_list[i])[0][0]
        chosen_list.append(filelist[a])
        ab=np.where(X_3days_mean==sorted_max_list[-i-1])[0][0]
        chosen_max_list.append(filelist[ab])
        
    above_predict=[]
    for symbol in chosen_list:
        df_temp = pd.read_csv('stock/kospi200/'+symbol, index_col='date',parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        df_pred = pd.read_csv('stock/results200/'+symbol[:-4]+'-'+DATE+'.csv',index_col='date',parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        if df_temp.loc[DATE].values[0] <= df_pred.values[0]:
            above_predict.append(symbol)
        
    #print(sorted_list)
    #print(X_tsne[:,1])
    #print(above_predict)
    '''
    return max_filename,above_predict
    
