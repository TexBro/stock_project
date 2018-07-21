import os 
import pandas as pd
import datetime

model_dir='saved_model'
predict_dir='./stock/result200'
history_dir='./stock/kospi200'
evrage=0

def evaluate_prediction(yy,mm,dd):
    ## evaluate only the next day's change is right compare predict and real data of next day
    kospi_list=pd.read_csv('kospi200.csv',sep='\t',engine='python')
    datetime.datetime(yy,mm,dd)
    count=0
    total=len(kospi_list.values)
    
    for ticker in os.listdir(history_dir):#kospi_list.values:
        #ticker=ticker[0][:6]
        ticker=ticker[:-4]
        try:
            y_df=pd.read_csv(history_dir+'/'+str(ticker)+'.csv',index_col=None)
            y_hat_filename=str(ticker)+'-'+datetime.datetime(yy,mm,dd).strftime("%Y-%m-%d")
            y_hat_df=pd.read_csv(predict_dir+'/'+ y_hat_filename +'.csv')
            
            y_index=y_df.loc[y_df['date']==y_hat_df.values.tolist()[0][0]].index[0]
            prev_day=y_index-1
            
            standard=y_df.loc[prev_day].values.tolist()
            y=y_df.loc[y_index].values.tolist()
            y_hat=y_hat_df.values.tolist()[0]
            
            if standard[3]-y[3] > 0 and standard[3]-y_hat[3] > 0:
                count+=1
            elif standard[3]-y[3] < 0 and standard[3]-y_hat[3] < 0:
                count+=1
        except Exception as e:
            #print(e)
            total-=1
            
    print(y_hat_df.values.tolist()[0][0]+" correct predict : ",round((count/total)*100,2),'%')
    return round((count/total)*100,2)

if __name__== '__main__':
    '''
    mm = 5    
    dates=[2,3,8,9,10,14,15,16,17,18,23,24,25,28,29,30]
    '''
    mm=6
    dates=[4,7,8,11,15,18,19,20,21,25,26,27]
    
    
    for dd in dates:
        evrage+=evaluate_prediction(2018,mm,dd)
        
    print(evrage/len(dates))    
        
    
