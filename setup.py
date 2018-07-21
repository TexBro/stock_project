#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:20:41 2018
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
@author: dsl
"""

from keras import backend as K
from stock_model import STOCK_model
import os
import gc
import save_kospi

model_dir='saved_model'
predict_dir='./stock/result200'
history_dir='./stock/kospi200'

def setup_floders():
    #floders init
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists('./stock'):
        os.makedirs('./stock')
        
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
def train_kospi200(year,month,dates):
    ##training particural day(year-mm-dd) 
    
    yy=year
    mm=month
    for dd in dates:
        ##get history data from yahoo finance
        save_kospi.save_kospi200_history(history_dir,yy,mm,dd)
        filenames=os.listdir(history_dir)
        
        for i, filename in enumerate(filenames[:]):
            print(i,'/',len(filenames),filename,"is opened")
            model=STOCK_model(filename,history_dir,model_dir,predict_dir)
            try:
                #make model
                predict_model=model.predict_model()     
                #read and make training data
                X_train, y_train =model.read_file()                    
                #training
                model.train_once(predict_model,X_train, y_train)
                #save model
                model.model_save(predict_model)
                #load model
                loaded_model=model.model_load()
                #make test data
                x_test,min_max,date=model.read_file_for_test()
                #predict next 5 days after particural day(year-mm-dd) 
                model.save_predition_as_csv(loaded_model,x_test,min_max,date)
                
                K.clear_session()
                gc.collect()
                
            except Exception as e:
                print(e)
            
if __name__=="__main__":
    year=2018
    
    month=5
    dates=[1,2,3,4,8,9,10,11,14,15,16,17,18,21,23,24,25,28,29,30,31]

    setup_floders()
    train_kospi200(year,month,dates)
