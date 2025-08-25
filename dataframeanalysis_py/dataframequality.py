# -*- coding: utf-8 -*-
"""
@author: Francesco Mariottini 
Created on 14/8/17
more information on the readme file 
"""

""" IMPORTING MAIN MODULES (used in most of the classes)"""
#importing pandas for dataframe
import pandas as pd



#WORKING ON PROGRESS fitting on time 
#from scipy import optimize
"""
import datetime as dtt

time = list()
for i in range (0,9):
    time.append(dtt.datetime(2018,8,2,i,0))
    
y1 = [1,2,3,4,5,6,7,8,9]
y2 = [2,3,4,5,6,7,8,9,10]

df1 = pd.DataFrame({"x":pd.DatetimeIndex(time),"y":y1})
df2 = pd.DataFrame({"x":pd.DatetimeIndex(time),"y":y2})

print(df1,df2)


df1 = pd.DataFrame({"x":pd.DatetimeIndex(time),"y":y1})
df2 = pd.DataFrame({"x":pd.DatetimeIndex(time)-dtt.timedelta(hours=1),"y":y2})

print(df1,df2)


import datetime as dtt

time = list()
for i in range (0,9):
    time.append(dtt.datetime(2018,8,2,i,0))

 
y1 = [1,2,3,4,5,6,7,8,9]
y2 = [2,3,4,5,6,7,8,9,10]



df1 = pd.DataFrame({"x":pd.DatetimeIndex(time),"y":y1})
df2 = pd.DataFrame({"x":pd.DatetimeIndex(time),"y":y2})

 
def xdeviationaverage(df1,df2,ymin,ymax,iteration):
    resolution = (ymax-ymin)/iteration 
    print(iteration)
    for i in range(0,iteration-1):
        centre_tmp=resolution/2+resolution*i
        x1_tmp = df1[(df1.y>=centre_tmp-resolution/2)&(df1.y<centre_tmp+resolution/2)]["x"]
        x1_tmp.

        #print(x1_tmp_mean)
        #x2_tmp_mean = s1[(s1>=centre_tmp-resolution/2)&(s1<centre_tmp-resolution/2)][0].mean()
        #print(x1_tmp_mean)
        #print(x2_tmp_mean)
        
    
xdeviationaverage(df1,df2,1,5,5)
"""



class Preview:
    """ basic functions to have a first look at a dataframe """
    # version 1 created on 14/8/17 by fm
    def dataframestatistics(df):
        """return overview of dataframe including valid entries, startistical parameters, first row as example, ect.""" 
        # version 1 created on 19/7/17 by fm
        # last updated on 3/8/17 by fm
        #IMPROVE(fm): columns are not ordered as declared 
        dfdata = pd.DataFrame()
        dfdata = dfdata.assign(dtypes=df.dtypes)
        dfdata = dfdata.assign(row_1=df.iloc[1])  
        dfstat = df.describe(percentiles=None, include=None, exclude=None).transpose()
        dfstat2 = pd.DataFrame(dfstat,index=dfdata.index)
        #preview = (dfdata, dfstat2) not working properly 
        #ordering column not working
        preview = dfdata.assign(non_nan_or_null=dfstat2["count"], min=dfstat2["min"],
        max=dfstat2["max"],mean=dfstat2["mean"],std=dfstat2["std"],
        p25=dfstat2["25%"],p50=dfstat2["50%"],p75=dfstat2["75%"])
        #calling max & min for 'datetime64[ns]' format    
        for index in preview["dtypes"].index:    
            if preview.loc[index,"dtypes"] == 'datetime64[ns]':          
                preview.loc[index,"min"]=min(df[index])
                preview.loc[index,"max"]=max(df[index])
        #shape & valid items
        print("\n"+"DATA SHAPE: "+ str(df.shape[0]) + "," + str(df.shape[1]))
        missing = False
        for index in preview['non_nan_or_null'].index:
            if preview.loc[index,'non_nan_or_null'] < df.shape[0]:
                not_valid = df.shape[0] - preview.loc[index,'non_nan_or_null']
                print("\n"+ str(not_valid) + " not valid item in column " + index)           
                missing = True
                preview.loc[index,'nan'] = not_valid                
        if missing == False: print("\n"+"No 'NaN' or 'null' items found"+"\n")
        #adding sum positive and
        for index in preview["dtypes"].index:
            #exclude datas from sum 
            if preview.loc[index,"dtypes"] == 'float64':
                preview.loc[index,"sum"]=df[index].sum(skipna=True)
        return preview
