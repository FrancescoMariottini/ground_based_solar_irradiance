# -*- coding: utf-8 -*-
"""
@author: Francesco Mariottini 
Created on 14/8/17
more information on the readme file 
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import for date tick labes
import matplotlib.dates as plt_d

#import axes for limit
#import matplotlib.axes as plt_ax



from pvlib import irradiance as irr
from pvlib import solarposition as sp
from pvlib import atmosphere as atm
from pvlib import clearsky as csky


#reporttemplate module required for the format of exported chart/graph
import fm.reporttemplate.reporttemplate as rtt

#manipulation of text
import textwrap as twrp

#time difference for data completeness
import datetime as dtt



#Additional modules used (depending on the class/function):
#datamanager(@fm): to manipulate dataframe
#datetime: for conversion/extraction of values from datetime values
#math: sin/cos function 
#pvlib(solarpositio,irradiance,spa,tools,atmosphere): to estimate solar position and radiation
#PySolar(solar,radiation,constants): to estimate solar position and radiation
#scipy: for use of statistical function (e.g. linear regression) 


_dt_format='%d/%m/%Y %H:%M'


class DataProviderId:
    """identification/localisation of the measurement device""" 
    # version 1 created on 14/8/17 by fm
    _par_lab = ['lat','lon']
    _par_def = ['latitude','longitude']
    dictionary = dict(zip(_par_lab,_par_def)) 
    #CREST
    #lat = 52.7616
    #lon = 1.2406
    #PV1
    lat=51+27/60+14.4/3600
    lon=3+23/60+14.3/3600
    alt = 79
    srf_tlt = 20
    srf_azm = 0
    #180 if south oriented
    

class Preview:
    """ basic functions to have a first look at a dataframe """
    # version 1 created on 14/8/17 by fm
    def DataframeStatistics(df):
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
                #added on 18/1/18
                preview.loc[index,'nan'] = not_valid
                
        if missing == False: print("\n"+"No 'NaN' or 'null' items found"+"\n")
        #print the overview table before returning it 
        print(preview)
        return preview

    def PlotVsTime(dataframe,columnindex=None,merged_y=None,title=None,save=False,legend_anchor=-0.4):
        """show scatters plot for all variables in a df based on time column (used as x axis)"""
        # Created 28/7/17 by fm
        # Updated 13/2/8 by fm: merge 
        if columnindex == None:
            dataframe.index = dataframe.index
        elif columnindex != None:
            dataframe.index=dataframe[columnindex]
        if merged_y == None:
            for i in dataframe.columns:
                if i != columnindex and "Unnamed" not in i:
                    plt.ylabel(i)
                    dataframe.plot(columnindex,i) 
        elif merged_y != True:
            columns = []
            for i in dataframe.columns:
                if i != columnindex and "Unnamed" not in i:
                    columns.append(i)
                       
            dataframe.plot(x=columnindex,y=columns,title="\n".join(twrp.wrap(title,70)))  
            
            plt.ylabel(merged_y)
            
            #box = ax.get_position()
            #ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])  
            
            plt.legend(loc=8,bbox_to_anchor=(0,legend_anchor,1,legend_anchor),
            ncol=1, mode="expand", borderaxespad=0.)
            
                    
            graph = rtt.Graph(str(title))
            graph.Save()



class Format:   
    #TO BE TRANSFERRED    
    #DEV NOTES: formatting is part of data quality
    # although it could interest also data manager       
    def TimeSeries(series,_format_inp=_dt_format):
        """Return the datetime format of a series or NaT if not identified"""
        #version 1 created on 27/10/17 by fm
        TimeFormats = {'%d/%m/%Y %H:%M:%S','%d/%m/%Y %H:%M','%Y-%m-%d %H:%M:%S','%H:%M:%S','%Y%m%d','%Y-%m-%d'}
        #Acceptable Time Formats (global variables)
        _FormatFound = False
        #initialise control

        for value in series:
        #explore intere series until find a valid record
            if str(pd.to_datetime(value, format=_format_inp, errors='coerce')) != "NaT":
            #test first the suggested default format
                _FormatFound = True
                index = _format_inp
                break
            for index in list(TimeFormats):
            #if suggested format does not work try other formats until valid
                if str(pd.to_datetime(value,format=index,errors='coerce')) != "NaT":
                    _FormatFound = True
                    break
            if _FormatFound == True:
                break
        if _FormatFound == False:
        # if no format valid return "NaT"
            index = "NaT"
        return index          


class Completeness:
    """function related to data completeness (e.g. missing days in records )"""
    #version 1 created on 14/8/17 by fm
    #version 1.1. now df
    #duplicate function not inserted
    
    def DailyStartEnd(datetime_series,solar_library,days_limit=1,StdCoef=2,dt_format='%d/%m/%Y %I:%M:%p',StartEndGraph=False,OutliersGraph=True): 
        """return a dataframe with starting hour and ending hour per day""" 
        """ minute is included as fraction of hour for representation scope """

        # version 1 created on 19/7/17 by fm
        # version 2 created on 30/10/17 by fm, including outlier identification
        # OUTLIERS DEFINITION (v.2)
        # If distance from expected values is higher than standard deviation *3  
        #solar library, more than one year, custom disconnection limit, added on 25/2/18
        formatted = Format.TimeSeries(datetime_series,dt_format)
        #try to format time series    
        
        
        
        if formatted != "NaT":     
            #NOT NECESSARY already there 
            #if format found used for all the series (added 27/10/17)
            #datetime_series =pd.to_datetime(datetime_series, format=formatted) 
                        
            #create a df including dayofyear, hour and date
            dt_full = pd.DatetimeIndex(datetime_series)
            df  = pd.DataFrame({'dayofyear':dt_full.dayofyear, 
            'hm': dt_full.hour+dt_full.minute/60,         
            'date':dt_full.date,
            'date2':dt_full.date,
            'datetime':dt_full
            },index=dt_full)
            
            timeframe = (df["date"].max()-df["date"].min()).days+1
            
  
            #group df per dayofyear and identify min & max hour per day
            df_day=df.groupby(['date'], as_index=True) 
            hm = df_day['hm'] 
            t_min = hm.min()
            t_max = hm.max()
            date = df_day['date2'].min()
            
            
                        
            #include both min & max in a new df and plot the result
            df_compare = pd.DataFrame({'data_on':t_min,'data_off': t_max,'date':date})
            #add dayofyear (previoulsy removed as labelled column by by group)
            dt_date = pd.DatetimeIndex(df_compare['date'])
            df_compare['dayofyear'] =dt_date.dayofyear
            ##StartEndGraph, printed if variable StartEndGraph=True 

            
            
            if StartEndGraph==True:           
                plt.ylabel('hour')
                #plt.xlabel('day of year')
                plt.xlabel('date')
                plt.plot(df_compare.index,t_min,'go',df_compare.index,t_max,'ro')
                plt.legend({'first entry','last entry'},bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                
                title=("Sensor daily first connections and last disconnections based on a "+ 
                str(StdCoef) + "x Standard Deviation limits"+"\n" + "(from "+str(df["date"].min())+
                " to "+str(df["date"].max())+")")
                
                plt.title(title)
                
                df_compare.index=date
                #LOW PRIORITY IMPROVEMENT: exporting the graph
            
            elif OutliersGraph==True:
                #Getting sunrise & sunset from SolarModels class
                
                               
                sunpos = solar_library.GetSunRiseSetTransit(df_compare['date'])
                sunrise = pd.DatetimeIndex(sunpos['sunrise'])
                sunset = pd.DatetimeIndex(sunpos['sunset'])
                #add sunrise & sunset as additional columns to the df
                df_compare=df_compare.assign(sunrise=sunrise.hour+sunrise.minute/60)
                df_compare=df_compare.assign(sunset=sunset.hour+sunset.minute/60)
                #delayed & premature from expected sunrise & sunset
                dif_rise = df_compare["data_on"]- df_compare["sunrise"]
                dif_set = df_compare["sunset"] - df_compare["data_off"]
                #add delays and anticipation to the df
                df_compare=df_compare.assign(dif_rise=dif_rise)
                df_compare=df_compare.assign(dif_set=dif_set)
                #identification of suspect outliers
                on_limit = df_compare["dif_rise"].std() * StdCoef
                off_limit = df_compare["dif_set"].std() * StdCoef
                #disaggregation of suspect outliers
                #indipendent cases: suspect on rise only, suspect on sunset only and suspect on both
                #addition of flag columns with a true value 

                #days-long disconnection start 
                dfdscbig = pd.DataFrame.copy(df_compare,deep=True)
                      
                dfdscbig["dsc_time"] = dfdscbig["dayofyear"].shift(-1) - dfdscbig["dayofyear"]
                dfdscbig =  dfdscbig[(dfdscbig["dsc_time"] > days_limit)]
    
                #days-long disconnection end
                dfrcvbig = pd.DataFrame.copy(df_compare,deep=True)
                dfrcvbig["rcv_nights"] = dfrcvbig["dayofyear"] - dfrcvbig["dayofyear"].shift()  
                dfrcvbig =  dfrcvbig[(dfrcvbig["rcv_nights"] >days_limit)]
                dfrcvbig = dfrcvbig.assign(dfrcvbig_ck = lambda x: True)
                                                                
                #delayed only
                dfdel = pd.DataFrame.copy(df_compare,deep=True)
                #yesterday value for dif_set
                dfdel["dif_set_y"] = dfdel["dif_set"].shift(1) 
                #filter for delayed only
                dfdel =  dfdel[( dfdel["dif_rise"] > on_limit) &
                (dfdel["dif_set"].shift(1) < off_limit) &
                (dfdel["dayofyear"] - dfdel["dayofyear"].shift()==1)]
                dfdel = dfdel.assign(dfdel_ck = lambda x: True)       

                #premature only
                dfprm  = pd.DataFrame.copy(df_compare,deep=True)
                #tomorrow value for dif_rise 
                dfprm["dif_rise_t"] = dfprm ["dif_rise"].shift(-1) 
                #filter for premature only
                dfprm  =  dfprm[( dfprm["dif_set"] > on_limit) &
                (dfprm["dif_rise_t"] < off_limit) &
                (dfprm["dayofyear"].shift(-1) - dfprm["dayofyear"]==1)]
                dfprm = dfprm.assign(dfprm_ck = lambda x: True) 
                   
                #suspected disconnection end 
                dfrcvsml = pd.DataFrame.copy(df_compare,deep=True)
                 #yesterday value for dif_set
                dfrcvsml["dif_set_y"] = dfrcvsml["dif_set"].shift(1)  
                dfrcvsml =  dfrcvsml[( dfrcvsml["dif_rise"] > on_limit) &
                (dfrcvsml["dif_set_y"] > off_limit) &
                (dfrcvsml["dayofyear"] - dfrcvsml["dayofyear"].shift(1)==1)]
                dfrcvsml = dfrcvsml.assign(dfrcvsml_ck = lambda x: True)
                     
                #suspected disconnection start
                dfdscsml = pd.DataFrame.copy(df_compare,deep=True)
                #tomorrow value for dif_rise 
                dfdscsml["dif_rise_t"] = dfdscsml["dif_rise"].shift(-1)  
                dfdscsml =  dfdscsml[( dfdscsml["dif_set"] > on_limit) &
                (dfdscsml["dif_rise_t"] > off_limit)&
                (dfdscsml["dayofyear"].shift(-1) - dfdscsml["dayofyear"]==1)]
                dfdscsml = dfdscsml.assign(dfdscsml_ck = lambda x: True)
                                               
                
                # definition list of key columns
                                
                keys = ('dayofyear','data_off','data_on','date','sunrise','sunset','dif_rise','dif_set')  
                
                                
                #merging of the different datasets into a single one
                if dfdel.empty == False:
                    df_compare= pd.merge(df_compare,dfdel,on=list(keys),how='left')
                elif dfdel.empty == True:
                    df_compare.loc[:,"dfdel_ck"] = False
                    
                if dfprm.empty == False:
                    df_compare= pd.merge(df_compare,dfprm,on=list(keys),how='left')
                elif dfprm.empty == True:
                    df_compare.loc[:,"dfprm_ck"] = False
                       
                if dfrcvbig.empty == False:
                    df_compare= pd.merge(df_compare,dfrcvbig,on=list(keys),how='left')
                elif dfrcvbig.empty == True:
                    df_compare.loc[:,"dfrcvbig_ck"] = False
                    df_compare.loc[:,"dsc_time"] = 1
                
                if dfdscbig.empty == False:
                    df_compare= pd.merge(df_compare,dfdscbig,on=list(keys),how='left')
                elif dfdscbig.empty == True:
                    df_compare.loc[:,"dfdscbig_ck"] = False
                    df_compare.loc[:,"dsc_time"] = 1
                    
                if dfrcvsml.empty == False:
                    df_compare= pd.merge(df_compare,dfrcvsml,on=list(keys),how='left')
                elif dfrcvsml.empty == True:
                    df_compare.loc[:,"dfrcvsml_ck"] = False
                                       
                if dfdscsml.empty == False:
                    df_compare= pd.merge(df_compare,dfdscsml,on=list(keys),how='left')
                elif dfdscsml.empty == True:
                    df_compare.loc[:,"dfdscsml_ck"] = False       

                #TO BE CHECKED: working for CREST not for Garn
                #if "dif_set_y" not in df_compare:
                #    df_compare.loc[:,'dif_set_y'].values = None
                    
                #if "dif_rise_t" not in df_compare:
                #    df_compare.loc[:,'dif_rise_t'].values = None
                    
            
    
                labels = ('day(s)-long disconnections',
                'suspected disconnections',
                'delayed only connections',
                'premature only disconnections',
                'delayed & premature',
                'operation modes')
                
                #calculation of number of cases for the different labels
                #Programmer note: count cannot be applied directly on query or database selection 

                dsclng_n =  df_compare["dsc_time"].sum()-df_compare["dsc_time"].count()

                dscbrv2 = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck == True")

                dscdly2  = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck != True & dfdel_ck == True & dfprm_ck != True")

                dscprm2 = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck != True & dfdel_ck != True & dfprm_ck == True")

                dscdp2 = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck != True & dfdel_ck == True & dfprm_ck == True")

                ndaysok=timeframe-dsclng_n-dscbrv2["date"].count()-dscdly2["date"].count()-dscprm2["date"].count()-dscdp2["date"].count()
                               
             
                fracs = [dsclng_n,
                dscbrv2["date"].count(),
                dscdly2["date"].count(),
                dscprm2["date"].count(),
                dscdp2["date"].count(),
                ndaysok]                       
              
                explode = (0.1, 0.2, 0.1, 0.1, 0.2, 0.1)
                
                
                graph = rtt.PieChart('DailyDelayedAnticipateStdC'+str(StdCoef),labels,fracs,explode)
                graph.Save()
                
                                  
                csv_output_folder = rtt.output_folder
                print(csv_output_folder)
                #creation of csv files if relevant 
                
                
                                                
                if dfdel["dif_rise"].count() != 0:
                    dfdel.to_csv(csv_output_folder+r"\DelayedStdC"+str(StdCoef)+".csv")
                if dfdscsml["dif_rise"].count() != 0:
                    dfdscsml.to_csv(csv_output_folder+r"\SuspectedDisconnectionStartStdC"+str(StdCoef)+".csv")
                if dfdscbig["dif_rise"].count() != 0:
                    dfdscbig.to_csv(csv_output_folder+r"\DaysLongDisconnectionStartStdC"+str(StdCoef)+".csv") 
                if dfprm["dif_set"].count() != 0:    
                    dfprm.to_csv(csv_output_folder+r"PrematureStdC"+str(StdCoef)+".csv") 
                if dfrcvsml["dif_rise"].count() != 0:
                    dfrcvsml.to_csv(csv_output_folder+r"SuspectedDisconnectionEndStdc"+str(StdCoef)+".csv")
                if dfrcvbig["dif_rise"].count() != 0:
                    dfrcvbig.to_csv(csv_output_folder+r"DaysLongDisconnectionEndStdC"+str(StdCoef)+".csv")  
                       
                #add dayofyear
                #dt_date = pd.DatetimeIndex(df_compare['date'])
                #df_compare['dayofyear'] =dt_date.dayofyear
                
                
                #creation of overview csv file 
                df_compare.to_csv(csv_output_folder+r"\DailyOnOffStdC"+str(StdCoef)+".csv")
                
                        
                #creation of a graph
                #FUTURE IMPROVEMENTS: externalisation of graph format in the reporttemplate module?
          
                title=("Analysis of sensor daily first connections and last disconnections based on a "
                + str(StdCoef) + "x Standard Deviation limits"+"\n" + "(from "+str(df["date"].min())
                +" to "+str(df["date"].max())+")")
                
                title="\n".join(twrp.wrap(title,70))

                plt.title(title)
                plt.ylabel('hour')
                plt.xlabel('date')
                plt.xticks(rotation='vertical')
  
                
                #p1,= plt.plot(df_compare['dayofyear'],df_compare['sunrise'],'yo')
                #p2,= plt.plot(df_compare['dayofyear'],df_compare['sunset'],'ko')
                #p3,= plt.plot(dfdel["dayofyear"],dfdel["data_on"],'y2')
                #p4,= plt.plot(dfprm["dayofyear"],dfprm["data_off"],'k1')
                #p5,= plt.plot(dfrcvsml["dayofyear"],dfrcvsml["data_on"],'b^')
                #p6,= plt.plot(dfdscsml["dayofyear"],dfdscsml["data_off"],'mv')
                #p7,= plt.plot(dfrcvbig["dayofyear"],dfrcvbig["data_on"],'gx')
                #p8,= plt.plot(dfdscbig["dayofyear"],dfdscbig["data_off"],'rx')
                
                p1,= plt.plot(df_compare['date'],df_compare['sunrise'],'yo')
                p2,= plt.plot(df_compare['date'],df_compare['sunset'],'ko')
                p3,= plt.plot(dfdel["date"],dfdel["data_on"],'y2')
                p4,= plt.plot(dfprm["date"],dfprm["data_off"],'k1')
                p5,= plt.plot(dfrcvsml["date"],dfrcvsml["data_on"],'b^')
                p6,= plt.plot(dfdscsml["date"],dfdscsml["data_off"],'mv')
                p7,= plt.plot(dfrcvbig["date"],dfrcvbig["data_on"],'gx')
                p8,= plt.plot(dfdscbig["date"],dfdscbig["data_off"],'rx')
                
                subplot = plt.subplot()
                
                #plt.title('Analysis of sensor connections/disconnection based on a ' + str(StdCoef) + 'x Standard Deviation limits')
                #title reported in the html
                    
                plt.legend([p1,p2,p3,p4,p5,p6,p7,p8],
                ["sunrise","sunset","delayed switch-on","premature switch-off",
                "suspected disconnection end","suspected disconnection start",
                "day(s)-long disconnection end","day(s)-long disconnection start"],
                bbox_to_anchor=(0,-1,1,-1), loc=8, mode="expand", ncol=1,
                borderaxespad=0.)
             
                #10/3/18
                
                plt.Axes.set_ylim(subplot,bottom=0,top=24)
                
                
                
                fn = 'DisconnectAnalysisStd'+ str(StdCoef) 
                
                graph = rtt.Graph(fn)
                graph.Save()
                
      
             

        elif formatted == "NaT":
            print("Error: not possible to format the time series")
            df_compare = pd.DataFrame().empty
        return df_compare   



#WORKING VERSION 
"""
def DailyStartEnd2(datetime_series,solar_library,StdCoef=2,hours_limit=12,dt_format='%d/%m/%Y %I:%M:%p',StartEndGraph=False,OutliersGraph=True): 
        # version 1 created on 19/7/17 by fm
        # version 2 created on 30/10/17 by fm, including outlier identification
        # OUTLIERS DEFINITION (v.2)
        # If distance from expected values is higher than standard deviation *3  
        #solar library, more than one year, custom disconnection limit, added on 25/2/18
        formatted = Format.TimeSeries(datetime_series,dt_format)
        #try to format time series    
         
        if formatted != "NaT":     
            #NOT NECESSARY already there 
            #if format found used for all the series (added 27/10/17)
            #datetime_series =pd.to_datetime(datetime_series, format=formatted) 
                        
            #create a df including dayofyear, hour and date
            dt_full = pd.DatetimeIndex(datetime_series)
            df  = pd.DataFrame({'dayofyear':dt_full.dayofyear, 
            'date':dt_full.date,
            'date2':dt_full.date,
            'datetime':dt_full,
            'hour':dt_full.hour,
            },index=dt_full)
            
            #TO BE CHECKED 
            timeframe = (df["date"].max()-df["date"].min()).days+1
            
  
            #group df per dayofyear and identify min & max hour per day
            df_day=df.groupby(['date'], as_index=True) 
            datetime_day = df_day['datetime'] 
            t_min = datetime_day.min()
            t_max = datetime_day.max()
            date = df_day['date2'].min()
            
            
                        
            #include both min & max in a new df and plot the result
            df_compare = pd.DataFrame({'data_on':t_min,'data_off': t_max,'date':date})
            #add dayofyear (previoulsy removed as labelled column by by group)
            dt_date = pd.DatetimeIndex(df_compare['date'])
            df_compare['dayofyear'] =dt_date.dayofyear
            ##StartEndGraph, printed if variable StartEndGraph=True 

            
            
            if StartEndGraph==True:           
                plt.ylabel('hour')
                #plt.xlabel('day of year')
                plt.xlabel('date')
                plt.plot(df_compare.index,t_min,'go',df_compare.index,t_max,'ro')
                plt.legend({'first entry','last entry'},bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                
                title=("Sensor daily first connections and last disconnections based on a "+ 
                str(StdCoef) + "x Standard Deviation limits"+"\n" + "(from "+str(df["date"].min())+
                " to "+str(df["date"].max())+")")
                
                plt.title(title)
                
                df_compare.index=date
                #LOW PRIORITY IMPROVEMENT: exporting the graph
            
            elif OutliersGraph==True:
                #Getting sunrise & sunset from SolarModels class
                
                               
                sunpos = solar_library.GetSunRiseSetTransit(df_compare['date'])
                sunrise = pd.DatetimeIndex(sunpos['sunrise'])
                sunset = pd.DatetimeIndex(sunpos['sunset'])
                
                #add sunrise & sunset as additional columns to the df
                #REPLACED ON 25/2/18 to apply pure datetime
                #df_compare=df_compare.assign(sunrise=sunrise.hour+sunrise.minute/60)
                #df_compare=df_compare.assign(sunset=sunset.hour+sunset.minute/60)
                df_compare=df_compare.assign(sunrise=sunrise)
                df_compare=df_compare.assign(sunset=sunset)
                
                #delayed & premature from expected sunrise & sunset
                dif_rise0 = (pd.DatetimeIndex(df_compare["data_on"])- 
                pd.DatetimeIndex(df_compare["sunrise"]))
                
                dif_set0 = (pd.DatetimeIndex(df_compare["sunset"]) - 
                pd.DatetimeIndex(df_compare["data_off"]))

                
               
                dif_rise = dif_rise0.seconds/3600
                dif_set = dif_set0.seconds/3600
           
                #add delays and anticipation to the df
                df_compare=df_compare.assign(dif_rise=dif_rise)
                df_compare=df_compare.assign(dif_set=dif_set)
                #identification of suspect outliers
                on_limit = df_compare["dif_rise"].std() * StdCoef
                off_limit = df_compare["dif_set"].std() * StdCoef
                #disaggregation of suspect outliers
                #indipendent cases: suspect on rise only, suspect on sunset only and suspect on both
                #addition of flag columns with a true value 

                #days-long disconnection start 
                dfdscbig = pd.DataFrame.copy(df_compare,deep=True)
                      
                dsc_time = (pd.DatetimeIndex(dfdscbig["data_off"].shift(-1))- 
                pd.DatetimeIndex(dfdscbig["data_on"]))
                
                #dfdscbig["dsc_time"] = dfdscbig["dayofyear"].shift(-1) - dfdscbig["dayofyear"]
                
                dfdscbig["dsc_time"] = dsc_time.seconds/3600
                
                
                
                
                dfdscbig =  dfdscbig[(dfdscbig["dsc_time"] >hours_limit)]
    
                #days-long disconnection end
                dfrcvbig = pd.DataFrame.copy(df_compare,deep=True)
                
                rcv_time = (pd.DatetimeIndex(dfrcvbig["data_on"]) 
                - pd.DatetimeIndex(dfrcvbig["data_off"].shift()))    
                
                dfrcvbig["rcv_nights"] = rcv_time.seconds/3600
                
                #dfrcvbig["rcv_nights"] = dfrcvbig["dayofyear"] - dfrcvbig["dayofyear"].shift() 
                
                
                
                dfrcvbig =  dfrcvbig[(dfrcvbig["rcv_nights"] >hours_limit)]
                
                dfrcvbig = dfrcvbig.assign(dfrcvbig_ck = lambda x: True)
                                                                
                #delayed only
                dfdel = pd.DataFrame.copy(df_compare,deep=True)
                #yesterday value for dif_set
                dfdel["dif_set_y"] = dfdel["dif_set"].shift(1) 
                #support 
                dfdel0 = (pd.DatetimeIndex(dfdel["data_off"].shift(-1)) 
                -pd.DatetimeIndex(dfdel["data_on"]))
                dfdel["dfdel_dif"] = dfdel0.seconds/3600 
                #filter for delayed only
                dfdel =  dfdel[( dfdel["dif_rise"] > on_limit) &
                (dfdel["dif_set"].shift(1) < off_limit) &
                (dfdel["dfdel_dif"] <= hours_limit)]
                #(dfdel["dayofyear"] - dfdel["dayofyear"].shift()==1)]
                                
                dfdel = dfdel.assign(dfdel_ck = lambda x: True)       

                #premature only
                dfprm  = pd.DataFrame.copy(df_compare,deep=True)
                #tomorrow value for dif_rise 
                dfprm["dif_rise_t"] = dfprm ["dif_rise"].shift(-1) 
                #support 
                dfprm0 = (pd.DatetimeIndex(dfprm["data_on"]) 
                - pd.DatetimeIndex(dfprm["data_off"].shift()))
                dfprm["dfprm_dif"] = dfprm0.seconds /3600
                
                #filter for premature only
                dfprm  =  dfprm[( dfprm["dif_set"] > on_limit) &
                (dfprm["dif_rise_t"] < off_limit) &
                (dfprm["dfprm_dif"] <= hours_limit)]
                
                #(dfprm["dayofyear"].shift(-1) - dfprm["dayofyear"]==1)]
                                
                dfprm = dfprm.assign(dfprm_ck = lambda x: True) 
                   
                #suspected disconnection end 
                dfrcvsml = pd.DataFrame.copy(df_compare,deep=True)
                 #yesterday value for dif_set
                dfrcvsml["dif_set_y"] = dfrcvsml["dif_set"].shift(1)  
                dfrcvsml =  dfrcvsml[( dfrcvsml["dif_rise"] > on_limit) &
                (dfrcvsml["dif_set_y"] > off_limit) &
                (dfrcvsml["dayofyear"] - dfrcvsml["dayofyear"].shift(1)==1)]
                dfrcvsml = dfrcvsml.assign(dfrcvsml_ck = lambda x: True)
                     
                #suspected disconnection start
                dfdscsml = pd.DataFrame.copy(df_compare,deep=True)
                #tomorrow value for dif_rise 
                dfdscsml["dif_rise_t"] = dfdscsml["dif_rise"].shift(-1)  
                dfdscsml =  dfdscsml[( dfdscsml["dif_set"] > on_limit) &
                (dfdscsml["dif_rise_t"] > off_limit)&
                (dfdscsml["dayofyear"].shift(-1) - dfdscsml["dayofyear"]==1)]
                dfdscsml = dfdscsml.assign(dfdscsml_ck = lambda x: True)
                                               
                
                # definition list of key columns
                                
                keys = ('dayofyear','data_off','data_on','date','sunrise','sunset','dif_rise','dif_set')  
                
                                
                #merging of the different datasets into a single one
                if dfdel.empty == False:
                    df_compare= pd.merge(df_compare,dfdel,on=list(keys),how='left')
                elif dfdel.empty == True:
                    df_compare.loc[:,"dfdel_ck"] = False
                    
                if dfprm.empty == False:
                    df_compare= pd.merge(df_compare,dfprm,on=list(keys),how='left')
                elif dfprm.empty == True:
                    df_compare.loc[:,"dfprm_ck"] = False
                       
                if dfrcvbig.empty == False:
                    df_compare= pd.merge(df_compare,dfrcvbig,on=list(keys),how='left')
                elif dfrcvbig.empty == True:
                    df_compare.loc[:,"dfrcvbig_ck"] = False
                    df_compare.loc[:,"dsc_time"] = 1
                
                if dfdscbig.empty == False:
                    df_compare= pd.merge(df_compare,dfdscbig,on=list(keys),how='left')
                elif dfdscbig.empty == True:
                    df_compare.loc[:,"dfdscbig_ck"] = False
                    df_compare.loc[:,"dsc_time"] = 1
                    
                if dfrcvsml.empty == False:
                    df_compare= pd.merge(df_compare,dfrcvsml,on=list(keys),how='left')
                elif dfrcvsml.empty == True:
                    df_compare.loc[:,"dfrcvsml_ck"] = False
                    
                if dfdscsml.empty == False:
                    df_compare= pd.merge(df_compare,dfdscsml,on=list(keys),how='left')
                elif dfdscsml.empty == True:
                    df_compare.loc[:,"dfdscsml_ck"] = False       

         
    
                labels = 'day(s)-long disconnections','suspected disconnections','delayed only connections','premature only disconnections','delayed & premature', 'operation modes'
                #calculation of number of cases for the different labels
                #Programmer note: count cannot be applied directly on query or database selection 

                dsclng_n =  df_compare["dsc_time"].sum()-df_compare["dsc_time"].count()

                dscbrv2 = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck == True")

                dscdly2  = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck != True & dfdel_ck == True & dfprm_ck != True")

                dscprm2 = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck != True & dfdel_ck != True & dfprm_ck == True")

                dscdp2 = df_compare.query("dfrcvbig_ck != True & dfrcvsml_ck != True & dfdel_ck == True & dfprm_ck == True")

                ndaysok=timeframe-dsclng_n-dscbrv2["date"].count()-dscdly2["date"].count()-dscprm2["date"].count()-dscdp2["date"].count()
                               
             
                fracs = [dsclng_n,
                dscbrv2["date"].count(),
                dscdly2["date"].count(),
                dscprm2["date"].count(),
                dscdp2["date"].count(),
                ndaysok]                       

              
                explode = (0.1, 0.2, 0.1, 0.1, 0.1, 0.1)
                
                
                graph = rtt.PieChart('DailyDelayedAnticipateStdC'+str(StdCoef),labels,fracs,explode)
                graph.Save()
                
                                  
                csv_output_folder = rtt.output_folder
                print(csv_output_folder)
                #creation of csv files if relevant 
                
                
                                                
                if dfdel["dif_rise"].count() != 0:
                    dfdel.to_csv(csv_output_folder+r"\DelayedStdC"+str(StdCoef)+".csv")
                if dfdscsml["dif_rise"].count() != 0:
                    dfdscsml.to_csv(csv_output_folder+r"\SuspectedDisconnectionStartStdC"+str(StdCoef)+".csv")
                if dfdscbig["dif_rise"].count() != 0:
                    dfdscbig.to_csv(csv_output_folder+r"\DaysLongDisconnectionStartStdC"+str(StdCoef)+".csv") 
                if dfprm["dif_set"].count() != 0:    
                    dfprm.to_csv(csv_output_folder+r"PrematureStdC"+str(StdCoef)+".csv") 
                if dfrcvsml["dif_rise"].count() != 0:
                    dfrcvsml.to_csv(csv_output_folder+r"SuspectedDisconnectionEndStdc"+str(StdCoef)+".csv")
                if dfrcvbig["dif_rise"].count() != 0:
                    dfrcvbig.to_csv(csv_output_folder+r"DaysLongDisconnectionEndStdC"+str(StdCoef)+".csv")  
                       
                #add dayofyear
                #dt_date = pd.DatetimeIndex(df_compare['date'])
                #df_compare['dayofyear'] =dt_date.dayofyear
                
                
                #creation of overview csv file 
                df_compare.to_csv(csv_output_folder+r"\DailyOnOffStdC"+str(StdCoef)+".csv")
                
                        
                #creation of a graph
                #FUTURE IMPROVEMENTS: externalisation of graph format in the reporttemplate module?
          
                title=("Analysis of sensor daily fist connections and last disconnections based on a "
                + str(StdCoef) + "x Standard Deviation limits"+"\n" + "(from "+str(df["date"].min())
                +" to "+str(df["date"].max())+")")
                
                title="\n".join(twrp.wrap(title,70))

                plt.title(title)
                plt.ylabel('hour')
                #plt.xlabel('dayofyear')
                plt.xlabel('date')
                
                
                
                #p1,= plt.plot(df_compare['dayofyear'],df_compare['sunrise'],'yo')
                #p2,= plt.plot(df_compare['dayofyear'],df_compare['sunset'],'ko')
                #p3,= plt.plot(dfdel["dayofyear"],dfdel["data_on"],'y2')
                #p4,= plt.plot(dfprm["dayofyear"],dfprm["data_off"],'k1')
                #p5,= plt.plot(dfrcvsml["dayofyear"],dfrcvsml["data_on"],'b^')
                #p6,= plt.plot(dfdscsml["dayofyear"],dfdscsml["data_off"],'mv')
                #p7,= plt.plot(dfrcvbig["dayofyear"],dfrcvbig["data_on"],'gx')
                #p8,= plt.plot(dfdscbig["dayofyear"],dfdscbig["data_off"],'rx')
                
                p1,= plt.plot(df_compare['date'],df_compare['sunrise'],'yo')
                p2,= plt.plot(df_compare['date'],df_compare['sunset'],'ko')
                p3,= plt.plot(dfdel["date"],dfdel["data_on"],'y2')
                p4,= plt.plot(dfprm["date"],dfprm["data_off"],'k1')
                p5,= plt.plot(dfrcvsml["date"],dfrcvsml["data_on"],'b^')
                p6,= plt.plot(dfdscsml["date"],dfdscsml["data_off"],'mv')
                p7,= plt.plot(dfrcvbig["date"],dfrcvbig["data_on"],'gx')
                p8,= plt.plot(dfdscbig["date"],dfdscbig["data_off"],'rx')
                
                #plt.title('Analysis of sensor connections/disconnection based on a ' + str(StdCoef) + 'x Standard Deviation limits')
                #title reported in the html
                    
                plt.legend([p1,p2,p3,p4,p5,p6,p7,p8],
                ["sunrise","sunset","delayed switch-on","premature switch-off",
                "suspected disconnection end","suspected disconnection start",
                "day(s)-long disconnection end","day(s)-long disconnection start"],
                bbox_to_anchor=(0,-0.8,1,-0.8), loc=8, mode="expand", ncol=1,
                borderaxespad=0.)
             
                fn = 'DisconnectAnalysisStd'+ str(StdCoef) 
                
                graph = rtt.Graph(fn)
                graph.Save()

        elif formatted == "NaT":
            print("Error: not possible to format the time series")
            df_compare = pd.DataFrame().empty
        return df_compare   
"""    
    
    

class StatisticalMethods:
        """statistical methods to assess data integrity (e.g. identification of outliers based on linear regression) """      
        #version 1 created on 14/8/17 by fm
        def LinearRegressionOutliers(y_series,x_label='x',y_label='y'):
            #Created 28/7/17 by fm
            #Last Updated 31/7/17 by fm
            #take an expected linear correlation and return possible outliers (i.e. outside correlation coefficient )
            from scipy import stats as ss   
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            #identify parameters of expected regression     
            slope, intercept, r_value, p_value, std_err = ss.linregress(y_series.index,y_series)
            #identify outlier as the ones outside correlation coefficient
            y_outliers = y_series[abs(y_series-(intercept + slope*y_series.index)>r_value)] 
            #print all graph
            plt.plot(y_series.index, y_series, 'go',
            y_series.index, intercept + slope*y_series.index, 'b',
            y_outliers.index, y_outliers, 'ro')
            #ERROR !!: no matter the order, input and linear regression always swapped !! 
            plt.legend({'input '+y_label,'linear regression of '+y_label,'outliers of '+y_label},
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
            return y_outliers




class SolarLibraries:
    output_available = ["AirMassAbsolute","AirMassRelative",
                        "AngleOfIncidence","AngleOfIncidenceProjection",
                        "ExtraTerrestrialRadiation","SolarAzimuthAngle",
                        "SolarZenithAngle","SolarZenithAngleCosin"]
    def __init__(self,package,latitude,longitude,altitude,surface_tilt,surface_azimuth,
        modeldiffuse='perez1990',
        transmittance=0.5,pressure=101325.):
        #initiliase solar libraries
        #version 1 created on 25/1/18 by fm based on previous SolarModels created on 16/8/17
        self.pack = package  
        self.lat = latitude
        self.lon = longitude
        self.alt = altitude
        self.stilt = surface_tilt
        self.sazm = surface_azimuth   
        self.mddff = modeldiffuse
        self.trns = transmittance
        self.prss = pressure
        
        
    def GetAirMassAbsolute(self,datetime,pressure=101325.,model='kastenyoung1989'):
        if self.pack == 'pvlib':
            if model == 'kastenyoung1989':
                amr = self.GetAirMassRelative(datetime,model=model)
                return atm.absoluteairmass(amr, pressure)
    def GetAirMassRelative(self,datetime,model='kastenyoung1989'):
        """Returns airmass_relative : numeric
        Relative airmass at sea level. Will return NaN values for any zenith angle greater than 90 degrees."""
        if self.pack == 'pvlib':
            if model == 'kastenyoung1989':
                sza = self.GetSolarZenithAngle(datetime)
            return atm.relativeairmass(sza,model=model)    
    def GetAngleOfIncidence(self,datetime):
        if self.pack == 'pvlib':
            aoi = irr.aoi(self.stilt,self.sazm,
            self.GetSolarZenithAngle(datetime),
            self.GetSolarAzimuthAngle(datetime))
            #modified to get None for impossible aoi (to be checked for other models)
            return np.minimum(aoi,90)        
    def GetAngleOfIncidenceProjection(self,datetime):
        if self.pack == 'pvlib':
            return irr.aoi_projection(self.stilt,self.sazm,
            self.GetSolarZenithAngle(datetime),
            self.GetSolarAzimuthAngle(datetime))
  
    
    def GetExtraTerrestrialRadiation(self,datetime_series):
        """Determine extraterrestrial radiation from day of year"""
        #Parameters. datetime_or_doy : numeric, array, date, datetime, Timestamp, DatetimeIndex
        if self.pack == 'pvlib':
            #extracting dayofyear to prevent error in pvlib
            dt_full = pd.DatetimeIndex(datetime_series)
            dt_dayofyear = dt_full.dayofyear
            return irr.extraradiation(dt_dayofyear)
        """PY SOLAR DEACTIVATED"""
        #if self.pack == 'pvlib':
            #return rd.GetApparentExtraterrestrialFlux(daytime_series)
    
    
            
    def GetIrradiance(self,datetime_series,model='liujordan'):
        #accuracty of radiance to be established (float)
        #strong dependence on transmittance, better to use GetIrradianceClearSkyHorizontal
     
        """
        #SCOPE TO BE CLARIFIED 
        dt_full= pd.DatetimeIndex(datetime_series)
        df_out=pd.DataFrame({'date':dt_full, 
        'sza': np.float,                   
        'ghi': np.float,
        'dni': np.float,
        'dhi': np.float},
        index = dt_full.date)
        """
        if self.pack == 'pvlib':
            if model == 'liujordan':
                #dni = dni_extra*tao**airmass
                #dhi = 0.3 * (1.0 - tao**airmass) * dni_extra * np.cos(np.radians(zenith))
                #ghi = dhi + dni * np.cos(np.radians(zenith)
                #Returns dataframe with:
                #dni: modeled direct normal irradiance in W/m^2
                #dhi: direct horizontal irradiance in W/m^2
                #gbi: global horizontal irradiance in W/m^2
                                
                """ [1] Campbell, G. S., J. M. Norman (1998) An Introduction to
                Environmental Biophysics. 2nd Ed. New York: Springer.
                [2] Liu, B. Y., R. C. Jordan, (1960). "The interrelationship and
                characteristic distribution of direct, diffuse, and total solar
                radiation".  Solar Energy 4:1-19"""
                solar_parameters = ['SolarZenithAngle','AirMassAbsolute','ExtraTerrestrialRadiation']     
                df_out = self.GetSolarSeries(datetime_series,solar_parameters)
                df_out.loc[:,'transmittance']=self.trns
                df_out.loc[:,'pressure']=self.prss
                #TO BE CHECKED IF WORKS AFTER REMOVED
                #df_out.loc[:,'dni_extra']=self.dnxt
                return irr.liujordan(df_out.loc[:,'SolarZenithAngle'], df_out.loc[:,'transmittance'],
                df_out.loc[:,'AirMassAbsolute'],df_out.loc[:,'pressure'],df_out.loc[:,'ExtraTerrestrialRadiation'])
    
        
    def GetIrradianceClearSkyHorizontal(self,datetime_series,linke_turbidity,model='Ineichen&Perez'):
        if self.pack =='pvlib' and model == 'Ineichen&Perez':
            
            apparent_zenith = self.GetSolarZenithAngleApparent(datetime_series)
            
            airmass_absolute = self.GetAirMassAbsolute(datetime_series)
            
            dni_extra = self.GetExtraTerrestrialRadiation(datetime_series)
            
            #print(type(apparent_zenith))
            
            #print(type( airmass_absolute))
            
            #print(type(dni_extra))
            
            return csky.ineichen(apparent_zenith,airmass_absolute,linke_turbidity,self.alt,dni_extra)                  
            
            """ def ineichen(apparent_zenith, airmass_absolute, linke_turbidity,
            altitude=0, dni_extra=1364.):                               
            INPUT Parameters (5/2/18):
            apparent_zenith : numeric Refraction corrected solar zenith angle in degrees.
            airmass_absolute : numeric
            Pressure corrected airmass.
            linke_turbidity : numeric Linke Turbidity.
            altitude : numeric, default 0: Altitude above sea level in meters.
            dni_extra : numeric, default 1364
            Extraterrestrial irradiance. The units of ``dni_extra determine the units of the output.
            OUTPUT Returns :
            clearsky : DataFrame (if Series input) or OrderedDict of arrays
            DataFrame/OrderedDict contains the columns/keys  ``'dhi', 'dni', 'ghi'``"""

    def GetIrradianceClearSkyInPlane(self,datetime_series,linke_turbidity,model_clearsky='Ineichen&Perez'):
        if self.pack =='pvlib' and model_clearsky == 'Ineichen&Perez':
            #calculating horizontal components   
            dhi_hor = self.GetIrradianceClearSkyHorizontal(datetime_series,linke_turbidity)["dhi"].values
            dni_hor = self.GetIrradianceClearSkyHorizontal(datetime_series,linke_turbidity)["dni"].values
            ghi_hor = self.GetIrradianceClearSkyHorizontal(datetime_series,linke_turbidity)["ghi"].values
            #importing diffuse irradiance from Perez
            amr = self.GetAirMassRelative(datetime_series)
            sza = self.GetSolarZenithAngle(datetime_series)
            saa = self.GetSolarAzimuthAngle(datetime_series)
            dnxt = self.GetExtraTerrestrialRadiation(datetime_series)
            poa_sky_diffuse =  irr.perez(self.stilt,self.sazm,dhi_hor,dni_hor,dnxt,sza,saa,amr,
            model='allsitescomposite1990',return_components=False)
            #importing other parameters for in plane irradiance
            aoi = self.GetAngleOfIncidence(datetime_series)
            poa_ground_diffuse = irr.grounddiffuse(self.stilt,ghi_hor,albedo=0.25,surface_type=None)            
            #calculating in plane irradiance        
            return irr.globalinplane(aoi,dni_hor,poa_sky_diffuse,poa_ground_diffuse)

    def GetIrradianceDiffuseSkyInPlane(self,datetime_series):
        if self.pack == 'pvlib' and self.mddff == 'klucher1979':
            #extracted from pvlib-pyhton function klucher in irradiance module
            #Determine diffuse irradiance from the sky on a tilted surface using Klucher's 1979 model#
            dhi = self.GetIrradiance(datetime_series)["dhi"]
            ghi = self.GetIrradiance(datetime_series)["ghi"]
            sza = self.GetSolarZenithAngle(datetime_series)
            saa = self.GetSolarAzimuthAngle(datetime_series)
            return irr.klucher(self.stilt,self.sazm,dhi,ghi,sza,saa)
            """Klucher's 1979 model determines the diffuse irradiance from the sky
            (ground reflected irradiance is not included in this algorithm) on a
            tilted surface using the surface tilt angle, surface azimuth angle,
            diffuse horizontal irradiance, direct normal irradiance, global
            horizontal irradiance, extraterrestrial irradiance, sun zenith
            angle, and sun azimuth angle.            
            REFERENCES
            ----------
            [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
            solar irradiance on inclined surfaces for building energy simulation"
            2007, Solar Energy vol. 81. pp. 254-267
            [2] Klucher, T.M., 1979. Evaluation of models to predict insolation on
            tilted surfaces. Solar Energy 23 (2), 111-114."""   
        if self.pack == 'pvlib' and self.mddff == 'perez1990':
            #extracted from pvlib-pyhton function perez in irradiance module
            #Determine diffuse irradiance from the sky on a tilted surface using one of the Perez models
            dhi = self.GetIrradiance(datetime_series)["dhi"]
            dni = self.GetIrradiance(datetime_series)["dni"]
            amr = self.GetAirMassRelative(datetime_series)
            sza = self.GetSolarZenithAngle(datetime_series)
            saa = self.GetSolarAzimuthAngle(datetime_series)
            dnxt = self.GetExtraTerrestrialRadiation(datetime_series)
            return irr.perez(self.stilt,self.sazm,dhi,dni, dnxt,sza,saa,amr,
            model='allsitescomposite1990', return_components=False)
        
    
    def GetIrradianceInPlane(self,datetime_series):
        if self.pack == 'pvlib':
            #extracted from pvlib-pyhton function globalinplane in irradiance module
            #determine the three components on in-plane irradiance  
            aoi = self.GetAngleOfIncidence(datetime_series)
            dni = self.GetIrradiance(datetime_series)["dni"]
            poa_sky_diffuse = self.GetIrradianceDiffuseSkyInPlane(datetime_series)
            poa_ground_diffuse = self.GetIrradianceDiffuseGroundInPlane(datetime_series)
            return irr.globalinplane(aoi,dni,poa_sky_diffuse,poa_ground_diffuse)
            """Return poa_global, poa_direct and poa_diffuse    
            Parameters
            aoi : numeric
                Angle of incidence of solar rays with respect to the module
                surface, from :func:`aoi`.
            dni : numeric
                Direct normal irradiance (W/m^2), as measured from a TMY file or
                calculated with a clearsky model.
            poa_sky_diffuse : numeric
                Diffuse irradiance (W/m^2) in the plane of the modules, as
                calculated by a diffuse irradiance translation function
            poa_ground_diffuse : numeric
                Ground reflected irradiance (W/m^2) in the plane of the modules,
                as calculated by an albedo model (eg. :func:`grounddiffuse`)"""
            
                
    
    def GetIrradianceDiffuseGroundInPlane(self,datetime_series):
        #extracted from pvlib-pyhton function grounddiffuse in irradiance module
        #Estimate diffuse irradiance from ground reflections given irradiance, albedo, and surface tilt
        if self.pack == 'pvlib':
            ghi = self.GetIrradiance(datetime_series)["ghi"]
            #albedo originally 0.25 in pvlib. put 0.15 as in Meteonorm
            return irr.grounddiffuse(self.stilt,ghi,albedo=0.25,surface_type=None)
        
    def GetLinkeTurbidity(self,datetime_series,filepath=None,interp_turbidity=True):
        if self.pack == 'pvlib':
            #Look up the Linke Turibidity from the ``LinkeTurbidities.mat``data file supplied with pvlib.
            #6/2/18 UPDATE: need formatting of series into datetimeindex if necessary
            return csky.lookup_linke_turbidity(datetime_series,self.lat,self.lon)
            #turbidity value per month
            """INPUT Parameters
            time : pandas.DatetimeIndex
            latitude : float
            longitude : float
            filepath : None or string, default None   The path to the ``.mat`` file.
            interp_turbidity : bool, default True
            If ``True``, interpolates the monthly Linke turbidity values found in ``LinkeTurbidities.mat`` to daily values.
        
            OUTPUT Returns turbidity : Series"""
            #REVIEWER's NOTES
            #OK 6/2/18 search/upload of mat working: same values if filepath provided or not (l)
        
    
    def GetSolarAzimuthAngle(self,datetime_series,orientation="NREL"):
        """[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
        applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
        [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
        solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007."""
        if self.pack == 'pvlib':
            if orientation == "NREL":
                return sp.spa_python(datetime_series,self.lat,self.lon).loc[:,'azimuth'].values
            
            
    
    def GetSolarZenithAngle(self,datetime_series,cosin=False):
        """[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
        applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
        [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
        solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007."""
        if self.pack == 'pvlib':
            sza = sp.spa_python(datetime_series,self.lat,self.lon).loc[:,'zenith'].values
            if cosin == True:
                sza = np.cos(np.deg2rad(sza)) 
            return sza                
        """PY SOLAR DEACTIVATED"""
        #if self.pack == 'pysolar':
            #if cosin == True:
            #CHECK if works with single data or series
                #return np.cos(np.deg2rad(90 - sl.GetAltitudeFast(self.lat, self.lon, self.datetime)))
    
    def GetSolarZenithAngleApparent(self,datetime_series):
        """[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
        applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
        [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
        solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007."""
        if self.pack == 'pvlib':
            sza = sp.spa_python(datetime_series,self.lat,self.lon).loc[:,'apparent_zenith'].values
            return sza                
   
    
    
    def GetSolarSeries(self,datetime_series,output_series):  
        #return series of solar parameters based on requested output list and existing functions 
        #version 1 created on 14/8/17 by fm, updated on 24/1/18 by fm
        
        dt_full= pd.DatetimeIndex(datetime_series)
        df_out=pd.DataFrame({'datetime':dt_full},index=dt_full) 
        
        if "dayofyear" in list(output_series) or "ExtraTerrestrialRadiation" in list(output_series): 
            df_out["dayofyear"] = dt_full.dayofyear   
        if "AirMassAbsolute" in list(output_series):
            try: df_out["AirMassAbsolute"] = self.GetAirMassAbsolute(datetime_series)
            except Exception as inst: df_out["AirMassAbsolute"] = None
        if "AirMassRelative" in list(output_series):
            try: df_out["AirMassRelative"] = self.GetAirMassAbsolute(datetime_series)
            except Exception as inst: df_out["AirMassRelative"] = None
        if "AngleOfIncidence" in list(output_series):
            try: df_out["AngleOfIncidence"] = self.GetAngleOfIncidence(datetime_series)
            except Exception as inst: df_out["AngleOfIncidence"] = None
        if "AngleOfIncidenceProjection" in list(output_series):
            try: df_out["AngleOfIncidenceProjection"] = self.GetAngleOfIncidenceProjection(datetime_series)
            except Exception as inst: df_out["AngleOfIncidenceProjection"] = None        
        if "ExtraTerrestrialRadiation" in list(output_series):
            try: df_out.loc[:,'ExtraTerrestrialRadiation'] =  self.GetExtraTerrestrialRadiation(df_out.loc[:,'dayofyear'].values)
            except Exception as inst: df_out.loc[:,'ExtraTerrestrialRadiation'] = None
            
        if "IrradianceBeamInPlane" in list(output_series): 
            try: 
                ibip = self.GetIrradiance(datetime_series)
                df_out["IrradianceBeamInPlane"] = ibip["dni"].values
                #df_out["IrradianceBeamInPlane"] = self.GetIrradianceBeamInPlane(datetime_series)
            except Exception as inst: df_out["IrradianceBeamInPlane"] = None
        
        if "SolarAzimuthAngle" in list(output_series): 
            try: df_out["SolarAzimuthAngle"] = self.GetSolarAzimuthAngle(datetime_series)
            except Exception as inst: df_out["SolarAzimuthAngle"] = None
        if "SolarZenithAngle" in list(output_series): 
            try: df_out["SolarZenithAngle"] = self.GetSolarZenithAngle(datetime_series)
            except Exception as inst: df_out["SolarZenithAngle"] = None
        if "SolarZenithAngleCosin" in list(output_series): 
            try: df_out["SolarZenithAngleCosin"] = self.GetSolarZenithAngle(datetime_series,cosin=True)
            except Exception as inst: df_out["SolarZenithAngleCosin"] = None    
        return df_out   

        
    def GetSunRiseSetTransit(self,datetime_series):
        #TO BEC CHECKED
        """ calculate sunrise, sunset & transit according to the proposed package"""
        #version 1 created on 30/10/17 by fm      
        #identify datetime components for the calculation
        dt_full= pd.DatetimeIndex(datetime_series)
        df_out=pd.DataFrame({'date':dt_full.date,        
        'sunrise': np.datetime_data,
        'sunset': np.datetime_data,
        'suntransit': np.datetime_data},
        index = dt_full.date)
        if self.pack == 'pvlib':
            sunpos = sp.get_sun_rise_set_transit(df_out['date'], self.lat, self.lon)
            df_out['sunrise']=pd.DatetimeIndex(sunpos['sunrise'])
            df_out['sunset']=pd.DatetimeIndex(sunpos['sunset'])
            df_out['transit']=pd.DatetimeIndex(sunpos['transit'])
            #Function information (extracted on 30/10/17):
            #Calculate the sunrise, sunset, and sun transit times using the NREL SPA algorithm described in
            #Reda, I., Andreas, A., 2003. Solar position algorithm for solar radiation applications. 
            #Technical report: NREL/TP-560- 34302. Golden, USA, http://www.nrel.gov.
        return df_out
        

                                    
class UserDefinedIntegrity:
    #15/05/18 fm: solar part should be separated    
    """flag data integrity according to a defined checks table"""
    #version 1 created on 26/7/17 by fm
    #version 1.1 updated on 14/8/17, adding auxiliary parameters for analysis
    # primary parameters labels
    _par_lab=('datetime',
    'global_swdn',
    'diffuse_sw',
    'direct_normal_sw',
    'sw_up',
    'lw_dn',
    'lw_up',
    't_a')  
    # primary parameters definition
    _par_def=('reference time (date,hour,minute and second)',
    'Short Wave radiation measured by unshaded pyranometer',
    'Short Wave radiation measured by shaded pyranometer',
    'direct normal component of Short Wave radiation',
    'direct normal Short Wave radiation times multiplied for the cosine of Solar Zenith Angle',
    'downwelling Long Wave radiation measured by a pyrgeometer',
    'upwelling Long Wave radiation measured by a pyrgeometer',
    'air temperature in Kelvin')    
    # auxiliary parameters (required for User Defined Integrity) labels    
    _aux_lab=('cos_sza',
    's_a')
    # auxiliary parameters definition
    _aux_def=('cosine of solar zenith angle(sza)',
    'solar constant adjusted for Earth-Sun distance')
    # definition of quality checks types and description    
    _qcodes_lab = ['l_phy_min',
    'l_rare_min',
    'l_func_min',
    'l_func_max',
    'l_rare_max',
    'l_phy_max']
    _qcodes_def = ['Measurement falls below the physically possible limit',
    'Measurement falls below the extremely rare limit',
    'Measurement falls below the declared function',
    'Measurement exceeds the declared function',
    'Measurement exceeds the extremely rare limit',
    'Measurement exceeds the physically possible limit']    
    #definition of quality levels ID. The higher, the bigger the error
    #odd/even ID numbers are for values which exceed/falls below the limit
    _qcodes_lvl = [4,2,0,1,3,5]
    #definition of external dictionaries    
    par_dict = dict(zip(_par_lab,_par_def))   
    aux_dict = dict(zip(_aux_lab,_aux_def))    
    qcodes = pd.DataFrame({'error':_qcodes_lab,'definition':_qcodes_def},index=_qcodes_lvl)    

    class ChecksTable:
        """initialise checks table based on suggested templates"""
         #version 1 created on 16/8/17 by fm
         #definition of a dictionary of filters
        _ct_lab=('BSRN','CRESTv1')
        _ct_def=('BSRN Global Network recommended QC tests, V2.0','BSRN modified')
        dictionary = dict(zip(_ct_lab,_ct_def))  
        def __init__(self,template):
            self.template = template
            if template == 'BSRN':
                _par_lab=('global_swdn',
                'diffuse_sw',
                'direct_normal_sw',
                'sw_up',
                'lw_dn',
                'lw_up')  
                _qcodes_lvl = [4,2,3,5]
                _global_swdn = ['<-4','<-2','>s_a*1.2*cos_sza**1.2+50','>s_a*1.5*cos_sza**1.2+100']     
                _diffuse_sw = ['<-4','<-2','>s_a*0.75*cos_sza**1.2+30','>s_a*0.95*cos_sza**1.2+50'] 
                _direct_normal_sw = ['<-4','<-2','>s_a*0.95*cos_sza**0.2+10','>s_a'] 
                _sw_up = ['<-4','<-2','>s_a*cos_sza**1.2+50','>s_a*1.2*cos_sza**1.2+50'] 
                _lw_dn = ['<40','<60','>500','>700'] 
                _lw_up = ['<40','<60','>700','>900']
                
                self.dataframe=pd.DataFrame([_global_swdn,
                _diffuse_sw,
                _direct_normal_sw,
                _sw_up,
                _lw_dn,
                _lw_up], 
                index=_par_lab,
                columns=_qcodes_lvl)
                
            if template == 'CRESTv1':
                _par_lab=('global_swdn')  
                _qcodes_lvl = [6,4,2,3,5,7]
                _global_swdn = ['<-4','<-2','<0','>s_a*cos_sza+50','>s_a*1.2*cos_sza**1.2+50','>s_a*1.5*cos_sza**1.2+100']     
                
                self.dataframe=pd.DataFrame([_global_swdn], 
                index=_par_lab,
                columns=_qcodes_lvl)
                
    def AuxiliaryTable(df,lat=DataProviderId.lat,lon=DataProviderId.lon,package='pvlib'):
        #version 1 separeted from FlagTable n 15/12/17 by fm
        ck  = pd.Series()
        for value in list(df.columns):
            if value != "datetime":
                ck[value] = value + "_ck"
                df.loc[:,ck[value]]= 0       
        #define reference timeseries 
        datetime_series = df['datetime']  
        #calculate auxiliary calculated parameters (e.g.sza)    
        output_series = ['dayofyear','"SolarZenithAngleCosin','SolarAzimuthAngle']
        df_aux = SolarLibraries.GetSolarSeries(datetime_series,output_series,lat,lon,package)
        df_aux.rename(index=str, columns={"SolarZenithAngleCosin": "cos_sza", "SolarAzimuthAngle": "s_a"})
        return df_aux
        
                                     
    def FlagTable(df,df_checks,lat=DataProviderId.lat,lon=DataProviderId.lon,package='pvlib'):
        """based on provided df and checks table flag the table""" 
        #version 1 created on 14/8/17 by fm
        #removed checks on datetime on 15/12/17 by fm        
        #perform ALL declared checks based on the modified check table       
        for row in df_checks.index:
            for column in df_checks.columns:
                #iterate the condition for each check type 
                check = df_checks.loc[row,column]
                if check != None:
                    #define & launch the check query                
                   query = row + str(df_checks.loc[row,column])
                   try:
                       df_tmp = df.query(query)
                   except:
                       print('\n' + query)
                   # save the not conform vales, labelling as false
                   if df_tmp.shape[0]!=None:
                        #mark the error for the row
                        ref = df_tmp.index
                        ck_label = row + "_ck"  
                        for index in list(ref):
                            val_old = int(str(df.loc[index,ck_label]),2)
                            val_new = 2**int(column) + val_old
                            df.loc[index,ck_label]=format(val_new,'b')
        return df

