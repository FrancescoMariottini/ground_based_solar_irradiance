# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:03:49 2018
Shorted version for simple CSV on 29/8/18

@author: wsfm

2.1.1.	Verification of syntactic and semantic quality 

Table 1 - Example of check of main information and statistical parameters from a pyranometer dataset 

"""

"""IMPORTING MODULES"""
#importing sys to locate modules in different paths
import sys
#poth to python modules
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_IT_R&D/Python_modules/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)
#importing pandas for datetimeindex conversion
import pandas as pd
#importing dataframequality for overview
import pvdata.dataframeanalysis.dataframequality as dfq
#importing pvlibinterface for solar zenith
import pvdata.solarlibraries.pvlibinterface as pvlibint
#importing meteodataquality for solar data
import pvdata.meteodataquality.meteodataquality as meteoq
#import dataframeshow for time visualisation (TEST)
import pvdata.dataframeshow.figureshow as fshow


#importing datetime to check processing time
import datetime as dtt
#importing numpy for mathematical operations
import numpy as np
#importing matplotlib
import matplotlib.pyplot as plt




#import pvdata.solarlibraries.pvlibinterface


"""GLOBAL VARIABLES"""
#path to the dataset
PATH_TO_FOLDER = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/crest_outdoor_measurements_fm/180824-30_fm/"
#files list: pyranometer file only
FILE_NAME = r"crest_outdoor_180824-9_fm.csv"#
FILE_NAME2 =r"Solys2180412-sunlog-20180829095411-all_181008_fm.csv"




#generic formulation to extract more than one file
#FILES_LIST = [f for f in os.listdir(PATH_TO_FOLDER) if (f.endswith(".csv") & f.startswith(FILES_NAME))]
#path to output
PATH_TO_OUTPUT = r"C:/Users/wsfm/Downloads/Python_Test/"

#parameter for pyranometer measurements 
#position CREST (temporary) ground facility, west from Greenwich 
#Note that M is measured eastward from north.
LATITUDE= 52.757774
LONGITUDE = 1.249035


ALTITUDE = 0 
SURFACE_TILT = 29

""" ENABLING DIFFERENT ANALYSIS """
#check modules in main python folder 
daily_start_end = False
irradiance_comparison = True
solar_zenith_comparison = False

#filter_str = "irradiance > 50Wm-2"

filter_str = "none"
filter_str = "d25 irr50 cr30"

#results = "sensors measurements"
results = "relative error"
#results = "absolute error"
#results = "relative error asym"
#results = "temperature"


"""FUNCTIONS"""
#definition of location 
crest_location = pvlibint.SolarLibrary(latitude = LATITUDE, longitude = LONGITUDE, altitude = ALTITUDE)

time = dtt.datetime.now()

if results == "test":
    #test of coordinates 
    print(crest_location.getsolarzenithangle(time))
    crest_location_2 = pvlibint.SolarLibrary(latitude = LATITUDE, longitude = +LONGITUDE, altitude = ALTITUDE)
    print(crest_location.getsolarzenithangle(time))



if results != "roof":
    #test for roof moved in other files after
    df_msrm = pd.read_csv(PATH_TO_FOLDER+FILE_NAME,header=0)
#,nrows=100)
    
        
    #timestamp to filter only clear sky ?? 

    df_msrm["TOA5 TIMESTAMP TS"] = pd.DatetimeIndex(df_msrm["TOA5 TIMESTAMP TS"])  
    #hour introduced
    datetime = pd.DatetimeIndex(df_msrm["TOA5 TIMESTAMP TS"])
    df_msrm["hour"] = datetime.hour

    
elif results == "roof":
    df_msrm_1s = pd.read_csv(PATH_TO_FOLDER+FILE_NAME,header=0)
    datetime = pd.DatetimeIndex(df_msrm_1s["TOA5 TIMESTAMP TS"])
   
    
    

#TEMPORARY MULTIPLE FILTERS SHOULD BE IMPROVED (LR on filtering & uncertainty) !!
#replaced previous filter with direct irradiance (relevant for directional response) higher than 0.0000659
#aim also to remove data out of the day 
#df_msrm = df_msrm[(df_msrm["CHP_Irrd_Avg_mV"]>0.0000659)]


"""
NOT WORKING IRRADIANCE TO BE CHECKED
datetime_series = df_msrm.loc[:,"TOA5 TIMESTAMP TS"].values  
p1 = crest_location.getirradiancediffuseskyinplane(df_msrm["TOA5 TIMESTAMP TS"])
p2 = crest_location.getirradiancediffuseskyinplane_test(df_msrm["TOA5 TIMESTAMP TS"],
     crest_location.getirradiance(datetime_series)["dhi"],
     crest_location.getirradiance(datetime_series)["dni"])                                                  
                                                        
                                                        
db = pd.DataFrame([datetime_series,p1,p2])

db.to_csv(PATH_TO_OUTPUT+"test.csv")
"""



plt.close()




    



if solar_zenith_comparison == True:
    #plot parameters
    DPI = 100   
    #DEV NOTES: frequency probably not necessary 
    FREQUENCY = "h"
    
    df_msrm["time"] = pd.DatetimeIndex(df_msrm["TOA5 TIMESTAMP TS"])
    
    df_msrm["solar_zenith_calculated"] = crest_location.getsolarzenithangle(df_msrm["TOA5 TIMESTAMP TS"]) 
    
    df_msrm2 = pd.read_csv(PATH_TO_FOLDER+FILE_NAME2,header=0)
    
    df_msrm2["time"] = pd.DatetimeIndex(df_msrm2["TOA5 TIMESTAMP TS"])
    
    
    df_msrm = df_msrm[(df_msrm["TOA5 TIMESTAMP TS"]>"25/08/2018  00:00:00") &
                      (df_msrm["TOA5 TIMESTAMP TS"]<"26/08/2018  00:00:00")]
    
    df_msrm2 = df_msrm2[(df_msrm2["TOA5 TIMESTAMP TS"]>"25/08/2018  00:00:00") &
                      (df_msrm2["TOA5 TIMESTAMP TS"]<"26/08/2018  00:00:00")]
    
    df_msrm2=df_msrm2.rename(index=str,columns={"Zenith":"solar_zenith_sun_tracker"})

    #plotting of functions
    p1,=plt.plot(df_msrm["time"],df_msrm["solar_zenith_calculated"],"r.")
    p2,=plt.plot(df_msrm2["time"]+dtt.timedelta(minutes=0),df_msrm2["solar_zenith_sun_tracker"],"g.")
    
    #definition of intervals and extreme for x
    xticks = pd.date_range(start=min(df_msrm2.loc[:,"time"].values),
    end=max(df_msrm.loc[:,"time"].values),freq=FREQUENCY)
    #plotting
    plt.show()

    
    


if daily_start_end == True:
    #analysis of data completeness 
    datetime_series = df_msrm.loc[:,"TOA5 TIMESTAMP TS"].values  
    timerstart = dtt.datetime.now()
    meteoq.SolarData.dailycompleteness(crest_location,datetime_series,days_limit=1,std_coef=3,
                                   dt_format='%d/%m/%Y %I:%M:%p',start_end_graph=False,outliers_graph=True,
                                   path_to_output=PATH_TO_OUTPUT)    
    print("daily completeness: "+str(dtt.datetime.now()-timerstart))





if irradiance_comparison == True: 
    #calculate irradiance through calibration 
    #DEV NOTE 30/8/18: lambda not efficient ?
    chp_sens = 9.29
    df_msrm.loc[:,"CHP_Beam_Avg_Wm-2"] = df_msrm.loc[:,"CHP_Irrd_Avg_mV"].apply(lambda x: 1000*x/chp_sens)
    cmp21_sens = 9.47
    df_msrm.loc[:,"CMP21_Glbl_Avg_Wm-2"] = df_msrm.loc[:,"CMP21_Irrd_Avg_mV"].apply(lambda x: 1000*x/cmp21_sens)
    cmp11a_sens = 9.31
    df_msrm.loc[:,"CMP11a_Shdd_Avg_Wm-2"] = df_msrm.loc[:,"CMP11a_Irrd_Avg_mV"].apply(lambda x: 1000*x/cmp21_sens)
    cmp11b_sens = 8.79
    df_msrm.loc[:,"CMP11b_Tltd_Avg_Wm-2"] = df_msrm.loc[:,"CMP11b_Irrd_Avg_mV"].apply(lambda x: 1000*x/cmp21_sens)
    #calculation of AOI (sun tracking) based on solar zenith angle defined through : 
    #"""[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
    #applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
    #[2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    #solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007."""
    #DEV NOTE 30/08/18: improved formulation faster (0:00:00.183489) than previous lambda formulation (0:00:08.240438)
    #df_msrm.loc[:,"solarzenith"] = df_msrm.loc[:,"TOA5 TIMESTAMP TS"].apply(lambda x:crest_location.getsolarzenithangle(x)[0])
    df_msrm["CMP21_aoi"] = crest_location.getsolarzenithangle(df_msrm["TOA5 TIMESTAMP TS"])  
    df_msrm["CMP11b_aoi"] = df_msrm["CMP21_aoi"] - SURFACE_TILT
    #calculation of values necessary for directional response
    df_msrm["Beam4Horz_Avg_Wm-2"] = np.cos(np.radians(df_msrm.loc[:,"CMP21_aoi"]))*df_msrm.loc[:,"CHP_Beam_Avg_Wm-2"]
    df_msrm["Beam4Tltd&_Avg_Wm-2"] = np.cos(np.radians(df_msrm.loc[:,"CMP11b_aoi"]))*df_msrm.loc[:,"CHP_Beam_Avg_Wm-2"]    
    df_msrm["Diffuse4Horz_Avg_Wm-2"] = df_msrm.loc[:,"CMP11a_Shdd_Avg_Wm-2"]
    #transposition of diffuse according to Perez
    #[3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R.,
    #1990. Modeling daylight availability and irradiance components from
    #direct and global irradiance. Solar Energy 44 (5), 271-289.
    df_msrm["Diffuse4HTltd_Avg_Wm-2"] = df_msrm.loc[:,"CMP11a_Shdd_Avg_Wm-2"]
    #crest_location.getirradiancediffuseskyinplane_test(df_msrm["TOA5 TIMESTAMP TS"],df_msrm.loc[:,"CMP11b_Tltd_Avg_Wm-2"],df_msrm.loc[:,"CHP_Beam_Avg_Wm-2"])
  
    
    df_msrm["CMP21_Beam_deviation_abs"] = (df_msrm.loc[:,"CMP21_Glbl_Avg_Wm-2"]-df_msrm["Diffuse4Horz_Avg_Wm-2"])-df_msrm["Beam4Horz_Avg_Wm-2"]   
    df_msrm["CMP11_Beam_deviation_abs"] = (df_msrm.loc[:,"CMP11b_Tltd_Avg_Wm-2"]-df_msrm["Diffuse4HTltd_Avg_Wm-2"])-df_msrm["Beam4Tltd&_Avg_Wm-2"]
    
    df_msrm["CMP21_Beam_deviation_prc"] = df_msrm["CMP21_Beam_deviation_abs"] / df_msrm["Beam4Horz_Avg_Wm-2"] * 100
    df_msrm["CMP11_Beam_deviation_prc"] = df_msrm["CMP11_Beam_deviation_abs"] / df_msrm["Beam4Tltd&_Avg_Wm-2"] * 100

    
    """
    #check of difference for the entire period#
    df_sum = pd.DataFrame({
    "CMP21_Glbl_Sum_Wm-2":[df_msrm.loc[:,"CMP21_Glbl_Avg_Wm-2"].sum()],
    "CMP11b_Glbl_Sum_Wm-2":[df_msrm.loc[:,"CMP11b_Tltd_Avg_Wm-2"].sum()],                                             
    "Beam4Horz&Diffuse_Sum_Wm-2":[df_msrm.loc[:,"Beam4Horz&Diffuse_Avg_Wm-2"].sum()],                                                
    "Beam4Tltd&Diffuse_Sum_Wm-2":[df_msrm.loc[:,"Beam4Tltd&Diffuse_Avg_Wm-2"].sum()]})                                                 
    """                                                   

    df_msrm = df_msrm.loc[:, ~df_msrm.columns.str.contains('Unnamed')]   
    

    
    import matplotlib.pyplot as plt   
    

    
    

    
    
    
    if "d25" in filter_str:
        df_msrm = df_msrm[(df_msrm["TOA5 TIMESTAMP TS"]>"25/08/2018  00:00:00") &
                          (df_msrm["TOA5 TIMESTAMP TS"]<"26/08/2018  00:00:00")]
    
    if "cr30" in filter_str:
        #definition of cloud ratio
        df_msrm["CR"]=df_msrm["CMP11a_Shdd_Avg_Wm-2"]/df_msrm["CMP21_Glbl_Avg_Wm-2"]    
        #filter on cloud ratio, originally at 0.3
        df_msrm = df_msrm[(df_msrm["CR"]<0.3)]

     
    
    if "irr50" in filter_str:     
        #TEMPORARY FILTER
        #Global higher than 50 W m-2, most strict BSRN checks
        #df_msrm = df_msrm[(df_msrm["CMP21_Glbl_Avg_Wm-2"]>50)]
        title="Directional error calculated assuming the same diffuse irradiance for the horizontal and tilted pyranometers (filter: "+ filter_str +")"
        df_msrm_11 = df_msrm[(df_msrm["Beam4Horz_Avg_Wm-2"]>50)]
        df_msrm_21 = df_msrm[(df_msrm["Beam4Tltd&_Avg_Wm-2"]>50)] 
    

    #close previous graph sessions
    plt.close()

    """ DEFINITION OF POSSIBLE RESULTS """       
    if results == "sensors measurements":        
        df_msrm = df_msrm[(df_msrm["TOA5 TIMESTAMP TS"]>"25/08/2018  00:00:00") &
                          (df_msrm["TOA5 TIMESTAMP TS"]<"26/08/2018  00:00:00")]
        #fshow.plotvstime(df_msrm,"TOA5 TIMESTAMP TS","D") 
        #fshow.plotvstime(df_msrm,"TOA5 TIMESTAMP TS","D",merged_y="Irradiance") 
        DPI = 100 
        FREQUENCY = "h"
        #plotting of functions
        p1,=plt.plot(df_msrm["TOA5 TIMESTAMP TS"],df_msrm["CHP_Beam_Avg_Wm-2"],"y.")
        p2,=plt.plot(df_msrm["TOA5 TIMESTAMP TS"],df_msrm["CMP11a_Shdd_Avg_Wm-2"],"b.")
        p3,=plt.plot(df_msrm["TOA5 TIMESTAMP TS"],df_msrm["CMP21_Glbl_Avg_Wm-2"],"g.")
        p4,=plt.plot(df_msrm["TOA5 TIMESTAMP TS"],df_msrm["CMP11b_Tltd_Avg_Wm-2"],"r.")
        #definition of intervals and extreme for x
        xticks = pd.date_range(start=min(df_msrm.loc[:,"TOA5 TIMESTAMP TS"].values),
        end=max(df_msrm.loc[:,"TOA5 TIMESTAMP TS"].values),freq=FREQUENCY)
        #plotting
        plt.show()
    

  
    
    
    
    if results == "relative error": 
        plt.title(title)
        plt.ylabel('directional response error [%]')
        plt.xlabel('angle of incidence')        
        p1,=plt.plot(df_msrm_11["CMP11b_aoi"],df_msrm_11["CMP11_Beam_deviation_prc"],"b.")
        p2,=plt.plot(df_msrm_21["CMP21_aoi"],df_msrm_21["CMP21_Beam_deviation_prc"],"g.")     
        plt.show()    
        

        
    if results == "relative error asym":
        plt.title(title)
        plt.ylabel('directional response error [%]')
        plt.xlabel('angle of incidence') 
        #split of data between the first and second part of the day
        df_msrm_11a = df_msrm_11[(df_msrm_11["hour"]>0) & (df_msrm_11["hour"]<12)]
        df_msrm_11b = df_msrm_11[(df_msrm_11["hour"]>12) & (df_msrm_11["hour"]<24)]
        df_msrm_21a = df_msrm_21[(df_msrm_21["hour"]>0) & (df_msrm_21["hour"]<12)]
        df_msrm_21b = df_msrm_21[(df_msrm_21["hour"]>12) & (df_msrm_21["hour"]<24)]
        #plot of splitted data  
        p1,=plt.plot(df_msrm_11a["CMP11b_aoi"],df_msrm_11a["CMP11_Beam_deviation_prc"],"y^")
        p2,=plt.plot(df_msrm_21a["CMP21_aoi"],df_msrm_21a["CMP21_Beam_deviation_prc"],"g^")  
        p3,=plt.plot(df_msrm_11b["CMP11b_aoi"],df_msrm_11b["CMP11_Beam_deviation_prc"],"bv")
        p4,=plt.plot(df_msrm_21b["CMP21_aoi"],df_msrm_21b["CMP21_Beam_deviation_prc"],"kv")  
    if results == "absolute error":
        plt.title("Absolute error (filter: Irradiance > 50Wm-2)")
        plt.ylabel('absolute error Wm-2')
        plt.xlabel('angle of incidence')
        p1,=plt.plot(df_msrm_11["CMP11b_aoi"],df_msrm_11["CMP11_Beam_deviation_abs"],"b.")
        p2,=plt.plot(df_msrm_21["CMP21_aoi"],df_msrm_21["CMP21_Beam_deviation_abs"],"g.")    
        plt.show()
    if results == "relative error" or results == "absolute error":
        df_msrm_11.to_csv(PATH_TO_OUTPUT+"measurements11.csv")
        df_msrm_21.to_csv(PATH_TO_OUTPUT+"measurements21.csv")
        #launch preview for 11 & 21 
        preview11 = dfq.Preview.dataframestatistics(df_msrm_11)
        preview21 = dfq.Preview.dataframestatistics(df_msrm_21)      
        #transpose preview
        preview11_t = preview11.T
        preview21_t = preview21.T
        #preview csv
        preview11_t.to_csv(PATH_TO_OUTPUT+"preview11.csv")
        preview21_t.to_csv(PATH_TO_OUTPUT+"preview21.csv")
                
    #output of all measurements temporary removed
    #df_msrm.to_csv(PATH_TO_OUTPUT+"measurements.csv")


    
    if results == "temperature":
        plt.title("Sensors temperature")
        plt.ylabel('temperature of sensors [degree Celsius]')
        plt.xlabel('angle of incidence') 
        #split of data between the first and second part of the day
        df_msrm_mrn = df_msrm[(df_msrm["hour"]>0) & (df_msrm["hour"]<12) & (df_msrm["CMP11b_aoi"]<=90)& (df_msrm["CMP21_aoi"]<=90)]
        df_msrm_aft = df_msrm[(df_msrm["hour"]>12) & (df_msrm["hour"]<24)& (df_msrm["CMP11b_aoi"]<=90)& (df_msrm["CMP21_aoi"]<=90)]

        #plot of splitted data  
        p1,=plt.plot(df_msrm_mrn["CMP21_aoi"],df_msrm_mrn["CHP_Temp_C_Avg_Deg C"],"y^")
        p2,=plt.plot(df_msrm_aft["CMP21_aoi"],df_msrm_aft["CHP_Temp_C_Avg_Deg C"],"bv")
        p3,=plt.plot(df_msrm_mrn["CMP21_aoi"],df_msrm_mrn["CMP21_Temp_C_Avg_Deg C"],"g^")  
        p4,=plt.plot(df_msrm_aft["CMP21_aoi"],df_msrm_aft["CMP21_Temp_C_Avg_Deg C"],"kv")  
       
    






