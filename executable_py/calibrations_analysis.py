# -*- coding: utf-8 -*-
"""
Created on 26/7/18
@author: wsfm

Related to calibration, still to be inserted in the thesis 

REF FOLDER: 21120_Calibration of Pyranometers

"""

"""MODULES"""
#importing sys to locate modules in different paths
import sys
#poth to python modules
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_IT_R&D/Python_modules/"
#importing os for basic manipulation
import os
#adding path to the modules
sys.path.append(PATH_TO_MODULES)
#importing pyplot for plotting 
import matplotlib #for font 1/5/20
import matplotlib.pyplot as plt
#importing numpy for mean
import numpy as np
#importing stats as ss for linear regeression
from scipy import stats as stats 
#import datetime for datetime manipulation
import datetime as dt
#importing pandas for dataframe
import pandas as pd

#importing data template for wacom 
import pvdata.datatemplate.crest_wacom as datacrest
#importing pvlibinterface for interface and solar zenith (not used yet for calibration outdoor)
import pvdata.solarlibraries.pvlibinterface as pvlibint
#importing meteodataquality including calibration functions 
import pvdata.meteodataquality.meteodataquality as mdq

#importing module with information on file format
#importing figureshow for visualisation
#import pvdata.dataframeshow.figureshow as figshow
#importing dataframequality for overview
#import pvdata.dataframeanalyis.dataframequality as dfq
#importing pandas for datetimeindex conversion
#import pandas as pd




"""GLOBAL VARIABLES FILES"""
#poth to python modules
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_IT_R&D/Python_modules/"
#path to the input dataset
PATH_TO_INPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/_crest_pyranometer_calibration_indoor/"
PATH_TO_INPUT_FOLDER_CREST_IN = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/_crest_pyranometer_calibration_indoor/"
#PATH_TO_INPUT = r"C:/Users/wsfm/Downloads/input/"
#path to output
PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"

DATETIME_FORMAT_EURAC="%d/%m/%Y %H:%M"
PATH_TO_INPUT_FOLDER_EURAC_2017=( #path to eurac folder
"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/170530-170614_eurac_calibration/")
FILE_EURAC_CHP="calibration_calculations_fm_181119.csv"
FILE_EURAC_CH4="calibration_calculations_Ch4_fm_181122.csv" #kz11_e24_c4
FILE_EURAC_CH5="calibration_calculations_Ch5_fm_181122.csv" #kz11_e20_c5
FILE_EURAC_CH6="calibration_calculations_Ch6_fm_181122.csv" #kz11_e21_c6
FILE_EURAC_CH7="calibration_calculations_Ch7_fm_181122.csv"  #hflpc7
#2018: "cmp21","cmp11_c12","cmp11_c13","vr_ji","cmp11_e46","cmp11_e48"
#2018: "tc5","tco","tc1","tc2","tc3","tc4"

EURAC_RESOLUTION_SECONDS_2017 = 60

PATH_TO_INPUT_FOLDER_EURAC_2018=(
"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/181115-181205_crest_at_eurac_calibration_outdoor/")
FILE_EURAC_irradiance_2018="Irradiation_df_01_15211118.csv"
FILE_EURAC_CH_2018="PyranometerCalib_15211118.csv"
#cmp21_c18_c5_18 in tc5, cmp11_c12_c0_18 in tc0, cmp11_c13_c1_18 in tc1
#cmp11_e46_c3_18 in tc2, cmp11_e48_c4_18 in tc4 (horizontal at abd)
#101850 not available 


EURAC_RESOLUTION_SECONDS_2018 = 5

#DEV NOTE 23/11/18: conditions could be removed if all values are null 
#no wind speed for CH4


COLUMNS = ["vr_ji","vf_ji","datetime","aoi","t_ambient",
         "diffuse","global","wind_speed","azimuth"]


#DEV NOTE 8/1/19: raw values, instead of scaled, necessary if calibration factor accounted
COLUMNS_EURAC_CLB_2017= ["PYR_REF_Ch3 raw","CH raw","date","Zenith angle (ETR) SolPos","T_ambient scaled",
                        "CMP11_diffuse scaled","CMP11_global_horiz scaled","Gill_wind_speed scaled","Azimuth angle SolPos"]

#COLUMNS_EURAC_CLB_CHP= ["CHP1_direct scaled","CMA11_albedo_top raw","date","Zenith angle (ETR) SolPos","T_ambient scaled",
#                        "CMP11_diffuse scaled","CMP11_global_horiz scaled","Gill_wind_speed scaled","Azimuth angle SolPos"]
#COLUMNS_EURAC_CLB_CH4= ["PYR_REF_Ch3 scaled","PYR_TEST_Ch4 raw","date","Zenith angle (ETR) SolPos","T_ambient scaled",
#                        "CMP11_diffuse scaled","CMP11_global_horiz scaled","Gill_wind_speed scaled","Azimuth angle SolPos"]
#COLUMNS_EURAC_CLB_CH5= ["PYR_REF_Ch3 scaled","PYR_TEST_Ch5 raw","date","Zenith angle (ETR) SolPos","T_ambient scaled",
#                        "CMP11_diffuse scaled","CMP11_global_horiz scaled","Gill_wind_speed scaled","Azimuth angle SolPos"]
#COLUMNS_EURAC_CLB_CH6= ["PYR_REF_Ch3 scaled","PYR_TEST_Ch6 raw","date","Zenith angle (ETR) SolPos","T_ambient scaled",
#                        "CMP11_diffuse scaled","CMP11_global_horiz scaled","Gill_wind_speed scaled","Azimuth angle SolPos"]
#COLUMNS_EURAC_CLB_CH7= ["PYR_REF_Ch3 scaled","PYR_TEST_Ch7 raw","date","Zenith angle (ETR) SolPos","T_ambient scaled",
#                        "CMP11_diffuse scaled","CMP11_global_horiz scaled","Gill_wind_speed scaled","Azimuth angle SolPos"]

#DEV NOTE 8/1/19: raw values, instead of scaled, necessary if calibration factor accounted


TEST_ROWS_NUMBER = None
#TEST_ROWS_NUMBER = 10000

COLUMNS_CALIBRATION =["vr_ji","vf_ji","datetime","aoi","t_ambient","diffuse","global","wind_speed","azimuth"]

LOCAL_CONNECTION="postgresql+psycopg2://postgres@localhost/research_sources"
LOCAL_SCHEMA = "eurac"
LOCAL_TABLE= "calibrations"


""" GLOBAL VARIABLES """
ITERATIONS_MAX = 100 #to deal with errors
DEVIATION_MAX = 2 
COVERAGE_FACTOR = 1.96

""" GLOBAL VARIABLES PARAMETERS OUTDOOR EURAC """
CLB_RQR=["series_count","series_minutes","readings_count","beam_min","diffuse_min","diffuse_max","dfraction_max","aoi_max","wind_max"]
RQR_EURAC=[999,20,20,700,10,150,0.15,70,2] #strict requirements while EURAC assigning 0-1 points per requirements based on 9 previous & 10 next measurements (stability) & best series manually selected
RQR_ISO=[999,20,20,20,0,999,0.2,70,2] #stable cloudless conditions
RQR_CS=[999,20,20,20,20,999,0.2,70,2] #35 min beam in E3 report but 20 in IEC 61724-1 2017

#stability parameters
RESOLUTIONS = [60] #,1200]
PERIODS = [1200] #,18000]
COUNTS = [20] #,15]
KCS_DEV_LMT = 0.2
KCS_CV_LMT = 0.02
PEARSON_LMT = 0.8
HOURS_LIMIT = 24 #for dates flagging 


#10 m/s wind speed according to WRR


s1=[1,'31/05/2017 13:11:00','31/05/2017 13:31:00'] #end not counted
s8=[8,'03/06/2017 10:54:00','03/06/2017 11:14:00']
s9=[9,'03/06/2017 11:14:00','03/06/2017 11:34:00']
s10=[10,'03/06/2017 11:34:00','03/06/2017 11:54:00']
s11=[11,'03/06/2017 11:54:00','03/06/2017 12:14:00']
s27=[27,'07/06/2017 11:05:00','07/06/2017 11:25:00']
s28=[28,'07/06/2017 11:25:00','07/06/2017 11:45:00']
s29=[29,'07/06/2017 11:45:00','07/06/2017 12:05:00']
s30=[30,'07/06/2017 12:05:00','07/06/2017 12:25:00']
s32=[32,'07/06/2017 12:45:00','07/06/2017 13:05:00']
s33=[33,'07/06/2017 13:05:00','07/06/2017 13:25:00']
s34=[34,'07/06/2017 13:25:00','07/06/2017 13:45:00']
s35=[35,'07/06/2017 13:45:00','07/06/2017 14:05:00']
s36=[36,'07/06/2017 14:05:00','07/06/2017 14:25:00']
s37=[37,'07/06/2017 14:25:00','07/06/2017 14:45:00']
s60=[60,'10/06/2017 13:16:00','10/06/2017 13:36:00']
s62=[62,'10/06/2017 13:56:00','10/06/2017 14:16:00']
s63=[63,'10/06/2017 14:16:00','10/06/2017 14:36:00']
s64=[64,'10/06/2017 14:36:00','10/06/2017 14:56:00']
s80=[80,'11/06/2017 10:57:00','11/06/2017 11:17:00']
s82=[82,'11/06/2017 11:37:00','11/06/2017 11:57:00']
s83=[83,'11/06/2017 11:57:00','11/06/2017 12:17:00']
s84=[84,'11/06/2017 12:17:00','11/06/2017 12:37:00']
s85=[85,'11/06/2017 12:37:00','11/06/2017 12:57:00']
s86=[86,'11/06/2017 12:57:00','11/06/2017 13:17:00']
s87=[87,'11/06/2017 13:17:00','11/06/2017 13:37:00']
s88=[88,'11/06/2017 13:37:00','11/06/2017 13:57:00']
s89=[89,'11/06/2017 13:57:00','11/06/2017 14:17:00']
s103=[103,'12/06/2017 11:11:00','12/06/2017 11:31:00']
s104=[104,'12/06/2017 11:31:00','12/06/2017 11:51:00']
s109=[109,'13/06/2017 11:12:00','13/06/2017 11:32:00']
s110=[110,'13/06/2017 11:32:00','13/06/2017 11:52:00']

s_eurac_20m=[s1,s8,s9,s10,s11,s27,s28,s29,s30,s32,s33,s34,s35,s36,s37,s60,s62,s63,s64,s80,s82,s83,s84,s85,s86,s87,s88,s89,s103,s104,s109,s110]
s_chp=s_eurac_20m
s_ch4=s_eurac_20m
s_ch5=s_eurac_20m
s_ch6=s_eurac_20m
s_ch7=s_eurac_20m
    
table = []    
for item in s_ch7:
    table.append(item)
    
SERIES_PREDEFINED_DF = pd.DataFrame(table,columns=["id","start","end"])


""" VARIABLES TO BE CHECKED"""
#nnote: ch7 is Husseflux

#EURAC LOCATION: approximation, ABD used instead of outdoor laboratory 
EURAC_LATITUDE = 46.464010
EURAC_LONGITUDE = -11.330354
EURAC_ALTITUDE = 274
#(46° 27’ 28’’ N, 11° 19’ 43’’ E, 247 meters above sea level


EURAC_REFERENCE_SENSITIVITY=8.35/10**6
EURAC_SURFACE_ZENITH = 0
EURAC_SURFACE_AZIMUTH = 180 #even if not horizontal anyway
EURAC_TIMEZONE = "Europe/Rome"
#BEWARE:
# surface_azimuth must be >=0 and <=360. The azimuth convention is dAPPARENT_ZENITH_MODELefined as degrees east of north 
#(e.g. North = 0, South=180 East = 90, West = 270).
#Source: perez(surface_zenith, surface_azimuth, dhi, dni, dni_extra,solar_zenith, solar_azimuth, airmass,model='allsitescomposite1990', return_components=False)          
   


#VARIABLES FOR CREST INDOOR CALIBRATION
STABILISATION_TIME = 60 #theoretical time
MEASUREMENT_TIME = 1


""" GLOBAL VARIABLES PARAMETERS CREST CALIBRATIONS INDOOR """

#generic formulation to extract more than one file
files_list = [f for f in os.listdir(PATH_TO_INPUT) if f.endswith('.txt')]


#excluding calibrations:
# WAC-010-20180719160818-CREST000026-000-RESTRICTED.txt    3 sessions
# WAC-010-20180710120633-CREST000091-000-RESTRICTED.txt    error ?
# WAC-010-20180103154849-20180103010-000-RESTRICTED.txt    error ?
# WAC-010-20171117154318-20171113020-000-RESTRICTED.txt    error ?
# WAC-010-20150729191646-20150729000-000-RESTRICTED.txt    error ?
# WAC-010-20171218141527-20171113020-000-RESTRICTED.txt    -100 to +300 than average
#"Initial test - Digital logg  ed through SmartExplorer" = spikes 
# WAC-010-20171208142548-20171113020-000-RESTRICTED.txt    

#DEV NOTE 7/12/18: to be check why excluded, some have 
files_excluded = ["WAC-010-20180719160818-CREST000026-000-RESTRICTED.txt",
                 "WAC-010-20180710120633-CREST000091-000-RESTRICTED.txt",
                 "WAC-010-20180103154849-20180103010-000-RESTRICTED.txt",
                 "WAC-010-20171117154318-20171113020-000-RESTRICTED.txt",
                 "WAC-010-20150729191646-20150729000-000-RESTRICTED.txt",
                 "WAC-010-20171218141527-20171113020-000-RESTRICTED.txt",
                 "WAC-010-20171208142548-20171113020-000-RESTRICTED.txt",
                 "WAC-010-20150729191646-20150729000-000-RESTRICTED_NEW.txt",
                 "WAC-010-20150729191646-20150729000-000-RESTRICTED_OLD.txt"
                 # initial tr too high ['"Initial', 'test', '-', 'Analogue', '0-1V', 'logging', '-', 'scales', 'to', 'default', 'range', '-200-2000W/m^2"']
                 ]



files_list = list(filter(lambda x: x not in files_excluded,files_list))
#deltaohm bad sensor experimental standard deviation 0.56% 
#files_list =["WAC-010-20180704160532-20180509010-000-RESTRICTED.txt"]

CREST_CALIBRATION_FILES = files_list
#overwriting CREST_CALIBRATION_FILES with only the interested ones




class Setup:
    #assign differente values to variables depending on chosen setup 
    #"STANDARD" SETUP VARIABLES
    #file location variables #100620 test not necessary anymore
    #path_to_input_folder: str
    #path_to_input_file: str
    #dataset format variables 
    #columns_dict: dict
    #location & meteo variables
    #latitude: float
    #longitude: float
    #altitude: float
    #solar_library: pvlibint.SolarLibrary
    #other
    #days: np.ndarray
    #resolution_second: int
    #dataframe: pd.DataFrame
    def __init__(self,source:str):
        #selected colums for all
        self.columns_selected = COLUMNS_CALIBRATION
        self.deviation_max=DEVIATION_MAX
        if "eurac" in source: 
            rqr=RQR_EURAC
            self.latitude = EURAC_LATITUDE
            self.longitude = EURAC_LONGITUDE
            self.altitude = EURAC_ALTITUDE
            self.reference_sensitivity=EURAC_REFERENCE_SENSITIVITY
            self.reference_calibration_factor=1/self.reference_sensitivity
            self.surface_azimuth = EURAC_SURFACE_AZIMUTH 
            self.surface_zenith = EURAC_SURFACE_ZENITH
            self.timezone = EURAC_TIMEZONE
            if "2017" in source: self.resolution = EURAC_RESOLUTION_SECONDS_2017
            elif "2018" in source: self.resolution = EURAC_RESOLUTION_SECONDS_2018 
            #assign as dictionary 
            #initialise solar library for all cases
            #DEV NOTE 20/11/18: "self" makes easier to retrieve variables rather than global variables columns
            #DEV NOTE 20/11/18: however global list make easier iterations and shorten writing
            self.solar_library = pvlibint.SolarLibrary(latitude=self.latitude,longitude=self.longitude,altitude=self.altitude,
                                                       surface_zenith=self.surface_zenith,surface_azimuth=self.surface_azimuth,timezone=self.timezone)
            if "2017" in source: 
                if "CH4" in source: ch_raw="PYR_TEST_Ch4 raw"; file_eurac=FILE_EURAC_CH4; file_suffix="kz11_e24_c4_17" #"CH4"
                elif "CH5" in source: ch_raw="PYR_TEST_Ch5 raw"; file_eurac=FILE_EURAC_CH5; file_suffix="kz11_e20_c5_17"
                elif "CH6" in source: ch_raw="PYR_TEST_Ch6 raw"; file_eurac=FILE_EURAC_CH6; file_suffix="kz11_e21_c6_17"
                elif "CH7" in source: ch_raw="PYR_TEST_Ch7 raw"; file_eurac=FILE_EURAC_CH7; file_suffix="hflpc7_c7_17"
                elif "CHP" in source: ch_raw="CHP1_direct scaled"; file_eurac=FILE_EURAC_CHP; file_suffix="CHP"
            elif "2018" in source:
                file_eurac=FILE_EURAC_CH4; file_suffix="2018"
                if "tc5" in source: file_suffix="cmp21_c18_c5_18"
                elif "tc0" in source: file_suffix="cmp11_c12_c0_18"
                elif "tc1" in source: file_suffix="cmp11_c13_c1_18"
                elif "tc2" in source: file_suffix="cmp11_e46_c3_18"
                elif "tc4" in source: file_suffix="cmp11_e48_c4_18"
            self.filename=file_eurac
            if "eurac filter" in source: file_suffix+="euracfilter"
            elif "iso" in source: rqr=RQR_ISO; file_suffix+="iso"      
            elif "dates flagging" in source: rqr=RQR_CS; file_suffix+="csdates"
            self.file_suffix=file_suffix
            rqr=dict(zip(CLB_RQR,rqr))            
            self.series_count=rqr["series_count"]
            self.series_minutes=rqr["series_minutes"]
            self.readings_count=rqr["readings_count"]
            self.beam_min=rqr["beam_min"]
            self.diffuse_min=rqr["diffuse_min"]
            self.diffuse_max=rqr["diffuse_max"]
            self.dfraction_max=rqr["dfraction_max"]
            self.wind_max=rqr["wind_max"]
            self.aoi_max=rqr["aoi_max"]
        
        if ("test" in source):
            #setup for testing function
            #sample series
            vr_j = [1,2,3]
            vf_j = [2,4,6]
            temperature = [10,10,10]
            datetime = ["2018-11-19 8:41:56","2018-11-19 9:41:56","2018-11-19 10:41:56"]
            #definition of adjusted aoi for the CMP11 (on the solar tracker)  
            diffuse_fraction = [0.1,0.3,0.1]
            aoi = self.solar_library.getsolarzenithangle(datetime)  
            #new parts not checked 
            series_array = [vr_j,vf_j,pd.DatetimeIndex(datetime),aoi,diffuse_fraction,temperature]
            self.dataframe = pd.DataFrame(dict(zip(COLUMNS_CALIBRATION,series_array)))
            self.dataframe["diffuse_fraction"]=diffuse_fraction   
        if ("calibration" in source) and ("eurac" in source) and ("excel" in source):
            if "2017" in source: 
                self.path_to_input_folder=PATH_TO_INPUT_FOLDER_EURAC_2017 #setup for eurac calibration file
                columns_eurac_clb=COLUMNS_EURAC_CLB_2017
                #columns_eurac_clb=[ch_scaled if x=="CH scaled" else x for x in columns_eurac_clb] 
                columns_eurac_clb=[ch_raw if x=="CH raw" else x for x in columns_eurac_clb] #280520 modify name depending on file
            if "2018" in source: 
                self.path_to_input_folder=PATH_TO_INPUT_FOLDER_EURAC_2018 #setup for eurac calibration file

            
            columns_eurac_clb=COLUMNS_EURAC_CLB_2017
            #columns_eurac_clb=[ch_scaled if x=="CH scaled" else x for x in columns_eurac_clb] 
            columns_eurac_clb=[ch_raw if x=="CH raw" else x for x in columns_eurac_clb] #280520 modify name depending on file
            
            columns = dict(zip(COLUMNS,columns_eurac_clb))            
            #selected only requested part of dictionary
            columns_selected = {x: columns[x] for x in COLUMNS_CALIBRATION}   
            columns_selected_list = list(columns_selected.values())
            #import interested columns
            df = pd.read_csv(self.path_to_input_folder+self.filename,
            header=0,skiprows=0,
            usecols=columns_selected_list,
            nrows=TEST_ROWS_NUMBER)                       
            #print(df)
            #inverting dictionary
            columns_selected_inv = {v:k for k,v in columns_selected.items()}         
            #rename columns according to default format     
            df.rename(columns=columns_selected_inv,inplace=True)    
            df["diffuse_fraction"]=df["diffuse"]/df["global"]  
            df["beam"]=df["global"]-df["diffuse"]#  
            df.loc[:,"datetime"]=df.loc[:,"datetime"].apply(lambda x:pd.to_datetime(x,format=DATETIME_FORMAT_EURAC)-np.timedelta64(3600,'s'))#120620 Time is expressed in UTC+2 (UTC+1 plus summertime). 
            #150620 EURAC_RESOLUTION_SECONDS_2017/2 shift removed since mess up data completeness after  
               
            df.loc[:,"datetime"]=pd.DatetimeIndex(df.loc[:,"datetime"]) #timezone already specified in Solarlibrary 

            
            #datetimeindex_tz = pd.DatetimeIndex(df["datetime"],tz=self.timezone,ambiguous='NaT')
            #df2 = df #test
            #df2["datetime"] = datetimeindex_tz.tz_convert(None)           
            
            #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.tz_convert.html
            self.dataframe = df
        if ("calibration" in source) and ("eurac" in source) and ("sql" in source):
            #DEV NOTES 5/11/18: prefilter removed since used later 
            #' sum(a."beam")/sum(a."global_30") as diffuse_fraction,'+
            #' avg(a."diffuse") as diffuse,'+
            if self.series_minutes == 20:
                querystring = (
                'SELECT avg(a."global_30") as global_30,'+
                ' avg(a."beam") as beam,'+
                ' c."Timestamp_m_end" as timestamp,'+
                ' avg(c."TC0") as tcO,'+
                ' avg(c."TC1") as tc1,'+
                ' avg(c."TC2") as tc2,'+
                ' avg(c."TC3") as tc3,'+
                ' avg(c."TC4") as tc4,'+
                ' avg(c."TC5") as tc5,'+
                ' avg(c."RTD0") as rtd0'+
                ' FROM eurac.calibrations as c LEFT JOIN eurac.abd as a'+
                ' ON c."Timestamp_m_end" = a."Timestamp"'+
                ' GROUP BY c."Timestamp_m_end"'+
                ' ORDER BY c."Timestamp_m_end"'
                ) 
            elif self.series_minutes == 5:
                #, a."diffuse_fraction", a."diffuse" 
                querystring = (
            'SELECT c."TC0" as tco,c."TC1" as tc1,c."TC2" as tc2,c."TC3" as tc3,c."TC4" as tc4,c."TC5" as tc5,'+
            'c."RTD0" as rtdo, c."Timestamp" as timestamp,'+
            'a."global_30", a."beam"'+
            ' FROM eurac.calibrations as c LEFT JOIN eurac.abd as a ON c."Timestamp_m_end" = a."Timestamp"')
            if "test" in source:
                querystring = querystring + " LIMIT 1000"
            querystring = querystring + ";"              
            df = pd.read_sql(querystring,LOCAL_CONNECTION)    
            if df.empty== True: print(querystring); print("sql dataframe empty")
                #sys.exit("dataframe empty")         
            #colums 
            COLUMNS_OUTPUT = ["datetime","cmp21_temp","beam","global","diffuse_fraction","diffuse",
            "cmp21","cmp11_c12","cmp11_c13",
            "vr_ji","cmp11_e46","cmp11_e48"]  
            #DEV NOTE 3/12/18: tc0 interpreted as tco in the code 
            COLUMNS_SQL = ["timestamp","rtdo","beam","global_30","diffuse_fraction","diffuse",
            "tc5","tco","tc1",
            "tc2","tc3","tc4"]                           
            #fm hypothesis     
            #"cmp21","cmp11_c12","cmp11_c13",
            #"vr_ji","cmp11_e24","cmp11_e20"]
            #"TC0","TC3","TC5",
            #"TC4","TC1","TC2"]
            #print(df.head(n=5))
            columns_selected=dict(zip(COLUMNS_SQL,COLUMNS_OUTPUT))            
            df.rename(columns=columns_selected    
            ,inplace=True) 
            df["aoi"]=df["datetime"].apply(lambda x: self.solar_library.getsolarzenithangle(x)[0])
            df["azimuth"]=df["datetime"].apply(lambda x: self.solar_library.getsolarazimuthangle(x)[0])
            df["wind_speed"]=np.empty(len(df["azimuth"]))
            df["t_ambient"]=np.empty(len(df["azimuth"]))          
            #test columns 
            #print(df.columns)          
            self.dataframe = df   
  

""" PROCESSING """

def run(process,test=True,path_to_output=PATH_TO_OUTPUT,markersize=None):
    setup = Setup(source=process)
    
    
    if "outdoor" and "eurac" in process: #"indoor" not in process:
        #VARIABLE FOR OUTDOOR CALIBRATION (EURAC)

        #5/12/18: remove filter on cloud ratio & diffuse since not properly calculated
        #consider using a filter indipendent from other measurements rather than global irradiance

        #crest 21/11/18 unstable sky conditions (clouds with distance > 30 degree)
        #DEV NOTE 2/12/18 issue when 20  minutes 
        
        #series_index = [1,15,2,14,3,13,4,12,5,11,6,10,7,9,8]
        a = dt.datetime.now()
        setup = Setup(source=process)   
        if ("dates flagging" in process):
           method="isoa_astm_cmp21" #also for class C since ref pyr is a class A
           if "uncertainty" in process:            
               Irradiance_Uncertainty = mdq.Irradiance_Uncertainty(sensitivity=setup.reference_sensitivity,method=method)
               setup.file_suffix+="uncertainty"
           elif "uncertainty" not in process:
               Irradiance_Uncertainty = None
        
        
        df=setup.dataframe 
        calibration = mdq.SolarData.Calibration(
        location=setup.solar_library, method="9847 1a",calibration_factor=setup.reference_calibration_factor)
        print("setup end: "+str(dt.datetime.now()-a))
        #setting options for visualisation 
        pd.set_option('display.max_columns',20)
        pd.set_option('display.max_rows',100)
        #setup
        cmp_id = "cmp21"
        cmp_id = "cmp11_c12"
        cmp_id = "cmp11_c13"
        cmp_id = "cmp11_e46"
        cmp_id = "cmp11_e48"  
        
        
            
        #DEV NOTE 26/11/18: rough approximation for cloud ratio & diffuse already in database
        #DEV NOTE 23/11/18: df.query could be used to have more dynamic filters ?
        #DEV NOTE 28/11/18: etr should be calculated for each point  
        #df.loc[:,"deviation"] = (df.loc[:,"beam"].shift(-1)/df.loc[:,"global"].shift(-1)-df.loc[:,"beam"]/df.loc[:,"global"])
        #df.loc[:,"next_measurement"] = df.loc[:,"datetime"].shift(-1)-df.loc[:,"datetime"]
        #df.loc[:,"next_measurement"] = df.loc[:,"next_measurement"].apply(lambda x: x.seconds/60)
        #df.loc[:,"deviation_minute"] = df.loc[:,"deviation"]/df.loc[:,"next_measurement"]  
        #df.to_csv(PATH_TO_OUTPUT+"test.csv")    
        #print(df[df["deviation_minute"]>0.75])
        df_old = df
        #insert null value for missing columns
        #df["diffuse_fraction"]=np.full(len(df),0)
        df["wind_speed"]=np.full(len(df),0)
        
 
        
        #DEV NOTE 5/12/18: introduce basic filter only on global for PV system purpose
        
        if df.empty== True: df_old.to_csv(path_to_output+"df_unfiltered.csv");print("filtered database empty")
            #sys.exit("filtered dataframe empty")
        #elif df.empty == False: print("dateframe ok: not empty
    
    """#3/6/20 currently not used 
    if ("calibration eurac sql" in process):
    
        #dateindex necessary
        #28/11/18: beam condition not inserted yet 
        filtered_series = pd.DataFrame(
        {"datetime":df.loc[:,"datetime"],
         "beam":df.loc[:,"beam"]},
         index=df.index)
        
        #df[["datetime","beam"]]
        filtered_series.loc[:,"end"] =   filtered_series.loc[:,"datetime"].apply(lambda x: x  + np.timedelta64(setup.series_minutes,'m'))    
        filtered_series.loc[:,"start"] =  filtered_series.loc[:,"datetime"].apply(lambda x:x)            
        def n_readings(dataframe,start,end):
            n=len(dataframe[(dataframe["datetime"]>=start)&(dataframe["datetime"]<=end)])
            return n            
        #conditions on variation could also be imposed see ancillary data 
        filtered_series.loc[:,"readings"] = filtered_series.apply(lambda x: n_readings(filtered_series,x["start"],x["end"]),axis=1)     
        filtered_series.to_csv(path_to_output+"all_series_"+cmp_id+".csv")   
        filtered_series=filtered_series[filtered_series["readings"]>=setup.readings_count]    
        if filtered_series.empty== True:
            sys.exit("filtered series not enough readings")        
        end_tmp = min(filtered_series.loc[:,"start"])
        series_start=[]
        series_end=[]
        series_n = []
        series_id = []
        for index in list(filtered_series.index):
            #equal or bigger to start, and series can also share one extreme
            if filtered_series.loc[index,"start"] >= end_tmp:
                series_start.append(filtered_series.loc[index,"start"])
                series_end.append(filtered_series.loc[index,"end"])
                series_n.append(filtered_series.loc[index,"readings"])
                series_id.append(filtered_series.loc[index][0])
                end_tmp = filtered_series.loc[index,"end"]
        #creating df with information about the potential series
        #overloaded, for some parameters (aoi )
        indipendent_series = pd.DataFrame({
        "id_s":series_id,
        "start":series_start,
        "end":series_end,
        "n":series_n,
        "aoi_median":np.full(len(series_n),None), 
        "azimuth_median":np.full(len(series_n),None), 
        "t_sensor_min":np.full(len(series_n),None),    
        "t_sensor_10":np.full(len(series_n),None),
        "t_sensor_median":np.full(len(series_n),None),
        "t_sensor_90":np.full(len(series_n),None),
        "t_sensor_max":np.full(len(series_n),None),     
        "global_min":np.full(len(series_n),None),    
        "global_10":np.full(len(series_n),None),
        "global_median":np.full(len(series_n),None),
        "global_90":np.full(len(series_n),None),
        "global_max":np.full(len(series_n),None),     
        "wind_speed_min":np.full(len(series_n),None),    
        "wind_speed_10":np.full(len(series_n),None),
        "wind_speed_median":np.full(len(series_n),None),
        "wind_speed_90":np.full(len(series_n),None),
        "wind_speed_max":np.full(len(series_n),None),
        "diffuse_fraction_min":np.full(len(series_n),None),    
        "diffuse_fraction_10":np.full(len(series_n),None),
        "diffuse_fraction_median":np.full(len(series_n),None),
        "diffuse_fraction_90":np.full(len(series_n),None),
        "diffuse_fraction_max":np.full(len(series_n),None),
        "diffuse_fraction_weighted":np.full(len(series_n),None)},
        index=series_id)
        if indipendent_series.empty== True:
            sys.exit("indipendent_series empty")
        elif indipendent_series.empty == False:
            print("indipendent_series ok")
            #print(indipendent_series.iloc[:,:])      
        for index in list(indipendent_series.index):
        #filling df with information about series 
            s_start_tmp=indipendent_series.loc[index,"start"]
            s_end_tmp=indipendent_series.loc[index,"end"]
            df_tmp = df[(df["datetime"]>=s_start_tmp)&(df["datetime"]<=s_end_tmp)]
            #DEV NOTE 8/1/19: could be iterated based on columns name but first ones
            indipendent_series.loc[index,"aoi_median"]=df_tmp["aoi"].median()
            indipendent_series.loc[index,"azimuth_median"]=df_tmp["azimuth"].median()
            indipendent_series.loc[index,"t_ambient_median"]=df_tmp["t_ambient"].median()
            indipendent_series.loc[index,"global50"]=df_tmp["vr_ji"].median()
            indipendent_series.loc[index,"wind_speed_median"]=df_tmp["wind_speed"].median()
            indipendent_series.loc[index,"t_ambient_90"]=np.percentile(df_tmp["t_ambient"],90)
            indipendent_series.loc[index,"global_90"]=np.percentile(df_tmp["vr_ji"],90)
            indipendent_series.loc[index,"wind_speed_90"]=np.percentile(df_tmp["wind_speed"],90)
            indipendent_series.loc[index,"t_ambient_10"]=np.percentile(df_tmp["t_ambient"],10)
            indipendent_series.loc[index,"global_10"]=np.percentile(df_tmp["vr_ji"],10)
            indipendent_series.loc[index,"wind_speed_10"]=np.percentile(df_tmp["wind_speed"],10)
            indipendent_series.loc[index,"t_ambient_min"]=df_tmp["t_ambient"].min()
            indipendent_series.loc[index,"global_min"]=df_tmp["vr_ji"].min()
            indipendent_series.loc[index,"wind_speed_min"]=df_tmp["wind_speed"].min()
            indipendent_series.loc[index,"t_ambient_max"]=df_tmp["t_ambient"].max()
            indipendent_series.loc[index,"global_max"]=df_tmp["vr_ji"].max()
            indipendent_series.loc[index,"wind_speed_max"]=df_tmp["wind_speed"].max()
            
            # Temporarily removed since no cloud ratio
            #indipendent_series.loc[index,"diffuse_fraction_min"]=df_tmp["diffuse_fraction"].min()
            #indipendent_series.loc[index,"diffuse_fraction_10"]=np.percentile(df_tmp["diffuse_fraction"],10)
            #indipendent_series.loc[index,"diffuse_fraction_median"]=df_tmp["diffuse_fraction"].median()
            #indipendent_series.loc[index,"diffuse_fraction_90"]=np.percentile(df_tmp["diffuse_fraction"],90)
            #indipendent_series.loc[index,"diffuse_fraction_max"]=df_tmp["diffuse_fraction"].max()  
            #indipendent_series.loc[index,"diffuse_fraction_weighted"]=df_tmp["diffuse"].sum()/df_tmp["global"].sum()
    
            
            
        indipendent_series.to_csv(path_to_output+"indipendent_series_"+cmp_id+".csv")   
        
    
        
        df_series = indipendent_series
        #defining calibration parameters    
        calibration = mdq.SolarData.Calibration(
        location=setup.solar_library, method="9847 1a",reference_calibration_f=setup.reference_calibration_factor)
        df=setup.dataframe 
        # start results table
        calibration_series = []
        calibration_measurements = pd.DataFrame(columns=["id_s","vr_ji","vf_ji","datetime","f_ji","f_dev"])  
        a = dt.datetime.now()
        for i in list(df_series.index.values):
        #calculation of calibration factor per series
            #filter per series
            df_s_tmp = df[(df["datetime"]>=df_series.loc[i,"start"])&(df["datetime"]<=df_series.loc[i,"end"])]        
            #skip if empty
            if df_s_tmp.empty == False:
                vr_j=df_s_tmp["vr_ji"]
    
            
        
                vf_j=df_s_tmp[cmp_id]
                
                datetime=df_s_tmp["datetime"]          
                #notify running series 
                #print("series from "+str(df_series.loc[i,"start"])+" to "+str(df_series.loc[i,"end"]))  
                #get dataframe and main parameters for calculation 
                valid,f_j,f_j_std,len_j,results_j_df=calibration.get_factor_df(#to be changed with df2
                reference_series=vr_j, 
                field_series=vf_j,
                time_series=datetime,
                deviation_max=setup.deviation_max,
                readings_count=setup.readings_count,
                iterations_max=ITERATIONS_MAX 
                )
                if valid == True:
                    #create array with series id
                    results_j_df.assign(id_s=np.full(len(results_j_df),df_series.loc[i,"id_s"]))
                    #concatenate series df with previous one
                    calibration_measurements = pd.concat([calibration_measurements,results_j_df],sort=False)
                    #define main results for series     
                    results_j=[
                    df_series.loc[i,"id_s"],
                    df_series.loc[i,"start"],
                    df_series.loc[i,"end"],            
                    df_series.loc[i,"aoi_median"],
                    df_series.loc[i,"t_ambient_median"],
                    df_series.loc[i,"global_median"],
                    df_series.loc[i,"wind_speed_median"],
                    df_series.loc[i,"diffuse_fraction_median"],
                    df_series.loc[i,"diffuse_fraction_weighted"],
                    df_series.loc[i,"azimuth_median"],
                    valid,
                    f_j,
                    f_j_std,
                    len_j]
                    #append results
                    calibration_series.append(results_j)
        #transform summary into dataframe
        calibration_series_df = pd.DataFrame(calibration_series,columns=
        ["id_s","start","end",
         "aoi_median","t_ambient_median","global_median","wind_speed_median","diffuse_fraction_median","diffuse_fraction_weighted","azimuth_median", 
         "valid","f","std","records"])
        calibration_series_df.index=calibration_series_df["id_s"]
        if calibration_series_df.empty== True:
            print("calibration_series_df")
        elif calibration_series_df.empty == False:
            print("calibration_series_df ok")
            #print(calibration_series_df.iloc[0,:])
        if calibration_measurements.empty== True:
            print("calibration_measurements_df")
        elif calibration_measurements.empty == False:
            print("calibration_measurements_df ok")
            #print(calibration_measurements.iloc[0,:])
        print("calibration calculation end: "+str(dt.datetime.now()-a))
        #export both summary and singles series measurements  
        calibration_series_df[calibration_series_df["valid"]==True].to_csv(path_to_output+"calibration_series_"+setup.file_suffix+cmp_id+".csv")
        calibration_measurements.to_csv(path_to_output+"calibration_measurements_"+setup.file_suffix+cmp_id+".csv")
        #selection of valid series
        calibration_series_df=calibration_series_df[calibration_series_df["valid"]==True]
        series_index=[]
        for i in range(0,int(setup.series_count/2)):
            series_index.append(i+1)
            series_index.append(setup.series_count-i)
        series_index_mean = int(np.average(series_index))
        series_index.append(series_index_mean)
        aoi_east_min = calibration_series_df[(calibration_series_df["azimuth_median"]<=azimuth_south)]["aoi_median"].min()
        aoi_west_min = calibration_series_df[(calibration_series_df["azimuth_median"]>=azimuth_south)]["aoi_median"].min() 
        step_east = (aoi_max-aoi_east_min)/(int(setup.series_count/2))
        step_west = (aoi_max-aoi_west_min)/(int(setup.series_count/2))
        step_deviation_east =  step_east/2
        step_deviation_west =  step_west/2
        calibration_measurements_limited=pd.DataFrame(columns=
        ["id_s","start","end",
        "aoi_median","t_ambient_median","global_median","wind_speed_median","diffuse_fraction_median","diffuse_fraction_weighted","azimuth_median", 
        "valid","f","std","records","index","aoi_target"]) 
        index_tmp = series_index
        calibration_series_df_tmp = calibration_series_df
        for iteration in range(0,ITERATIONS_MAX-1):   
        #impose a limit to the ammissible iterations (not in standard, just to avoid infinite loops)
            for index_s in list(index_tmp):
                #DEV NOTE 22/11/18: aoi_target may be defined separately for different purpose
                if index_s == series_index_mean:
                    aoi_target = 0
                    series_candidates = calibration_series_df_tmp[
                    (calibration_series_df_tmp["aoi_median"]==calibration_series_df_tmp["aoi_median"].min())]      
                elif (index_s>=1) and (index_s<series_index_mean):
                    aoi_target = aoi_max-step_east*(index_s-1)
                    series_candidates = calibration_series_df_tmp[
                    (calibration_series_df_tmp["aoi_median"]>=aoi_target-step_deviation_east)&
                    (calibration_series_df_tmp["aoi_median"]<=aoi_target+step_deviation_east)&
                    (calibration_series_df_tmp["azimuth_median"]<=azimuth_south)]
                elif (index_s>series_index_mean) and (index_s<=15):
                    aoi_target = aoi_max-step_west*(15-index_s)
                    series_candidates = calibration_series_df_tmp[
                    (calibration_series_df_tmp["aoi_median"]>=aoi_target-step_deviation_west)&
                    (calibration_series_df_tmp["aoi_median"]<=aoi_target+step_deviation_west)&
                    (calibration_series_df_tmp["azimuth_median"]>=azimuth_south)]
                if series_candidates.empty == False:   
                    #if series 
                    series_candidates=series_candidates.assign(aoi_median_dev=
                    series_candidates.loc[:,"aoi_median"].apply(lambda x:abs(x-aoi_target)))                                       
                    series_chosen = series_candidates[(series_candidates["aoi_median_dev"]==series_candidates.loc[:,"aoi_median_dev"].min())]
                    series_chosen = series_chosen.assign(aoi_target=
                    np.full(len(series_chosen),aoi_target))                                   
                    series_chosen = series_chosen.assign(index=
                    np.full(len(series_chosen),index_s))
                    index_tmp.remove(index_s)
                    #index_tmp.remove(index_s)  # for list
                    calibration_series_df_tmp=calibration_series_df_tmp.drop(index=series_chosen.loc[:,"id_s"])                
                    calibration_measurements_limited = calibration_measurements_limited.append(series_chosen,sort='False')
            if not index_tmp: 
                break
            elif index_tmp:
                #DEV NOTE 22/11/18: not perfect since higher aoi target may be assigned to lower aoi target if closer
                step_deviation_east =  step_deviation_east*(1+step_deviation_increase)
                step_deviation_west =  step_deviation_west*(1+step_deviation_increase)
        if  calibration_measurements_limited.empty== True:
            print("calibration_measurements_limited empty")
        elif  calibration_measurements_limited.empty == False:
            print("calibration_measurements_limited ok")
            #print(calibration_measurements_limited.iloc[0,:])
        calibration_measurements_limited.to_csv(path_to_output+"calibration_measurements_limited_"+cmp_id+".csv") 
    """
   
    
    if ("series finding" in process):         #extraction of valid series 
        #pd.set_option('display.max_columns',30)
        #pd.set_option('display.max_rows',30)          
        #DEV NOTE 23/11/18: df.query could be used to have more dynamic filters ?
        df = df[(df["beam"]>=setup.beam_min)&
                (df["diffuse"]>setup.diffuse_min)&(df["diffuse"]<setup.diffuse_max)&
                (df["diffuse_fraction"]<setup.dfraction_max)&
                (df["aoi"]<=setup.aoi_max)] #4/5/20 no wind speed since not measured
        if df.empty== True: print ("filtered dataframe empty"); #sys.exit(error_message)
 
        
        if ("stability flagging" in process):
            df2, dfsout = mdq.SolarData.stabilityflagging(SolarLibrary=setup.solar_library,
                          datetimeindex_utc=df.loc[:,"datetime"].values,
                          irradiance_values=df.loc[:,"global"].values,
                          resolutions=RESOLUTIONS,
                          periods=PERIODS,
                          counts=COUNTS,
                          irr_min=setup.beam_min,
                          aoi_max=setup.aoi_max,
                          kcs_range=[1-KCS_DEV_LMT,1+KCS_DEV_LMT], #kcs max deviation
                          kcs_cv_max=KCS_CV_LMT, #used also for the power
                          #pwr_cv_max=PWR_CV_LMT,
                          pearson_min=PEARSON_LMT, #Pearson only for higher resolution
                          beam=False,#provide beam irradiance for the flagging
                          power_values=None,
                          keep_invalid=False #if True filter out only irradiances < irr_min 
                          )
            df=df2[(df2.i_unstable==0)]
        
        elif ("dates flagging" in process):
           #if "CH7" in process: method="isoc_astm"
           #elif "CH7" not in process: method="isoa_astm_cmp21"
           #if "uncertainty" in process:            
           #    Irradiance_Uncertainty = mdq.Irradiance_Uncertainty(sensitivity=setup.reference_sensitivity,method=method)
           #elif "uncertainty" not in process:
           #    Irradiance_Uncertainty = None
            #if "CH7" not in process: 
           #    Irradiance_Uncertainty.uncertainties_df.loc["calibration","acceptance"]=

          
           df_vld, df_srs, df_days  =mdq.SolarData.datesflagging(SolarLibrary=setup.solar_library, #calibration #8/5/20 difference with days flagging?
                          Irradiance_Uncertainty=Irradiance_Uncertainty,
                          datetimeutc_values=df.loc[:,"datetime"].values, #timezone:str,
                          irradiance_values=df.loc[:,"global"].values, #no index, len should be same as datetimeindex,
                          timeresolution=60,
                          hours_limit=HOURS_LIMIT,
                          coverage_factor=COVERAGE_FACTOR,
                          series_count=setup.series_count,
                          series_values=setup.readings_count, #20
                          irr_min=setup.beam_min,
                          aoi_max=setup.aoi_max,
                          kcs_dev_lmt=KCS_DEV_LMT,
                          pearson_min=PEARSON_LMT)
           
           if len(df_vld)>0:
               df_vld.to_csv(path_to_output+"valid_measures_"+setup.file_suffix+".csv")
               df=df_vld[(df_vld.cloudy_d.isna()) &
                         (df_vld.cloudy_s.isna()) &
                         (df_vld.cloudy_m.isna())]
           elif len(df_vld)==0: sys.exit("valid dataframe empty")
   

  
        #dateindex necessary
        filtered_series = df.loc[:,"datetime"].values
        series_start=[]; series_end=[]; series_n = []; series_id = []
        
        #series_values = pd.Series()
        
        start = filtered_series.min()
        series = [start]
        id_s = 0 
  
        for value in list(filtered_series):  
            if value not in list(series):
                s_tmp = filtered_series[(filtered_series>=value)&(filtered_series<value+np.timedelta64(setup.series_minutes,'m'))] #instead of grouping dynamic per valid value
                n_tmp = len(s_tmp)
                if n_tmp >= setup.readings_count:
                    id_s+=1
                    series = s_tmp
                    series_id.append(id_s)
                    series_start.append(value)
                    series_end.append(value+np.timedelta64(setup.series_minutes,'m'))
                    series_n.append(n_tmp)

  
        #setting options for visualisation        
        pd.set_option('display.max_columns',20)
        pd.set_option('display.max_rows',100) 
    

            
        #creating df with information about the potential series
        #overloaded, for some parameters (aoi )
        indipendent_series = pd.DataFrame({
        "id":series_id,
        "start":series_start,
        "end":series_end,
        "n":series_n})
        
        """
        ,
        "aoi_median":np.full(len(series_n),None), 
        "azimuth_median":np.full(len(series_n),None), 
        "t_ambient_min":np.full(len(series_n),None),    
        "t_ambient_10":np.full(len(series_n),None),
        "t_ambient_median":np.full(len(series_n),None),
        "t_ambient_90":np.full(len(series_n),None),
        "t_ambient_max":np.full(len(series_n),None),     
        "global_min":np.full(len(series_n),None),    
        "global_10":np.full(len(series_n),None),
        "global_median":np.full(len(series_n),None),
        "global_90":np.full(len(series_n),None),
        "global_max":np.full(len(series_n),None),     
        "wind_speed_min":np.full(len(series_n),None),    
        "wind_speed_10":np.full(len(series_n),None),
        "wind_speed_median":np.full(len(series_n),None),
        "wind_speed_90":np.full(len(series_n),None),
        "wind_speed_max":np.full(len(series_n),None),
        "diffuse_fraction_min":np.full(len(series_n),None),    
        "diffuse_fraction_10":np.full(len(series_n),None),
        "diffuse_fraction_median":np.full(len(series_n),None),
        "diffuse_fraction_90":np.full(len(series_n),None),
        "diffuse_fraction_max":np.full(len(series_n),None),
        "diffuse_fraction_weighted":np.full(len(series_n),None)},
        index=series_id)
        """
        
        if indipendent_series.empty== True: print("indipendent_series EMPTY !!")
        #elif indipendent_series.empty == False: print("indipendent_series valid")
          
        
        
        """   
        for index in list(indipendent_series.index):
        #filling df with information about series 
            s_start_tmp=indipendent_series.loc[index,"start"]
            s_end_tmp=indipendent_series.loc[index,"end"]
            df_tmp = df[(df["datetime"]>=s_start_tmp)&(df["datetime"]<=s_end_tmp)]
            indipendent_series.loc[index,"aoi_median"]=df_tmp["aoi"].median()
            indipendent_series.loc[index,"azimuth_median"]=df_tmp["azimuth"].median()
            indipendent_series.loc[index,"t_ambient_median"]=df_tmp["t_ambient"].median()
            indipendent_series.loc[index,"global_median"]=df_tmp["vr_ji"].median()
            indipendent_series.loc[index,"wind_speed_median"]=df_tmp["wind_speed"].median()
            indipendent_series.loc[index,"t_ambient_90"]=np.percentile(df_tmp["t_ambient"],90)
            indipendent_series.loc[index,"global_90"]=np.percentile(df_tmp["vr_ji"],90)
            indipendent_series.loc[index,"wind_speed_90"]=np.percentile(df_tmp["wind_speed"],90)
            indipendent_series.loc[index,"t_ambient_10"]=np.percentile(df_tmp["t_ambient"],10)
            indipendent_series.loc[index,"global_10"]=np.percentile(df_tmp["vr_ji"],10)
            indipendent_series.loc[index,"wind_speed_10"]=np.percentile(df_tmp["wind_speed"],10)
            indipendent_series.loc[index,"t_ambient_min"]=df_tmp["t_ambient"].min()
            indipendent_series.loc[index,"global_min"]=df_tmp["vr_ji"].min()
            indipendent_series.loc[index,"wind_speed_min"]=df_tmp["wind_speed"].min()
            indipendent_series.loc[index,"t_ambient_max"]=df_tmp["t_ambient"].max()
            indipendent_series.loc[index,"global_max"]=df_tmp["vr_ji"].max()
            indipendent_series.loc[index,"wind_speed_max"]=df_tmp["wind_speed"].max()
            indipendent_series.loc[index,"diffuse_fraction_min"]=df_tmp["diffuse_fraction"].min()
            indipendent_series.loc[index,"diffuse_fraction_10"]=np.percentile(df_tmp["diffuse_fraction"],10)
            indipendent_series.loc[index,"diffuse_fraction_median"]=df_tmp["diffuse_fraction"].median()
            indipendent_series.loc[index,"diffuse_fraction_90"]=np.percentile(df_tmp["diffuse_fraction"],90)
            indipendent_series.loc[index,"diffuse_fraction_max"]=df_tmp["diffuse_fraction"].max()  
            indipendent_series.loc[index,"diffuse_fraction_weighted"]=df_tmp["diffuse"].sum()/df_tmp["global"].sum()
        indipendent_series.to_csv(path_to_output+"indipendent_series.csv")    
        """
        
        df_series = indipendent_series
        
        #defining calibration parameters    
    
        """    
        calibration = mdq.SolarData.Calibration(
        location=setup.solar_library, method="9847 1a",reference_calibration_f=setup.reference_calibration_factor)
        df=setup.dataframe 
        # start results table
        calibration_series = []
        calibration_measurements = pd.DataFrame(columns=["id_s","vr_ji","vf_ji","datetime","f_ji","f_dev"])  
       
        for i in list(df_series.index.values):
        #calculation of calibration factor per series
            #filter per series
            df_s_tmp = df[(df["datetime"]>=df_series.loc[i,"start"])&(df["datetime"]<=df_series.loc[i,"end"])]
            
            #df_s_tmp.to_csv(path_to_output+"manual_test.csv")
            
            #skip if empty
            if df_s_tmp.empty == False:
                #vr_j=df_s_tmp["vr_ji"] not needed for get_factor_df2
                #vf_j=df_s_tmp["vf_ji"]
                #datetime=df_s_tmp["datetime"]  
                
                #notify running series 
                #print("series from "+str(df_series.loc[i,"start"])+" to "+str(df_series.loc[i,"end"]))
                #get dataframe and main parameters for calculation 
                #f_j_std is the standard deviation of the series normalised by the calibration factor
                valid,f_j,f_j_std,len_j,results_j_df=calibration.get_factor_df2(
                df=df_s_tmp,                       
                reference_column="vr_j", 
                field_column="vf_j",
                datetime_column="datetime",
                deviation_max=setup.deviation_max,
                readings_min=setup.readings_count,
                iterations_max=ITERATIONS_MAX 
                )
                
                #require adding the series only if calibration calculation successful 
                if valid == 0:             
                    #create array with series id
                    results_j_df.assign(id_s=np.full(len(results_j_df),df_series.loc[i,"id_s"]))
                    #concatenate series df with previous one
                    #DEV NOTE 8/1/19: sort=False removed but not clear why necessary
                    calibration_measurements = pd.concat([calibration_measurements,results_j_df])
                    #define main results for series     
                    results_j=[
                    df_series.loc[i,"id_s"],
                    df_series.loc[i,"start"],
                    df_series.loc[i,"end"],            
                    df_series.loc[i,"aoi_median"],
                    df_series.loc[i,"t_ambient_median"],
                    df_series.loc[i,"global_median"],
                    df_series.loc[i,"wind_speed_median"],
                    df_series.loc[i,"diffuse_fraction_median"],
                    df_series.loc[i,"diffuse_fraction_weighted"],
                    df_series.loc[i,"azimuth_median"],
                    valid,
                    f_j,
                    f_j_std,
                    len_j]
                    #append results
                    calibration_series.append(results_j)
                    
                    
                #sys.exit("test end")
                
        #transform summary into dataframe
        calibration_series_df = pd.DataFrame(calibration_series,columns=
        ["id_s","start","end",
         "aoi_median","t_ambient_median","global_median","wind_speed_median","diffuse_fraction_median","diffuse_fraction_weighted","azimuth_median", 
         "valid","f","std","records"])
        calibration_series_df.index=calibration_series_df["id_s"]
                
        if calibration_series_df.empty== True:
            error_message = "calibration series summary EMPTY !!" 
            sys.exit(error_message)
        elif calibration_series_df.empty == False:
            print("calibration_series ok: filled")
        
        if calibration_measurements.empty== True:
            error_message = "calibration series EMPTY !!"
            sys.exit(error_message)        
        elif calibration_measurements.empty == False:
            print("calibration_measurements ok: filled")
            #print(calibration_measurements.iloc[0,:])
        """
        
        """
        #export both summary and singles series measurements  
        calibration_series_df[calibration_series_df["valid"]==True].to_csv(path_to_output+"calibration_series.csv")
        calibration_measurements.to_csv(path_to_output+"calibration_measurements.csv")
        #selecting only valid series
        calibration_series_df=calibration_series_df[calibration_series_df["valid"]==True]
        #definition of parameters 
        """
             

   
    
    if ("calibration" in process) and ("eurac" in process):
        if "predefined" in process:df_series = SERIES_PREDEFINED_DF 
        #max columns and rows
        pd.set_option('display.max_columns',10)
        pd.set_option('display.max_rows',30)
        #predefined selected time series (EURAC)
        
        #defining calibration parameters    
        calibration = mdq.SolarData.Calibration(
        location=setup.solar_library, method="9847 1a",calibration_factor=setup.reference_calibration_factor)
        df=setup.dataframe 
        # start resultsable
        calibration_series = pd.DataFrame()      
        calibration_measurements = pd.DataFrame(columns=["id","vr_ji","vf_ji","datetime","f_ji","f_dev"])  
        for i in list(df_series.index.values):
            #print(i,df_series.loc[i,"start"],df_series.loc[i,"end"]) CHECK
            df_s_tmp = df[(df["datetime"]>=df_series.loc[i,"start"])&(df["datetime"]<df_series.loc[i,"end"])] #filter per series
            if df_s_tmp.empty == False:  #skip if empty
                #vr_j=df_s_tmp["vr_ji"] #not required with get_factor_df2
                #vf_j=df_s_tmp["vf_ji"]
                #datetime=df_s_tmp["datetime"]
      
                #print("series from "+str(df_series.loc[i,"start"])+" to "+str(df_series.loc[i,"end"]))  
                #get dataframe and main parameters for calculation 
                valid,f_j,f_j_std,r_j_std,len_j,results_j_df=calibration.get_factor_df2(
                df_s_tmp,
                reference_column="vr_j",
                field_column="vf_j",
                datetime_column="datetime",
                #reference_series=vr_j, 
                #field_series=vf_j,
                #time_series=datetime,
                deviation_max=setup.deviation_max,
                readings_min=setup.readings_count,#for eurac
                iterations_max=ITERATIONS_MAX,
                median=False
                )
                #create array with series id
                if results_j_df is not None:
                    results_j_df.loc[:,"id"]=np.full(len(results_j_df),df_series.loc[i,"id"])
                    #concatenate series df with previous one
                    calibration_measurements = pd.concat([calibration_measurements,results_j_df],sort=False)
                    #define main results for series   
                    results_j=pd.DataFrame()
                    for column in list(df.columns):
                        if (column!="vr_ji")and(column!="vf_ji"):
                            if column=="datetime": results_j.loc[0,"datetime_quantile"]=results_j_df.loc[:,"datetime"].astype('datetime64[ns]').quantile(.5)
                            #https://stackoverflow.com/questions/43889611/median-of-panda-datetime64-column
                            elif column!="datetime": 
                                results_j.loc[0,str(column+"_median")]=results_j_df.loc[:,column].median()
                                if (column!="azimuth") and (column!="aoi"): results_j.loc[0,str(column+"_std")]=results_j_df.loc[:,column].std()
                    results_j.loc[0,"id"]=df_series.loc[i,"id"]
                    results_j.loc[0,"start"]=df_series.loc[i,"start"]
                    results_j.loc[0,"end"]=df_series.loc[i,"end"]   
                    results_j.loc[0,"f_j"]= f_j
                    results_j.loc[0,"f_j_std"]=f_j_std
                    results_j.loc[0,"r_j_std"]=r_j_std
                    results_j.loc[0,"len_j"]=len_j
                    #append results
                    if calibration_series.empty==True: calibration_series=results_j
                    elif calibration_series.empty==False: calibration_series = calibration_series.append(results_j)
            #elif df_s_tmp.empty == True: print("series "+str(df_series.loc[i,"id"])+" from "+str(df_series.loc[i,"start"])+" to "+str(df_series.loc[i,"end"])+" is empty in the dataframe")
            
            
        #3/6/20 calculate uncertainty of found calibration values        
        method="isoa_astm_cmp21"; iu_c_ref=mdq.Irradiance_Uncertainty(sensitivity=setup.reference_sensitivity,method=method)
        #total zero offset kept, different also between CMP21 & 11 due to temperature compensation, plus individual tests are required      
        iu_c_ref.uncertainties_df.loc["non_stability","acceptance"]=0 #4620 not relevant for calibration, reference recently calibrated at JRC
        #non-linearity kept but later removed for same model
        iu_c_ref.uncertainties_df.loc["directional_response","acceptance"]=0#4620 different angles considered
        #spectral_error kept
        #temperature response kept,  different also between CMP21 & 11 due to temperature compensation, plus individual tests are required      
        iu_c_ref.uncertainties_df.loc["tilt_response","acceptance"]=0 #4620 assumed negligible since close to 0 (while range from 0 to 180)
        iu_c_ref.uncertainties_df.loc["signal_processing","acceptance"]=0.2 #see files uncertainty analysis and calibration analysis
        iu_c_ref.uncertainties_df.loc["signal_processing","parameter"]="sensitivity" #as voltage percentage
        #calibration CMP21 kept
        iu_c_ref.uncertainties_df.loc["maintenance","acceptance"]=0 #4620 not relevant for calibration, reference recently calibrated at JRC
        #alignment zenith kept, assumed same operator accuracy
        iu_c_ref.uncertainties_df.loc["alignment_azimuth","acceptance"]=0 #4620 alignment error not relevant for horizontal if azimuth not considered
        iu_c_ref.uncertainties_df.loc["spectral_error","acceptance"]=0
        iu_c_test=iu_c_ref #4/6/20 same uncertainties transferred to test
        iu_c_test.uncertainties_df.loc["calibration","acceptance"]=0 #090620 to be considered later/
        #if "CH7" not in process:
        #    iu_c_ref.uncertainties_df.loc["non_linearity","acceptance"]=0 #4620 assumed from same design
        #    iu_c_test.uncertainties_df.loc["non_linearity","acceptance"]=0
        #    iu_c_ref.uncertainties_df.loc["spectral_error","acceptance"]=0 #4620 assumed from same design
        #    iu_c_test.uncertainties_df.loc["spectral_error","acceptance"]=0
        if "CH7" in process:
            iu_c_test.uncertainties_df.loc["total_zero_offset","acceptance"]=iu_c_test.uncertainties_df.loc["total_zero_offset","acceptance_c"]
            iu_c_test.uncertainties_df.loc["non_linearity","acceptance"]=iu_c_test.uncertainties_df.loc["non_linearity","acceptance_c"]
            #iu_c_test.uncertainties_df.loc["spectral_error","acceptance"]=iu_c_test.uncertainties_df.loc["spectral_error","acceptance_c"]
            iu_c_test.uncertainties_df.loc["temperature_response","acceptance"]=iu_c_test.uncertainties_df.loc["temperature_response","acceptance_c"]

        def unc(irradiance_total,zenith,azimuth):#11/6/20 to be replaced with uncertainty of                
           u=Irradiance_Uncertainty.get_uncertainty_gum(irradiance_total=irradiance_total,
                                                     diffuse_fraction=0,
                                                     zenith=zenith,
                                                     azimuth=azimuth,
                                                     surface_zenith=SolarLibrary.surface_zenith,
                                                     surface_azimuth=SolarLibrary.surface_azimuth,
                                                     temperature=None)
           return u    
        df_vld["irradiance_uncertainty"] = np.vectorize(unc)(df_vld['irradiance'], df_vld['zenith'], df_vld["azimuth"])
        
        calibration_series.to_csv(path_to_output+"calibration_series_"+setup.file_suffix+".csv")
        calibration_measurements.to_csv(path_to_output+"calibration_measurements_"+setup.file_suffix+".csv")
   
    """12/6/20 removed
    if "aoi distribution" in process: #5/5/2020 not necessary could be done manually
        azimuth_east = 90
        azimuth_south = 180
        azimuth_west = 270  
        aoi_max = 70    
        
        aoi_east_min = calibration_series_df[(calibration_series_df["azimuth_median"]<=azimuth_south)]["aoi_median"].min()
        aoi_west_min = calibration_series_df[(calibration_series_df["azimuth_median"]>=azimuth_south)]["aoi_median"].min() 
        step_east = (aoi_max-aoi_east_min)/7
        step_west = (aoi_max-aoi_west_min)/7
        step_deviation_east =  step_east/2
        step_deviation_west =  step_west/2
        step_deviation_increase = 0.02 
    
        calibration_measurements_limited=pd.DataFrame(columns=
        ["id_s","start","end",
        "aoi_median","t_ambient_median","global_median","wind_speed_median","diffuse_fraction_median","diffuse_fraction_weighted","azimuth_median", 
        "valid","f","std","records","index","aoi_target"])
    
        #creation of a series_index 
        series_index=[]
        for i in range(0,int(setup.series_count/2)):
            series_index.append(i+1)
            series_index.append(setup.series_count-i)
        series_index_mean = int(np.average(series_index))
        series_index.append(series_index_mean)
        
    
        
        index_tmp = series_index
       
    
        calibration_series_df_tmp = calibration_series_df
        
    
        
        for iteration in range(0,ITERATIONS_MAX-1):           
        #impose a limit to the ammissible iterations (not in standard, just to avoid infinite loops)
            for index_s in list(index_tmp):
                #DEV NOTE 22/11/18: aoi_target may be defined separately for different purpose
                if index_s == 8:
                    aoi_target = 0
                    series_candidates = calibration_series_df_tmp[
                    (calibration_series_df_tmp["aoi_median"]==calibration_series_df_tmp["aoi_median"].min())]      
                elif (index_s>=1) and (index_s<8):
                    aoi_target = aoi_max-step_east*(index_s-1)
                    series_candidates = calibration_series_df_tmp[
                    (calibration_series_df_tmp["aoi_median"]>=aoi_target-step_deviation_east)&
                    (calibration_series_df_tmp["aoi_median"]<=aoi_target+step_deviation_east)&
                    (calibration_series_df_tmp["azimuth_median"]<=azimuth_south)]
                elif (index_s>8) and (index_s<=15):
                    aoi_target = aoi_max-step_west*(15-index_s)
                    series_candidates = calibration_series_df_tmp[
                    (calibration_series_df_tmp["aoi_median"]>=aoi_target-step_deviation_west)&
                    (calibration_series_df_tmp["aoi_median"]<=aoi_target+step_deviation_west)&
                    (calibration_series_df_tmp["azimuth_median"]>=azimuth_south)]
                
                if series_candidates.empty == False:                
                    
                    series_candidates=series_candidates.assign(aoi_median_dev=
                    series_candidates.loc[:,"aoi_median"].apply(lambda x:abs(x-aoi_target)))                                       
             
                    series_chosen = series_candidates[(series_candidates["aoi_median_dev"]==series_candidates.loc[:,"aoi_median_dev"].min())]
                    
                    series_chosen = series_chosen.assign(aoi_target=
                    np.full(len(series_chosen),aoi_target))                                   
                                 
                    series_chosen = series_chosen.assign(index=
                    np.full(len(series_chosen),index_s))
                    
                    index_tmp.remove(index_s)
                    #index_tmp.remove(index_s)  # for list
                    
                    calibration_series_df_tmp=calibration_series_df_tmp.drop(index=series_chosen.loc[:,"id_s"])                
                    
                    #DEV NOTE 8/1/18: sort='False' removed not clear why
                    calibration_measurements_limited = calibration_measurements_limited.append(series_chosen)
                                    
                    
                            
            if not index_tmp: 
                break
            elif index_tmp:
                #DEV NOTE 22/11/18: not perfect since higher aoi target may be assigned to lower aoi target if closer
                step_deviation_east =  step_deviation_east*(1+step_deviation_increase)
                step_deviation_west =  step_deviation_west*(1+step_deviation_increase)
                
        
        if  calibration_measurements_limited.empty== True:
            err_msg = "ERROR: calibration_measurements_limited empty!!"
            sys.exit(err_msg)
        elif  calibration_measurements_limited.empty == False:
            print(" calibration_measurements_limited ok: not empty")
        
     
        
        calibration_measurements_limited.to_csv(path_to_output+"calibration_measurements_limited.csv") 
    """
    
    
        
     
  
    if "crest indoor time response" in process: #1/5/20 updated
        #DEV NOTE 19/11/18: to be checked and eventually improved
        PATH_TO_INPUT = PATH_TO_INPUT_FOLDER_CREST_IN
        
        #DEV NOTE 10/1/19: output in Python test
        #path_to_output = PATH_TO_INPUT
          
        
        files_list = CREST_CALIBRATION_FILES 
        
        
        #retrieve positions of the sections
        all_sections_positions = datacrest._all_sections_positions(path_to_input=PATH_TO_INPUT,files_list=files_list,path_to_output=path_to_output)
        
        
        
        
        if all_sections_positions.empty== True:
            sys.exit("all_sections_positions empty")
        elif all_sections_positions.empty == False:
            print("all_sections_positions ok")
            #print(df.iloc[0,:])
        
        
        #print(all_sections_positions[all_sections_positions.info_value.str.startswith("Wacom Irradiance Calibration Record Data")])
        
        
        
        #as reference extract and save main information
        # dev note 23/08/18: path output not necessary for deeper analysis 
        df_main_information = datacrest._main_info_to_dataframe(all_sections_positions,files_list,
                                    path_to_input=PATH_TO_INPUT,
                                    path_to_output=path_to_output)
        
        if df_main_information.empty== True:
            sys.exit("df_main_information empty")
        elif df_main_information.empty == False:
            print("df_main_information ok")
            #print(df.iloc[0,:])
        
        
        #modified section to retrieve CMP21 only 
        #files_list_CMP21 = list(df_main_information[(df_main_information.sample_type == "CMP21")]["filename"])
        #files_list = files_list_CMP21
        
        files_list = list(df_main_information["filename"])
        
        #files_list_CMP21 = list(df_main_information[(df_main_information.reference_id == "CMP21-140465")]["filename"])
    
        files_list=list([
                "WAC-010-20180704160532-20180509010-000-RESTRICTED.txt",
                "WAC-010-20180720123750-CMP11060118-000-RESTRICTED.txt",
                "WAC-010-20180720105136-CMP11174112-000-RESTRICTED.txt",
                "WAC-010-20180720113819-CMP11174113-000-RESTRICTED.txt",
                "WAC-010-20180831174208-CMP21090318-000-INTERNAL.txt",
                "WAC-010-20180720143009-CMP21140464-000-RESTRICTED.txt",
                "WAC-010-20171128154513-CREST000026-000-RESTRICTED.txt",
                "WAC-010-20171030125533-CREST000027-000-RESTRICTED.txt",
                "WAC-010-20180710150157-CREST000091-000-RESTRICTED.txt",
                "WAC-010-20171220152217-20171113020-000-RESTRICTED.txt",#20171113020	SMP11-V	['"Ref', 'as', 'atandard', 'sensitivity', 'reading,', 'DUT', 'on', 'digital', 'to', 'analogue', 'converter', 'output']
                ])

        #files_list=list(["WAC-010-20171220152217-20171113020-000-RESTRICTED.txt",#20171113020	SMP11-V	['"Ref', 'as', 'atandard', 'sensitivity', 'reading,', 'DUT', 'on', 'digital', 'to', 'analogue', 'converter', 'output']
        #        "WAC-010-20171220155237-20171113020-000-RESTRICTED.txt",#20171113020	SMP11-V	['"Ref', 'as', 'atandard', 'sensitivity', 'reading,', 'DUT', 'digital', 'read', 'through', 'Logging', 'software"']
        #        "WAC-010-20171208152315-20171113020-000-RESTRICTED.txt"])

               
        
        #extract into a datframe the sub meas results
        # dev note 23/08/18: path output not necessary for deeper analysis 
        df_submeasures = datacrest._submeasures_to_dataframe(all_sections_positions=all_sections_positions,
                                  path_to_input=PATH_TO_INPUT,
                                  files_list=files_list,
                                  path_to_output=path_to_output)
        
        
        if df_submeasures.empty== True:
            sys.exit("df_submeasures empty")
        elif df_submeasures.empty == False:
            print("df_submeasures ok")
            #print(df.iloc[0,:])
        
        
        #extract into a dataframe the results 
        df_results = datacrest._results_to_dataframe(all_sections_positions=all_sections_positions,
                                                     path_to_input=PATH_TO_INPUT,files_list=files_list,
                                                     path_to_output=path_to_output)
        
        
        if df_results.empty== True:
            sys.exit("df_results empty")
        elif df_results.empty == False:
            print("df_results ok")
            #print(df.iloc[0,:])
        
        
    
    
    
        #CODE TO DOUBLE CHECK (CALCULATE) SENSITIVITY FOR ALL FILES 
        # DEV NOTE 30/7/18: to update measurement units V 
        #definition and initialisation of new output dataframe
        df_sub_meas_columns = ("filename","sub_meas_id","sam_voltage_av","ref_voltage_av",
                               "irradiance","ref_sensitivity",
                               "sam_sensitivity","sam_sensitivity_calc")
        df_sub_meas = pd.DataFrame(columns=df_sub_meas_columns)
        #iteration to extract all the values
        for item in files_list:
            #filtering for results
            df_results_tmp  = df_results[(df_results.filename == item)][["datetime","value","parameter"]]  
            #filtering for submeasures
            df_submeasures_tmp = df_submeasures[(df_submeasures.filename == item)] 
            #getting reference pyranometer sensitivty
            ref_sensitivity_tmp = float(df_main_information [(df_main_information.filename == item)][("reference_sensitivity [uV/Wm-2]")])
            #iteration for single measurements
            
            #for i in range(5): #30/04/20 considering only first series of measurements
            if df_results.empty== True:  
                 i=0
                
                 #acquisition of parameters for reference pyranometer
                 df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-REF-Meas_Dark-R"))]["value"]
                 ref_dark_av = np.average(df_meas_tmp)                  
                 df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-REF-Meas_Light-R"))]["value"]
                 ref_light_av = np.average(df_meas_tmp)
                 #ref_voltage transformed from mV to microVolt
                 ref_voltage_av = (ref_light_av - ref_dark_av) * 1000
                 #acquisition of parameters for sample pyranometer
                 df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-SAM-Meas_Dark-R"))]["value"]
                 sam_dark_av = np.average(df_meas_tmp)
                 df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-SAM-Meas_Light-R"))]["value"]
                 sam_light_av = np.average(df_meas_tmp)        
                 #sam_voltage transformed from mV to microVolt
                 sam_voltage_av = (sam_light_av - sam_dark_av) * 1000 
                 #acquisition of already provided values
                 
                 #TypeError: cannot convert the series to <class 'float'>
                 irradiance_av = float(df_submeasures_tmp[df_submeasures_tmp["Sub-Meas ID"].str.contains("M"+str(i))]["Irradiance [uV/Wm-2]"])             
                 
                 
                 sensitivity_av = float(df_submeasures_tmp[df_submeasures_tmp["Sub-Meas ID"].str.contains("M"+str(i))]["Sensitivity [uV/Wm-2]"])                           
                 
                    
                    
                 #calculation of sensitivity as comparison 
                 
                 #0 division
                 
                 try:
                     sensitivity_av_calc = float(sam_voltage_av)/float(ref_voltage_av)*float(ref_sensitivity_tmp)
                 except:
                     print(item)
                     sys.exit("check: "+item)
                 
                 #assignement of values to the df 
                 
                 
                 df_sub_meas = df_sub_meas.append({
                 df_sub_meas_columns[0]:item,
                 df_sub_meas_columns[1]:i,
                 df_sub_meas_columns[2]:sam_voltage_av,
                 df_sub_meas_columns[3]:ref_voltage_av,
                 df_sub_meas_columns[4]:irradiance_av,
                 df_sub_meas_columns[5]:ref_sensitivity_tmp,
                 df_sub_meas_columns[6]:sensitivity_av,
                 df_sub_meas_columns[7]:sensitivity_av_calc,
                 },ignore_index=True)
             
                 """
                 df_sub_meas = df_sub_meas.append({
                 df_sub_meas_columns[0]:item,
                 df_sub_meas_columns[1]:i,
                 df_sub_meas_columns[2]:sam_voltage_av,
                 df_sub_meas_columns[3]:ref_voltage_av,
                 },ignore_index=True)
                 """
                 
            
        #export table with all values  
        if len(df_sub_meas)>0: df_sub_meas.to_csv(path_to_output+"report.csv")
          
        
        
        
        
        #CODE TO CALCULATE SENSITIVITY VARIATION FOR ALL FILES 
        #definition and initialisation of new output datafra
        df_v_delta_columns = ("filename","sample_type","sample_details",
                              "sub_meas_id","sam_voltage_av [V]","ref_voltage_av [V]",
                               "irradiance [Wm-2]","ref_sensitivity [uv/Wm-2]",
                               "sam_sensitivity",
                               "sam_v_light_slope","v_%_av_slope",
                               "intercept","r_value","p_value","std_err",
                               "std_irradiance",
                               "start [s]","end [s]",
                               "time_response","response_percentage")
        df_v_delta = pd.DataFrame(columns=df_v_delta_columns)
        #iteration to extract all the values
        
        
        
        axes_y_max=100
        time_annotations= []
        for item in files_list:
            
            #create dataframe for plotting meas
            df_plot_meas_columns = ["seconds","value"]
            df_plot_meas = pd.DataFrame(columns=df_plot_meas_columns)    
            
            #create dataframe for plotting settle
            df_plot_settle_columns = df_plot_meas_columns 
            df_plot_settle = pd.DataFrame(columns=df_plot_settle_columns) 
            
            
            
            #filtering for results
            df_results_tmp  = df_results[(df_results.filename == item)][["datetime","value","parameter"]]
            #filtering for submeasures
            df_submeasures_tmp = df_submeasures[(df_submeasures.filename == item)] 
            #getting reference pyranometer sensitivty
            ref_sensitivity_tmp = float(df_main_information [(df_main_information.filename == item)][("reference_sensitivity [uV/Wm-2]")])
        
            #DEV NOTE 6/8/18: expection introduced to deal with 3 only measurements in old format
            #DEV NOTE 6/8/18: creation of number column to identify sessions 
            
            
            df_results_tmp = df_results_tmp.assign(session_id_number=lambda x: x.parameter.str[1])
            last_session_tmp = int(max(df_results_tmp.session_id_number))
        
            #set up of seconds as x variable 
            #df_sam_meas_tmp = df_sam_meas_tmp.assign(seconds= lambda x: x.datetime)
            #df_sam_meas_tmp.loc[:,"seconds"] = df_sam_meas_tmp.loc[:,"seconds"].apply(lambda x:x.second+x.minute*60+x.hour*3600+x.microsecond/1000000)
            
            
                #iteration for set of measurements Mi 
            #for i in range(last_session_tmp+1):#30/04/20 considering only first series of measurements
            if df_results.empty== False:  
                i=0 #12/6/20 to be checked
                
                
                #ACQUISITION OF FINALISED PARAMETERS FROM SUBMEASURED
                irradiance_av = float(df_submeasures_tmp[df_submeasures_tmp["Sub-Meas ID"].str.contains("M"+str(i))]["Irradiance [uV/Wm-2]"])
                sensitivity_av = float(df_submeasures_tmp[df_submeasures_tmp["Sub-Meas ID"].str.contains("M"+str(i))]["Sensitivity [uV/Wm-2]"])                               
                #ACQUISITION OF VALUES FROM RESULTS
                #acquisition of parameters for reference pyranometer
                df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-REF-Meas_Dark-R"))]["value"]
                ref_dark_av = np.average(df_meas_tmp)                  
                df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-REF-Meas_Light-R"))]["value"]
                ref_light_av = np.average(df_meas_tmp)
                ref_voltage_av = (ref_light_av - ref_dark_av)
                #acquisition of parameters for sample pyranometer
                df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-SAM-Meas_Dark-R"))]["value"]
                sam_dark_av = np.average(df_meas_tmp)
                df_meas_tmp  = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)+"-SAM-Meas_Light-R"))]["value"]
                sam_light_av = np.average(df_meas_tmp)        
                sam_voltage_av = (sam_light_av - sam_dark_av) 
                #interpolation calculation for measurements stability 
                
                #FILTERING OF SAMPLE MEAS VALUES ONLY    
                df_sam_meas_tmp = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)))&
                                           (df_results_tmp.parameter.str.contains("-SAM-Meas_Light-R"))]
                #DEV NOTE: separated creation of columns to avoid alert later 
                #set up of seconds as x variable 
                df_sam_meas_tmp = df_sam_meas_tmp.assign(seconds= lambda x: x.datetime)
                df_sam_meas_tmp.loc[:,"seconds"] = df_sam_meas_tmp.loc[:,"seconds"].apply(lambda x:x.second+x.minute*60+x.hour*3600+x.microsecond/1000000)
                #aggregating line as if continous time
                
             
                start = min(df_sam_meas_tmp.loc[:,"seconds"].values)
                #DEV NOTE 6/8/18: relative end replaced by 10s to compare different calibrations
                end = start + MEASUREMENT_TIME
                #end = max(df_sam_meas_tmp.loc[:,"seconds"].values)
        
                
                #to simplify imposing 0 for the starting series measurement 
                df_sam_meas_tmp.loc[:,"seconds"] = df_sam_meas_tmp.loc[:,"seconds"].apply(lambda x: x-start+STABILISATION_TIME*i)
                #set up of sam voltage as y variable
                df_sam_meas_tmp = df_sam_meas_tmp.assign(sam_voltage= lambda x: x.value - sam_dark_av)
                
                #DEV NOTE: check relevance of calculated statistical parameters 
                #identify parameters of expected regression         
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_sam_meas_tmp.loc[:,"seconds"].values,
                                                                               df_sam_meas_tmp.loc[:,"sam_voltage"].values)
                
                #FILTERING OF SAMPLE SETTLE VALUES ONLY
                df_sam_settle_tmp = df_results_tmp[(df_results_tmp.parameter.str.contains("M"+str(i)))&
                                           (df_results_tmp.parameter.str.contains("-SAM-Settle_Light-R"))]
                #DEV NOTE: separated creation of columns to avoid alert later 
                #set up of seconds as x variable 
                df_sam_settle_tmp = df_sam_settle_tmp.assign(seconds= lambda x: x.datetime)
                df_sam_settle_tmp.loc[:,"seconds"] = df_sam_settle_tmp.loc[:,"seconds"].apply(lambda x:x.second+x.minute*60+x.hour*3600+x.microsecond/1000000)
                
                #aggregating line as if continous time
                start = min(df_sam_settle_tmp.loc[:,"seconds"].values)
                 
    
                if i == 0:
                    start = min(df_sam_settle_tmp.loc[:,"seconds"].values)
                    #DEV NOTE 6/8/18: relative end replaced by 10s to compare different calibrations
                    end = start + STABILISATION_TIME
                    end = max(df_sam_settle_tmp.loc[:,"seconds"].values)
                elif i > 0:
                    start = min(df_sam_settle_tmp.loc[:,"seconds"].values)
                    #start = start + min(df_sam_settle_tmp.loc[:,"seconds"].values) - end            
                    #DEV NOTE 6/8/18: relative end replaced by 10s to compare different calibrations     
                    end = start + STABILISATION_TIME
                    end = max(df_sam_settle_tmp.loc[:,"seconds"].values)
    
                
                #select only values for the selected interval
                df_sam_settle_tmp = df_sam_settle_tmp[(df_sam_settle_tmp.seconds >= start)&
                                                      (df_sam_settle_tmp.seconds <= end)]                                               
                
                
                if df_sam_settle_tmp.shape[0] > 1:             
                    #to simplify imposing 0 for the starting series measurement 
                    df_sam_settle_tmp.loc[:,"seconds"] = df_sam_settle_tmp.loc[:,"seconds"].apply(lambda x: x-start+STABILISATION_TIME*i)
                    #set up of sam voltage as y variable
                    df_sam_settle_tmp = df_sam_settle_tmp.assign(sam_voltage= lambda x: x.value - sam_dark_av)
                    
                    #DEV NOTE: check relevance of calculated statistical parameters 
                    #identify parameters of expected regression         
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df_sam_settle_tmp.loc[:,"seconds"].values,
                                                                                   df_sam_settle_tmp.loc[:,"sam_voltage"].values)
              
                    #TO BE DONE 
                    #acquisition of standard deviation assumed for 1000 Wm-2 (beware different measurement units)
                    #DEV NOTE: IPC exclusion critera std > 1 within 1 second
                    std_irradiance_1000 = np.std(df_sam_meas_tmp.loc[:,"sam_voltage"].values)
                    #*1000/sensitivity_av*1000/irradiance_av       
                    #/sam_voltage_av/irradiance_av*1000
                   
                    #calculation slope %
                    slope_p = slope / ref_light_av * 100# 120620 later in delta      

                    #append values for the plot
                    
                    #DEV NOT: create tmp df before appending 
                    df_plot_meas_tmp = pd.DataFrame({
                    df_plot_meas_columns[0]: df_sam_meas_tmp.loc[:,'seconds'],
                    df_plot_meas_columns[1]: df_sam_meas_tmp.loc[:,'value']})
                    df_plot_meas = df_plot_meas.append(df_plot_meas_tmp, ignore_index=True)
                    #DEV NOT: create tmp df before appending 
                    df_plot_settle_tmp = pd.DataFrame({
                    df_plot_settle_columns[0]: df_sam_settle_tmp.loc[:,'seconds'],
                    df_plot_settle_columns[1]: df_sam_settle_tmp.loc[:,'value']})
                    df_plot_settle = df_plot_settle.append(df_plot_settle_tmp, ignore_index=True)
                
        
            
            #if i == last_session_tmp:
            #i=0 # 12/6/20 if 0 and then 0 not clear
            #if i == 0: #1/5/20 only one minute
                
            label=df_main_information.loc[df_main_information.filename==item,"sample_type"].values[0]#1206020 check for previous sample_type into df
                
                #calculate percentage variation of light - dark measurements
                #df_plot_meas.loc[:,df_plot_meas_columns[1]] = df_plot_meas.loc[:,df_plot_meas_columns[1]].apply(lambda x: x/np.average(df_plot_meas.loc[:,df_plot_meas_columns[1]])*100)       
                # plot graph 
                #p, = plt.plot(df_plot_meas[df_plot_meas_columns[0]],df_plot_meas[df_plot_meas_columns[1]])
                # DEV NOTES: subplot not necessary ? 
                
                #calculate percentage variation of light - dark measurements
            df_plot_settle.loc[:,df_plot_settle_columns[1]] = df_plot_settle.loc[:,df_plot_settle_columns[1]].apply(lambda x: x/np.average(df_plot_meas.loc[:,df_plot_settle_columns[1]])*100)       
            # plot graph p,
            tresponse_df=df_plot_settle.iloc[(df_plot_settle['value']-95).abs().argsort()[:1]]#120620 approximate time response
            tr_seconds=tresponse_df.iloc[0,0]
            tr_value=tresponse_df.iloc[0,1]
            
            #assignement of values to the df   
            df_v_delta = df_v_delta.append({
            df_v_delta_columns[0]:item,
            df_v_delta_columns[1]:str(i), 
            df_v_delta_columns[2]:df_main_information.loc[df_main_information.loc[:,"filename"]==item,"sample_type"].values,
            df_v_delta_columns[3]:df_main_information.loc[df_main_information.loc[:,"filename"]==item,"sample_details"].values,
            df_v_delta_columns[4]:sam_voltage_av,
            df_v_delta_columns[5]:ref_voltage_av,
            df_v_delta_columns[6]:irradiance_av,
            df_v_delta_columns[7]:ref_sensitivity_tmp,
            df_v_delta_columns[8]:sensitivity_av,
            df_v_delta_columns[9]:slope,
            df_v_delta_columns[10]:slope_p,
            df_v_delta_columns[11]:intercept,
            df_v_delta_columns[12]:r_value,
            df_v_delta_columns[13]:p_value,
            df_v_delta_columns[14]:std_err,
            df_v_delta_columns[15]:std_irradiance_1000,
            df_v_delta_columns[16]:start,
            df_v_delta_columns[17]:end,
            df_v_delta_columns[18]:tr_seconds,
            df_v_delta_columns[19]:tr_value
            },
            ignore_index=True)  
            
                        
            
            
            font = {'family' : 'normal',
                    #'weight' : 'bold',
                    'size'   : 22}

            matplotlib.rc('font', **font)
            
            #fig, axes = plt.subplots()
            if  axes_y_max<df_plot_settle[df_plot_settle_columns[1]].max():
                axes_y_max=df_plot_settle[df_plot_settle_columns[1]].max()
            
            
            axes = plt.gca()
            time_annotation= str(round(tr_seconds,1))+'s'
            if time_annotation not in time_annotations: 
                axes.annotate(time_annotation,xy=(tr_seconds,tr_value))
                time_annotations.append(time_annotation)
            axes.set_ylim([80,axes_y_max])#120620 105 instead of 100 to see changing
            axes.set_xlim([0,180])
            
                        
            
            plt.ylabel("calibration response [%]")
            plt.xlabel("time [s]")
            plt.grid(b=True,which="both")
            
            
            
            p, = plt.plot(df_plot_settle[df_plot_settle_columns[0]],df_plot_settle[df_plot_settle_columns[1]],label=label)
            # DEV NOTES: subplot not necessary ?        
            # TO BE ADAPTED FOR VOLTAGE     
            #plt.show()
                
        
        
        #DEV NOTE 7/12/18: AttributeError: 'list' object has no attribute 'get_label'
        #plt.legend(handles=[files_list])
            
           
        
        #export table with all values  
        df_v_delta.to_csv(path_to_output+"sens_var.csv")
    
                       
                       
                           
    """ 
    #PLOT TEST TO BE CHECKED 23/8/18
           
    p1,= plt.plot(df_do['seconds'],df_do['value'],'yo')
    subplot = plt.subplot()
    plt.legend([p1],["WAC-010-20180704160532-20180509010-000"],
    bbox_to_anchor=(0,-1,1,-1), loc=8, mode="expand", ncol=1,borderaxespad=0.)
                
    
    figshow.plotvstime(df_do,time_label="datetime",frequency="d",
                       merged_y=False,title=None,
                       legend_anchor=-0.4,path_to_output=None)
    
    
    df = pd.read_csv(PATH_TO_INPUT+"Test.csv",header=0)[["V_ref","A_ref","V_sam","A_sam"]]
    
    df["P_ref"]=df["V_ref"].multiply(df["A_ref"])
    df["P_sam"]=df["V_sam"].multiply(df["A_sam"])
    
    p1,= plt.plot(df['P_ref'],df['V_ref'],'yo')
    p2,= plt.plot(df['P_sam'],df['V_sam'],'ko')
    
            
    subplot = plt.subplot()
    
    plt.legend([p1,p2],
    ["reference","sample"],
     bbox_to_anchor=(0,-1,1,-1), loc=8, mode="expand", ncol=1,
     borderaxespad=0.)
    
                    
    plt.Axes.set_ylim(subplot)
       
    
    plt.title("SAM-Meas_Light-R")
    plt.ylabel('voltage variation from average [%]')
    plt.xlabel('seconds')
    plt.xticks(rotation='vertical')
    
    
        df_tmp["seconds"] = df_tmp["datetime"].apply(lambda x: x.second+x.minute*60+x.hour*3600)
        start = min(df_tmp.loc[:,"seconds"].values)
        average = np.mean(df_tmp.loc[:,"value"].values)
        df_tmp["seconds"] = df_tmp["seconds"].apply(lambda x: x-start)
        #df_tmp["value"] = df_tmp["value"].apply(lambda x: 100*x/average)
    """
    
    
    
    
    
    
    
    
    
    
     
    
    
    
    
    
