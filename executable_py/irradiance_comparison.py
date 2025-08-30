# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
HISTORY
Created on Thu Oct 18 09:11:53 2018
DEV NOTE 4/3/19: EURAC ABD different than EURAC ground
DEV NOTE 18/10/18: overlaps with crest_roof to be solved

DESCRIPTION
Analyse and compare irradiance data from different sensors.
Used by PhD02

@author: wsfm
"""



""" MODULES """
#importing sys to locate modules in different paths
import sys
#poth to python modules
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_IT_R&D/Python_modules/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)
import sqlalchemy as sqla #import sqla to define variable format 
import pvdata.datamanager.sql as sql #import sql module
import pandas as pd #import pandas for dataframe manipulation
import datetime as dtt #import datetime for time detla
import numpy as np #importing meteodataquality for solar data
import matplotlib.pyplot as plt #importing matplotlib
import pvdata.meteodataquality.meteodataquality as mdq #importing meteodataquality for testing
import pvdata.solarlibraries.pvlibinterface as pvlibint #importing pvlibinterface for interface and solar zenith
import pvdata.dataframeshow.figureshow as fs #standard formatting of graphs 
import pvdata.dataframeshow.chart as chart #DEV NOTE 24/2/19 new version


""" GLOBAL VARIABLES """

PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"

PATH_TO_FOLDER_ROOF=r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/2018-08-24 to 2018-08-30 W roof data/"
FILE_NAME_ROOF="ms2_NL115_CR1000_TableMeasurements_2018-08-25.csv"

PATH_TO_FOLDER_EURAC=r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/EURAC/doc_shared/181018_Lindig/wetransfer-1a4051/"
FILE_NAME_EURAC_2015 = "Weather_2015.txt"
FILE_NAME_EURAC_2016 = "Weather_2016.txt"
#FILE_NAME_EURAC_2016 = "Weather_2016.txt"

PATH_TO_FOLDER_GROUND2018 = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/180703-08_crest_outdoor_measurements_fm/180824-30_fm/"
FILE_NAME_GROUND = "crest_ground_180824-9_fm_tcorrected.csv"
PATH_TO_FOLDER_GROUND2019=r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/191129_CREST_characterisation_outdoor/"


LOCAL_CONNECTION="postgresql+psycopg2://postgres@localhost/research_sources"
LOCAL_SCHEMA = "roof_monitoring"
LOCAL_TABLE= "radiation"
LOCAL_ID_HEADER = "RECNBR"
RADIATION_HEADERS = ["TMSTAMP",
 "RECNBR",
 "pyrheliometer_chp1_01_mms2",
 "pyrheliometer_chp1_02_mms2",	
 "pyro_cmp11_w07",	
 "pyro_cmp11_w08",	
 "pyro_cmp11_w01",	
 "pyrheliometer_chp1_01_pt100",	
 "pyrheliometer_chp1_02_pt100"]
#NOTES FROM TB  
#Pyranometer CMP11_W01 measures global irradiance and is not on a ventilation unit.
#Pyranometer CMP11_W07 measures global irradiance and is on a ventilation unit.
#Pyranometer CMP11_W08 measures diffuse horizontal irradiance and is on a ventilation unit.

LOCAL_SCHEMA_EURAC="eurac"
LOCAL_TABLE_EURAC_CALIBRATIONS="calibrations"
LOCAL_TABLE_EURAC_METEOSTATION="meteostation"


#for importing from csv to local database
SQLA_DTYPES = {"TMSTAMP":sqla.types.TIMESTAMP(timezone=False),
 "RECNBR":sqla.types.INTEGER,
 "pyrheliometer_chp1_01_mms2":sqla.types.FLOAT(2,10),
 "pyrheliometer_chp1_02_mms2":sqla.types.FLOAT(2,10),	
 "pyro_cmp11_w07":sqla.types.FLOAT(2,10),	
 "pyro_cmp11_w08":sqla.types.FLOAT(2,10),	
 "pyro_cmp11_w01":sqla.types.FLOAT(2,10),	
 "pyrheliometer_chp1_01_pt100":sqla.types.FLOAT(2,10),	
 "pyrheliometer_chp1_02_pt100":sqla.types.FLOAT(2,10)}


REMOTE_CONNECTION:str
REMOTE_SCHEMA = "w_meas" #DEV NOTE 5/3/19: no data for pyro_cmp11_w01 in 2015 or 2016
REMOTE_TABLE = "mms2_met"
HEADERS_MMS2_MET = [
  "measurementdatetime",
  "closestfiveseconddatetime",
  "closestfifteenseconddatetime",
  "closestminutedatetime",
  "closesthour",
  "logger_rec_no",
  "pyrheliometer_chp_01_mms2",
  "pyrheliometer_chp_02_mms2",
  "pyro_cmp11_w07",
  "pyro_cmp11_w08",
  "pyro_cmp11_w01",
  "pyrheliometer_chp_01_mms2_voltage",
  "pyrheliometer_chp_02_mms2_voltage",
  "pyro_cmp11_w07_voltage",
  "pyro_cmp11_w08_voltage",
  "pyro_cmp11_w01_voltage"]

""" SETUP """
_DAYS_CREST_ALMOST_CLEAR = [
"2015-06-04",
"2015-06-07",
"2015-06-11",
"2015-06-15",
"2015-06-18",
"2015-06-30",
"2015-07-03",
"2015-09-06",
"2015-09-10", 
"2015-09-11",
"2015-11-21",
"2015-12-23",
"2015-12-29",
"2015-12-31", 
"2016-01-15",
"2016-02-15", 
"2016-02-16",
"2016-02-18", 
"2016-02-24",
"2016-02-25", 
"2016-03-07",
"2016-03-25",
"2016-03-31",
]

_DAYS_CREST_180824_29 = ["2018-08-24","2018-08-25","2018-08-26","2018-08-27","2018-08-28","2018-08-29"]

_MONTHS_ALL = [1,2,3,4,5,6,7,8,9,10,11,12]

#location variables
CREST_GROUND_LATITUDE = 52.763793
CREST_GROUND_LONGITUDE = -1.249695
CREST_GROUND_ALTITUDE = 0
CREST_ROOF_LATITUDE = 52.762463
CREST_ROOF_LONGITUDE = -1.240622 
CREST_ROOF_ALTITUDE = 79

EURAC_ABD_LATITUDE = 46.457778
EURAC_ABD_LONGITUDE = -11.328611
EURAC_ABD_ALTITUDE = 247  
#Italy (46° 27’ 28’’ N, 11° 19’ 43’’ E, 247 meters above sea level)
#From: Wind effect on PV module temperature: Analysis of different techniques for an accurate estimation


#radiometers sensitivities 
CHP_170080_SENSITIVITY = 9.29
# to be consistent, manufacturers value used for other sensors too 
#DEV NOTE 6/11/18: however different calibration years, degradation may be relevant !!
CMP11_174112_SENSITIVITY_KZ = 9.31
CMP11_174112_SENSITIVITY_CREST = 9.331 #Aug 18 calibration
CMP11_174112_SENSITIVITY =  CMP11_174112_SENSITIVITY_KZ
# average annual degradation of -5.43% found, KZ used 
CMP21_090318_SENSITIVITY_KZ =  9.47
CMP21_090318_SENSITIVITY_CREST =  9.431 #Aug 18 calibration
CMP21_090318_SENSITIVITY = CMP21_090318_SENSITIVITY_KZ
# average annual degradation of -2.18% found, KZ used 
CMP11_174113_SENSITIVITY_KZ = 8.79
CMP11_174113_SENSITIVITY_CREST = 8.83 #Aug 18 calibration
CMP11_174113_SENSITIVITY =  CMP11_174113_SENSITIVITY_KZ
# average annual degradation of -4.34% found, KZ used (tilted pyranometer)

CMP11_174113_TILT = 29 
CMP11_174113_AZIMUTH = 180 


""" SETUP """# Setup default variables
RESOLUTION_SECONDS = 60
FILTER_IRRADIANCE_BEAM = 200
#filter irradiance for PR contracts: 35, 50, 200 or none
#DIS/ISO 9060 test non-linearity from 100 to 1000 
FILTER_CLOUD_RATIO = 0.2
#DEV NOTE 4/3/19 graph limit which could be transferred
_XLIM = [None,None]
_YLIM = [None,None]

filter_label="_"+"bi"+str(FILTER_IRRADIANCE_BEAM)+"cr"+str(int(FILTER_CLOUD_RATIO*100))

sources_description = {
"eurac local sql meteo station":"database of eurac irradiance data",
"eurac local sql calibrations":"database of indoor measurements data at eurac",
"crest ground 180824_29":"crest outdoor data with custom made tilted platform",
"180825":"roof data for an almost clear day",
"almost clear days":"roof data for selected almost clear days",
"months": "one year of data",
"eurac abd":"Bolzano airport data"}

processes_description = {
"characterisation test":"eurac characterisation test, no setup data used", #to be checked
"eurac ground 181126_1205 irradiances":"a few days comparison of eurac irradiance ground data with airport data",
"overview eurac calibrations":"plot of outdoor eurac calibration data",
"crest ground 180824_29 error":"calculation of directional error based on outdoor data",
"crest roof 180825 irradiances":"overview of roof irradiance measurements",
"crest roof":"processes for directional error absolute and/or cosine",
"eurac abd error":"directional error absolute or cosine for hour or month"}


df=pd.DataFrame(index=processes_description.keys(),columns=sources_description.keys())
df.loc["eurac ground 181126_1205 irradiances","eurac local sql meteo station"]=True
df.loc["overview eurac calibrations","eurac local sql calibrations"]=True
df.loc["crest ground 180824_29 error","crest ground 180824_29"]=True
df.loc["crest roof",["180825","almost clear days","months"]]=True
df.loc["eurac abd error","eurac abd"]=True


#print("PROCESSES PER SOURCE AVAILABLE") #7/1/20 temporarily hidden
#print(df)


class Setup:
    beam_column: str
    diffuse_column :str
    global_column: str
    tilted_column: str
    time_column:str
    aoi_column: str
    temp_amb_column:str
    latitude: float
    longitude: float
    altitude: float
    days: np.ndarray
    months: np.ndarray
    resolution_second: int
    dataframe: pd.DataFrame    
    filter_irradiance: int
    filter_cloud_ratio: float
    test: bool

    def __init__(self,source:str,
                 filter_irradiance=FILTER_IRRADIANCE_BEAM,
                 filter_cloud_ratio=FILTER_CLOUD_RATIO,
                 test=True,
                 days=_DAYS_CREST_ALMOST_CLEAR,
                 months=_MONTHS_ALL):
        #DEV NOTE 6/11/18: irradiance filter could be already set when importing from database
        self.filter_irradiance = filter_irradiance
        self.filter_cloud_ratio = filter_cloud_ratio
        self.days = days #select days
        self.months = months #select days
        #TEST = True limit processing time by: reducing input (100 rows max)
        self.test = test
        #define no names for key dataframe columns to be defined later when loading the dataframe
        if source=="ground 2019":
            self.diffuse_column="diffuse"
            self.tilted_column="CMP21_Irrd_Avg"
            self.global_column="CMP11_Irrd_Avg"
            self.beam_column="CHP_Irrd_Avg"
            self.time_column="TIMESTAMP"
            self.phase_column="SEVolt_Avg"
            self.temp_sns_column="CMP21_Temp_C_Avg"
            self.latitude = CREST_GROUND_LATITUDE
            self.longitude = CREST_GROUND_LONGITUDE
            self.altitude = CREST_GROUND_ALTITUDE
            self.folder=PATH_TO_FOLDER_GROUND2019
            self.surface_tilt=0
            f191129="191129_00_a"
            f191204="191204_05_1257"
            f191215="191215_5_1231"
            de191129="2019-11-29 14:10:28"
            de191204="2019-12-04 13:27:00"
            de191215="2019-12-15 12:57:19"
            s0="01/11/1991  00:00:00"
            self.files_info=pd.DataFrame({                         
            "folder":[f191129,
                      f191204,f191204,f191204,f191204,
                      f191215,f191215]
            ,"fend":[de191129,
                     de191204,de191204,de191204,de191204,
                     de191215,de191215]
            ,"tilt":[0,
                     70,35,15,5,
                     0,5]
            ,"start":[s0,
                         "2019-12-04 11:20:00","2019-12-04 11:57:00","2019-12-04 12:29:00","2019-12-04 12:57:00",
                         "2019-12-15 12:04:00","2019-12-15 12:31:00"] #12:29 artificial
            ,"modified":[de191129,
                         "2019-12-04 11:52:53","2019-12-04 12:28:57","2019-12-04 12:55:28",de191204,
                         "2019-12-15 12:28:57",de191215]
            },index=[19112900,
                  19120470,19120435,19120415,19120405,
                  19121500,19121505])
                        #"2019-11-29 12:49:23"
            self.dfsout=dict() #7/1/20 good for iterations
              #self.files_info=self.files_info.loc[range(0,6),:]#good for testing
            for index in self.files_info.index: #different tilt data into dictionary
                filename=self.folder+str(self.files_info.loc[index,"folder"])+"/CR1000E15427_Radiometers_fm.csv" 
                #31/12/19 fm version: removed ", separated into columns, seconds added if missed
                df_srs=pd.read_csv(filename,header=1,sep=',',skiprows=[2,3])
                try: #exit if error in file 
                    df_srs[self.time_column]=pd.DatetimeIndex(df_srs[self.time_column])
                except:
                    sys.exit(filename)
                max_old=max(df_srs.loc[:,self.time_column])
                start=self.files_info.loc[index,"start"]
                end=self.files_info.loc[index,"modified"]    
                fend=self.files_info.loc[index,"fend"] 
                self.files_info.loc[index,"dt_max_old"]=max_old
                dt_delta=np.datetime64(fend)-np.datetime64(max_old)            
                df_srs[self.time_column]=df_srs.apply(lambda x: x[self.time_column]+dt_delta,axis=1)#shift datetime for last value to be datetime of last modification
                df_srs=df_srs.loc[(df_srs[self.time_column]>=start)&(df_srs[self.time_column]<=end),:]
                if len(df_srs)==0: sys.exit(filename)
                df_srs[self.phase_column]=df_srs.apply(lambda x: round((x[self.phase_column]*14/2-3)/2,0),axis=1)
                self.dfsout[index]=df_srs #good for iterations


        if source == "crest ground 180824_29":
            self.diffuse_column = "CMP11a_Irrd_Avg"
            self.global_column = "CMP21_Irrd_Avg"
            self.beam_column = "CHP_Irrd_Avg"
            self.time_column = "TMSTAMP"
            self.tilted_column = "CMP11b_Irrd_Avg"            
            self.aoi_column = ""
            self.latitude = CREST_GROUND_LATITUDE
            #DEV NOTE 29/10/18: if necessary redo graph with negative based on Solar Position Algorithm for Solar Radiation Applications
            self.longitude = CREST_GROUND_LONGITUDE
            self.altitude = CREST_GROUND_ALTITUDE
            #Manufacturer sensitivity
            chp_sens = CHP_170080_SENSITIVITY
            cmp11a_sens= CMP11_174112_SENSITIVITY
            cmp21_sens = CMP21_090318_SENSITIVITY
            cmp11b_sens= CMP11_174113_SENSITIVITY
            df = pd.read_csv(PATH_TO_FOLDER_GROUND2018+FILE_NAME_GROUND,header=0,usecols={
                    "TOA5 TIMESTAMP TS","CHP_Irrd_Avg_mV","CMP11a_Irrd_Avg_mV",
                    "CMP21_Irrd_Avg_mV","CMP11b_Irrd_Avg_mV","CMP21_Irrd_Avg_mV_tcrr"
                    })
            #Added test (100 values) 23/2/19                
            df["TOA5 TIMESTAMP TS"]=pd.DatetimeIndex(df["TOA5 TIMESTAMP TS"])
            #time shift 3 seconds equivalent to assuming 5s average measuread at 30 happening at 27 (to match other datasets)
            df["TMSTAMP"]= df.loc[:,"TOA5 TIMESTAMP TS"].apply(lambda x: x-dtt.timedelta(seconds = 2.5))
            df["CHP_Irrd_Avg"]=df["CHP_Irrd_Avg_mV"]/chp_sens*1000
            df["CMP11a_Irrd_Avg"]=df["CMP11a_Irrd_Avg_mV"]/cmp11a_sens*1000
            df["CMP21_Irrd_Avg"]=df["CMP21_Irrd_Avg_mV"]/cmp21_sens*1000
            df["CMP21_Irrd_Avg_tcrr"]=df["CMP21_Irrd_Avg_mV_tcrr"]/cmp21_sens*1000
            #calculate for tilted
            df["CMP11b_Irrd_Avg"]=df["CMP11b_Irrd_Avg_mV"]/cmp11b_sens*1000
            #df["seconds_unit"]=df.loc[:,"TMSTAMP"].apply(lambda x: x.second % 10)
            #df_select=df[(df.seconds_unit == 7)]
            #DEV NOTE: arbitrary days to be implemented
            #df =df[(df["TMSTAMP"]>"25/08/2018 00:00:00")&(df["TMSTAMP"]<"26/08/2018 00:00:00")]
        if "crest roof" in source:
            #DEV NOTE 24/10/18: to be double checked
            self.diffuse_column = "pyro_cmp11_w08"
            self.global_column = "pyro_cmp11_w07" #DEV NOTE 24/10/18: CHP1 not so reliable ? Redo graph
            self.beam_column = "pyrheliometer_chp1_02_mms2"
            self.time_column = "tmstamp"
            self.aoi_column = ""
            self.latitude = CREST_ROOF_LATITUDE
            #DEV NOTE 29/10/18: if necessary redo graph with negative based on Solar Position Algorithm for Solar Radiation Applications
            self.longitude = CREST_ROOF_LONGITUDE
            self.altitude = CREST_ROOF_ALTITUDE
            self.resolution_seconds = RESOLUTION_SECONDS
            #modified version of select_on_list(parameter,table,schema,variable,lista,connection) since day needed 
        
            querystring = "SELECT "
            started = False
            for value in list(HEADERS_MMS2_MET):
                if started == True: 
                    querystring = querystring + ","
                querystring = querystring + "tabella." + value
                started = True
            querystring = querystring + ", date_trunc('day', measurementdatetime) AS day "   
            if "2015" not in source and "days" in source: 
                querystring = querystring + " FROM "+str(REMOTE_SCHEMA)+"."+REMOTE_TABLE+" AS tabella WHERE ("
                started = False   
                for value in list(self.days):
                    if started == True: 
                        querystring = querystring + " OR "
                    querystring = querystring+"date_trunc('day', measurementdatetime)"+"='" + str(value)+"'"
                    started = True                
                querystring = querystring + ") AND extract('second' from measurementdatetime)=0"
                #querystring = querystring + ") AND extract('minute' from measurementdatetime)=0 AND extract('second' from measurementdatetime)=0"
                #read sql
            if "2015" not in source and "months" in source: 
                self.months = months #select days
                querystring = querystring + " FROM "+str(REMOTE_SCHEMA)+"."+REMOTE_TABLE+" AS tabella WHERE ("
                started = False   
                for value in list(self.months):
                    if started == True: 
                        querystring = querystring + " OR "
                    querystring = querystring+"date_part('month', measurementdatetime)"+"='" + str(value)+"'"
                    started = True                
                querystring = querystring + ") AND extract('second' from measurementdatetime)=0"
                #querystring = querystring + ") AND extract('minute' from measurementdatetime)=0 AND extract('second' from measurementdatetime)=0"
                #read sql
            if "2015" in source:
                querystring = (querystring + " FROM "+str(REMOTE_SCHEMA)+"."+REMOTE_TABLE+
                " AS tabella WHERE date_trunc('day', measurementdatetime)>'2014-12-31'"+
                " OR date_trunc('day', measurementdatetime)<'2015-12-31'")            
            start = dtt.datetime.now()
            print(querystring) 
            
            #2/11/20 checking connection
            try:
                df = pd.read_sql(querystring,REMOTE_CONNECTION)   
            except Exception as e:
                import sys
                sys.exit(e)
                    
            print(str(dtt.datetime.now()-start)) 
            #renaming since not working before
            #DEV NOTE: dummy values for pt100
            #DEV NOTE: could be adapted into similar recognizable name through a dictionary
            #rename tmstamp to have same headersof the local ?
            df = df.rename(columns={
            "measurementdatetime":"tmstamp",
            "pyrheliometer_chp_01_mms2":"pyrheliometer_chp1_01_mms2",
            "pyrheliometer_chp_02_mms2":"pyrheliometer_chp1_02_mms2",
            "pyro_cmp11_w07":"pyro_cmp11_w07",
            "pyro_cmp11_w08":"pyro_cmp11_w08",
            "pyro_cmp11_w01":"pyro_cmp11_w01",
            "pyrheliometer_chp_01_mms2_voltage":"pyrheliometer_chp1_01_pt100",
            "pyrheliometer_chp_02_mms2_voltage":"pyrheliometer_chp1_02_pt100"
            })
            if (self.resolution_seconds == 5) & ("crest" in source): 
                #aggregation resolution which could be added to the data management module 
                #DEV NOTE 18/10/18: function to be generalised and put into datamanager ?
                #definite minute for aggregation
                datetime = pd.DatetimeIndex(df["TMSTAMP"])
                df["minute"]=datetime.minute
                #define 5s timestamp for aggregation
                df.loc[:,"TMSTAMP5s"] = df.loc[:,"TMSTAMP"] - df.loc[:,"TMSTAMP"].apply(lambda x: dtt.timedelta(seconds = (x.second % 5)))         
                #filter by 5s aggregation
                df_5s = df.groupby(['TMSTAMP5s'], as_index=True)
                pyrheliometer_chp1_01_mms2 = df_5s["pyrheliometer_chp1_01_mms2"]
                pyrheliometer_chp1_02_mms2 = df_5s["pyrheliometer_chp1_02_mms2"]
                pyro_cmp11_w07 = df_5s["pyro_cmp11_w07"]
                pyro_cmp11_w08 = df_5s["pyro_cmp11_w08"]
                pyro_cmp11_w01 = df_5s["pyro_cmp11_w01"]
                pyrheliometer_chp1_01_pt100 = df_5s["pyrheliometer_chp1_01_pt100"]
                pyrheliometer_chp1_02_pt100 = df_5s["pyrheliometer_chp1_02_pt100"]
                tmstamp = df_5s["TMSTAMP"]
                df  = pd.DataFrame({
                'pyrheliometer_chp1_01_mms2':pyrheliometer_chp1_01_mms2.mean(),
                'pyrheliometer_chp1_02_mms2':pyrheliometer_chp1_02_mms2.mean(),
                'pyro_cmp11_w07':pyro_cmp11_w07.mean(),
                'pyro_cmp11_w08':pyro_cmp11_w08.mean(),
                'pyro_cmp11_w01':pyro_cmp11_w01.mean(),
                'pyrheliometer_chp1_01_pt100':pyrheliometer_chp1_01_pt100.mean(),
                'pyrheliometer_chp1_02_pt100':pyrheliometer_chp1_02_pt100.mean(),
                "tmstamp":tmstamp.max()})

    
        if source == "crest roof 180825":
            #only data for the 180825 august as provided by TB 
            querystring = "SELECT * FROM "+ LOCAL_SCHEMA +"."+LOCAL_TABLE
            start = dtt.datetime.now() 
            print(querystring)  
            df = pd.read_sql(querystring,LOCAL_CONNECTION)
            print(str(dtt.datetime.now()-start))   
            df = df.rename(columns={
            "TMSTAMP":"tmstamp"})   
         
        if "eurac ground 181126_1205" in source:
            #DEV NOTE: to be changed, location is not abd 
            self.aoi_column = ""
            self.latitude = EURAC_LATITUDE
            #DEV NOTE 29/10/18: if necessary redo graph with negative based on Solar Position Algorithm for Solar Radiation Applications
            self.longitude = EURAC_LONGITUDE
            self.altitude = EURAC_ALTITUDE            
            if "calibrations" in source:  
                table = LOCAL_TABLE_EURAC_CALIBRATIONS   
                self.time_column = "tmstamp"
                querystring = "SELECT * FROM "+ LOCAL_SCHEMA_EURAC +"."+table
            elif "meteo station" in source:  
                table = LOCAL_TABLE_EURAC_METEOSTATION   
                self.time_column = "time"
                querystring = (
                'SELECT  m."time",m.id,m.wind_speed,m.wind_direction,m.temperature,m.relative_humidity,'+
                'm.diffuse,m.direct,m.global,m.azimuth,a.global_30 as global_abd,a.beam as beam_abd FROM '+
                'eurac.meteostation as m LEFT JOIN eurac.abd as a ON m."time" = a.time_plus12s'        
                )
            print(querystring)
            start = dtt.datetime.now() 
            df = pd.read_sql(querystring,LOCAL_CONNECTION)
            print(source+" read_sql:"+str(dtt.datetime.now()-start))   
            df = df.rename(columns={
            "Timestamp":self.time_column})    
            df[self.time_column]=pd.DatetimeIndex(df[self.time_column])
            
        if "eurac abd" in source:
            #DEV NOTE 24/10/18 values to be moved into global
            self.beam_column = "CHP1_direct"
            self.diffuse_column = "CMP11_diffuse"
            self.global_column = "CMP11_global_horiz"
            self.time_column = "timestamp"
            self.aoi_column = "aoi"
            self.tilted_column = "CMP11_global_30"    
            self.temp_amb_column = "T_ambient"
            self.latitude = EURAC_ABD_LATITUDE
            self.longitude = EURAC_ABD_LONGITUDE
            self.altitude = EURAC_ABD_ALTITUDE
            #DEV NOTE 24/10/18: to be checked and completed
            if "2015" in  source:            
                filepath_eurac = str(PATH_TO_FOLDER_EURAC)+str(FILE_NAME_EURAC_2015)
            elif "2016" in source:
                filepath_eurac = str(PATH_TO_FOLDER_EURAC)+str(FILE_NAME_EURAC_2016)
            
            df = pd.read_csv(filepath_or_buffer=filepath_eurac,
                             sep=",",
                             header=0)
            dt_series = pd.DatetimeIndex(df["timestamp"])
            df["date"] = dt_series.date
            df["aoi"]=90-df["Altitude"]
            #df2 = df[df.loc[:,"date"].isin(list(self.days))]
            #print(type(list(pd.DatetimeIndex(self.days))))        
        
        if ("eurac ground" not in source) and ("ground 2019" not in source): #31/12/19 to be checked: currently use specific filters for characterisation outdoor" 
            #filter on cloud ratio, originally at 0.3            
            df = df[(df[self.beam_column]>self.filter_irradiance)]
            df["CR"]=df[self.diffuse_column]/df[self.global_column]
            #assign dataframe in the end
            if df.empty == True: sys.exit("Dataframe empty")
            if self.test == True: self.dataframe = df.head(n=1000)
            elif self.test == False: self.dataframe = df
    
""" PROCESSING EXAMPLES """

def save(filename,CFormat:chart.Format,dataframe=None,path_to_output=PATH_TO_OUTPUT,xlim=None,ylim=None):       
    if (xlim is not None) or (ylim is not None):           
        axes = plt.gca()
        if (xlim is not None): axes.set_xlim(xlim)
        if (ylim is not None): axes.set_ylim(ylim)
    plt.show() 
    plt.tight_layout(.5) # DEV NOTE 24/2/19: necessary or not?  
    chart.save(CFormat,filename=filename+filter_label)
    plt.close()
    if dataframe is not None:
        dataframe.d                  
        #explode = (0.1, 0.2, 0.1, 0.1, 0.2, 0.1) #define empty spaces bewteen the pie parts                                 
        #graph = fshw.PieChart('DailyDelayedAnticipateStdC'+str(std_coef),labels,fracs,explode)

def run(process,test=True,path_to_output=PATH_TO_OUTPUT,markersize=None,
        ymin=_YLIM[0],ymax=_YLIM[1],xmin=_XLIM[0],xmax=_XLIM[1]):
    CFormat=chart.Format(path_to_output=path_to_output) #add general formatting for all outputs  
    if markersize is None: markersize=CFormat.markersize
    elif markersize is int:markersize=markersize
    
    
    if process=="crest characterisation ground 2019":
        setup = Setup(source="ground 2019",
        test=test)                 
        #outdoor characterisation parameters shared 
        location=pvlibint.SolarLibrary(latitude=setup.latitude,
                                       longitude=setup.longitude,
                                       altitude=setup.altitude,
                                       surface_tilt=setup.surface_tilt) 
        
        print("latitude:"+str(setup.latitude)+" "+
              "longitude:"+str(setup.longitude)+" "+
              "altitude:"+str(setup.altitude)+" "+
              "surface_tilt:"+str(setup.surface_tilt))
        
        #tilt to be accounted
        
        resolutions=[1] 
        periods=[60]
        #resolutions=[60] 
        #periods=[1800]
        counts=21
        irr_min=100 #100 for calibration in cloudy conditions 9847
        aoi_max=95 #characterisation (not used anyway)
        kcs_range= [0.7,1.3] #a bit cloudy increasing  
        # 0.2: uncertainty 95% of HOURLY values for moderate quality (class C) radiometers [WMO 2017) @fm: enough due to additional filter and uncertainty of cs modelling
        # 0.3 added for 04/12/19 
        kcs_cv_max= 0.04 #increased from 0.02 of calibration since it may include 60 values
        pwr_cv_max=kcs_cv_max
        pearson_min= 0.8    
        deviation_max= 0.02
        
        
               
        #iteration among the files 
        df_results=pd.DataFrame()
        
        for index in setup.files_info.index:
            print(index)
            df_tmp=setup.dfsout[index]
            df_tmp.loc[:,setup.time_column]=pd.DatetimeIndex(df_tmp.loc[:,setup.time_column],ambiguous='NaN')
            df_tmp=df_tmp.dropna(subset=[setup.time_column])#drop na for datetime to avoid grouping problems later
            datetimeindex_utc=df_tmp.loc[:,setup.time_column].values
            hemispherical_values=df_tmp.loc[:,setup.tilted_column].values
            temperature_values=df_tmp.loc[:,setup.temp_sns_column].values
            beam_values=df_tmp.loc[:,setup.beam_column].values
            #global_values=df_tmp.loc[:,setup.global_column].values
            phase_values=df_tmp.loc[:,setup.phase_column].values
            tilt=setup.files_info.loc[index,"tilt"]
            count_characterisation=21
            #print(index)            
            df_results_tmp,df_stb,dfsout_stb=mdq.SolarData.characterisation_outdoor_alternating(location=location,
                                 datetimeindex_utc=datetimeindex_utc, #timezone:str,
                                 hemispherical_values=hemispherical_values,
                                 temperature_values=temperature_values, #temperature of sensor for irradiance correction also for pyrheliometer?
                                 beam_values=beam_values, 
                                 phase_values=phase_values,
                                 resolutions=resolutions,
                                 periods=periods,
                                 counts=counts,
                                 deviation_max=deviation_max,
                                 tilt=tilt,
                                 irr_min=irr_min,
                                 aoi_max=aoi_max,
                                 kcs_range=kcs_range,
                                 kcs_cv_max=kcs_cv_max, #used also for the power
                                 pwr_cv_max=pwr_cv_max,
                                 pearson_min=pearson_min,
                                 count_characterisation=count_characterisation,
                                 keep_invalid=True)
            if len(df_results_tmp)!=0:
                df_results=df_results.append(df_results_tmp)
            df_stb.to_csv(path_or_buf=path_to_output+str(index)+"_stb"+".csv")
        
        
        #filename= str("chr_")+str(setup.files_info.loc[index,"folder"])
        filename="characterisation"  
        
        df_results.to_csv(path_or_buf=path_to_output+filename+".csv")
        
        #save(filename,CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax])
            
        
    
    if process == "characterisation test":
        df = pd.DataFrame({
            'serial':[60011,90318,174112,174113],
            'sensitivity':[8.35,9.431230763,9.330519382,8.81646163],
            'position':[0,1,2,3],
            'channel':[4,5,6,7],
            'voltage':[7.374892391	,8.023208362,8.547420543,7.955149851]})
        lshd=150 #sun shades lenght  
        dmin=7649-68 #distance from light
        df.loc[:,"distance"]=df.loc[:,"position"].apply(lambda x:x*lshd)
        df.loc[:,"irradiance"]=df.voltage/df.sensitivity
        
        print(df)
        
        #.apply(lambda x:x["voltage"]/x["sensitivity"])
        from scipy.optimize import fsolve
        
        df_results=pd.DataFrame(columns=['dr','dt','ar','at'])     
        for serial in list(df.loc[:,"serial"].values):
            if serial != 60011:       
                drt=df.loc[df.serial==serial,"distance"].values[0]
                frt= (df.loc[df.serial==serial,"irradiance"].values[0]/
                df.loc[df.serial==60011,"irradiance"].values[0])
                def function(z):
                    dr, dt, ar, at = z
                    F = (dr*np.cos(ar)-dmin,
                         dt*np.cos(at)-dmin,
                         dt*np.sin(ar)-dr*np.sin(ar)-drt,
                         (np.cos(at)*dr**2)/(np.cos(ar)*dt**2)-frt)
                    return F
                #definition of guess
                zGuess = [dmin,dmin,-np.pi*5/180,+np.pi*5/180,]
                #solve functions 
                results = fsolve(function,zGuess)
                
                df_results=df_results.append({
                "serial":serial,
                "dr":results[0],
                "dt":results[1],
                "ar":results[2],
                "at":results[3]},ignore_index=True)                
        
  
    if process == "eurac ground 181126_1205 irradiances": #compare ground eurac data with abd
        #Note: no comparison function needed, only setup used          
        #setting options for visualisation 
        pd.set_option('display.max_columns',20)
        pd.set_option('display.max_rows',100)
        FREQUENCY = "h"
        setup = Setup(source="eurac local sql meteo station",
        test=test)         
        df = setup.dataframe #recovering dataframe from setup 
        #definition of locations depending on chosen setup 
        location = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)        
        d0 = ["26/11/2018 0:00:00","29/11/2018 0:00:00"]        
        d1 = ["26/11/2018 0:00:00","27/11/2018 0:00:00"]
        d2 = ["27/11/2018 0:00:00","28/11/2018 0:00:00"]
        d3 =["28/11/2018 0:00:00","29/11/2018 0:00:00"]     
        df.loc[:,setup.time_column] = pd.DatetimeIndex(df.loc[:,setup.time_column])  
        #d = d0
        #df = df[(df[setup.time_column]>d[0])&(df[setup.time_column]<d[1])]    
        if df.empty == True:
            sys.exit("empty df")
        t_format='%H'#formatting for better late visualisation
        timeseries = pd.DatetimeIndex(df.loc[:,setup.time_column],format=t_format)
        timeseries_shifted =  pd.DatetimeIndex(df.loc[:,setup.time_column].apply(lambda x: x+np.timedelta64(1,'h')),format=t_format)
        lt = location.getlinketurbidity(timeseries)
        cs = location.getirradianceclearskyhorizontal(timeseries,lt)
        p1,=chart.scatter(CFormat,timeseries_shifted,df["direct"],marker=".y",label="Beam irradiance razon")
        p2,=chart.scatter(CFormat,timeseries_shifted,df["diffuse"],marker=".r",label="Diffuse irradiance razon")
        p3,=chart.scatter(CFormat,timeseries_shifted,df["global"],marker=".b",label="Global irradiance razon")
        p4,=chart.scatter(CFormat,timeseries,cs["dhi"],marker=".k",label="Diffuse irradiance mod.")
        p5,=chart.scatter(CFormat,timeseries,cs["ghi"],marker=".g",label="Global irradiance mod.")
        p6,=chart.scatter(CFormat,timeseries,cs["dni"],marker=".m",label="Beam irradiance mod.")
        p7,=chart.scatter(CFormat,timeseries,df["global_abd"],marker="xc",label="Global tilt irradiance abd")
        p8,=chart.scatter(CFormat,timeseries,df["beam_abd"],marker="xr",label="Beam irradiance abd")
        #definition of intervals and extreme for x
        xticks = pd.date_range(start=min(df.loc[:,setup.time_column].values),
        end=max(df.loc[:,setup.time_column].values),freq=FREQUENCY)
        #plotting
        plt.xticks(rotation='vertical')        
        plt.show()       
        filename= "eurac_ground_irradiances"         
        save(filename,CFormat,dataframe=df,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax])
    
    if process == "overview eurac calibrations":
        #DEV NOTE 4/3/19: SOURCE NOT INCLUDED IN SETUP!!
        #Note: no comparison function needed, only setup used 
        FREQUENCY = "h"
        setup = Setup(source="eurac local sql calibrations",
        test=test)
        #recovering dataframe from setup 
        df = setup.dataframe
        #definition of locations depending on chosen setup 
        location = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)
        p1,=plt.plot(df[setup.time_column],df["TC0"],".y")
        p2,=plt.plot(df[setup.time_column],df["TC1"],".r")
        p3,=plt.plot(df[setup.time_column],df["TC2"],".b")
        p4,=plt.plot(df[setup.time_column],df["TC3"],".k")
        p5=plt.plot(df[setup.time_column],df["TC4"],".g")
        p5=plt.plot(df[setup.time_column],df["TC5"],".m")
        #definition of intervals and extreme for x
        xticks = pd.date_range(start=min(df.loc[:,setup.time_column].values),
        end=max(df.loc[:,setup.time_column].values),freq=FREQUENCY)
        #plotting
        
    
        
    if ("crest ground 180824_29 error" in process):      
        setup = Setup(source="crest ground 180824_29",days=_DAYS_CREST_180824_29,test=test)
        #recovering dataframe from setup 
        df = setup.dataframe
        #definition of locations depending on chosen setup 
        target_CMP21 = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)
        #definition of adjusted aoi for the CMP11 (on the solar tracker) 
        df.loc[:,"AOI tilted"]= target_CMP21.getsolarseries(
        datetimeindex_utc=pd.DatetimeIndex(df.loc[:,setup.time_column]),
        outputs="zenith")  
        #df.loc[:,"AOI tilted"]= target_CMP21.getsolarzenithangle(df.loc[:,setup.time_column])
        df.loc[:,"AOI tilted"] = df.loc[:,"AOI tilted"].apply(lambda x: x-CMP11_174113_TILT)        #first calculating and plotting for the horizontal one 
        
        
        df1= mdq.SolarData.irradiance_cosine_error(        
        location=target_CMP21,
        global_series=df.loc[:,setup.global_column],
        beam_series=df.loc[:,setup.beam_column],
        diffuse_series=df.loc[:,setup.diffuse_column],
        time_series=df.loc[:,setup.time_column],
        aoi_series=[])
        
        label_suffix1=" horizontal pyranometer"
        #then calculating and plotting for the tilted one or corrected temperature        .
        if "tcorrected" not in process: 
            global_column=setup.tilted_column
            aoi_series=df.loc[:,"AOI tilted"]
        elif "tcorrected" in process: 
            global_column="CMP21_Irrd_Avg_tcrr"        
            aoi_series=[]
            
        df2=mdq.SolarData.irradiance_cosine_error(        
        location=target_CMP21,
        global_series=df.loc[:,setup.tilted_column],
        beam_series=df.loc[:,setup.beam_column],
        diffuse_series=df.loc[:,setup.diffuse_column],
        time_series=df.loc[:,setup.time_column],
        aoi_series=aoi_series)
        
        if "tcorrected" not in process: 
            global_column=setup.tilted_column
            label_suffix2=" tilted pyranometer " + str(CMP11_174113_TILT)
        elif "tcorrected" in process: 
            global_column="CMP21_Irrd_Avg_tcrr" 
            label_suffix2=" horizontal pyranometer temperature corrected"
        #merging results
        df_results = df1.merge(df2,
        left_on=["datetime","hour","beam","diffuse"],
        right_on=["datetime","hour","beam","diffuse"],
        how="outer",
        suffixes=["horizontal pyranometer","tilted pyranometer"])
        if "tcorrected" not in process:  filename0= "crest_ground_180824_29_error_" #general label
        if "tcorrected" in process:  filename0= "crest_ground_180824_29_error_tcorrected_" #general label
    
        if "absolute" in process:            
            f = plt.figure(3)                       
            f.add_subplot(111) 
            label1 = "absolute error before 12:00 "+label_suffix1
            p1,=chart.scatter(CFormat,df1[df1.hour<12]["aoi"],
            df1[df1.hour<12]["abs_err"],label1,marker="^")
            label2 = "absolute error after 12:00 "+label_suffix1
            p2,=chart.scatter(CFormat,df1[df1.hour>12]["aoi"],
            df1[df1.hour>12]["abs_err"],label2,marker="v")  
            label1 = "absolute error before 12:00 "+label_suffix2
            p3,=chart.scatter(CFormat,df2[df2.hour<12]["aoi"],
            df2[df2.hour<12]["abs_err"],label1,marker="^")
            label2 = "absolute error after 12:00 "+label_suffix2
            p4,=chart.scatter(CFormat,df2[df2.hour>12]["aoi"],
            df2[df2.hour>12]["abs_err"],label2,marker="v")  
            axes = plt.gca()
            #axes.set_xlim([xmin,xmax])
            #axes.set_ylim([-40,80])
            plt.xlabel("angle of incidence [degree]")
            plt.ylabel("absolute error [W/m2]")
            df_results=df1
            filename=str(filename0)+"abs"
            if "cosine" in process: ylim_abs=[ymin[0],ymax[0]]
            elif "cosine" not in process: ylim_abs=[ymin,ymax]
            save(filename,CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=ylim_abs)
        
        if "cosine" in process:            
            f = plt.figure(3)                       
            f.add_subplot(111) 
            label1 = "cosine error before 12:00 "+label_suffix1
            p1,=chart.scatter(CFormat,df1[df1.hour<12]["aoi"],
            df1[df1.hour<12]["cos_err"],label1,marker="^")
            label2 = "cosine error after 12:00 "+label_suffix1
            p2,=chart.scatter(CFormat,df1[df1.hour>12]["aoi"],
            df1[df1.hour>12]["cos_err"],label2,marker="v")  
            label1 = "cosine error before 12:00 "+label_suffix2
            p3,=chart.scatter(CFormat,df2[df2.hour<12]["aoi"],
            df2[df2.hour<12]["cos_err"],label1,marker="^")
            label2 = "cosine error after 12:00 "+label_suffix2
            p4,=chart.scatter(CFormat,df2[df2.hour>12]["aoi"],
            df2[df2.hour>12]["cos_err"],label2,marker="v")  
            plt.xlabel("angle of incidence [degree]")
            plt.ylabel("cosine error [%]")
            filename=str(filename0)+"cos"
            if "absolute" in process: ylim_cos=[ymin[0],ymax[0]]
            elif "absolute" not in process: ylim_cos=[ymin,ymax]
            save(filename,CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=ylim_cos)

    if "crest roof 180825 irradiances" in process:
        #plot all irradiance measurements from the roof.
        #Note: no comparison function needed, only setup used 
        FREQUENCY = "h"
        setup = Setup(source="crest roof 180825",test=test)
        #recovering dataframe from setup 
        df = setup.dataframe
        #definition of locations depending on chosen setup 
        location = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)    
        t_format='%H'#formatting for better late visualisation
        df["tmstamp"]=pd.DatetimeIndex(df["tmstamp"],format=t_format)        
        p1,=chart.scatter(CFormat,df["tmstamp"],df["pyrheliometer_chp1_01_mms2"],label='pyrheliometer 1',marker=".y")
        p2,=chart.scatter(CFormat,df["tmstamp"],df["pyrheliometer_chp1_02_mms2"],label='pyrheliometer 2',marker=".r")
        p3,=chart.scatter(CFormat,df["tmstamp"],df["pyro_cmp11_w07"],label='secondary standard 7',marker=".b")
        p4,=chart.scatter(CFormat,df["tmstamp"],df["pyro_cmp11_w08"],label='secondary standard 8',marker=".k")
        p5=chart.scatter(CFormat,df["tmstamp"],df["pyro_cmp11_w01"],label='secondary standard 1',marker=".g")
        #definition of intervals and extreme for x
        xticks = pd.date_range(start=min(df.loc[:,"tmstamp"].values),end=max(df.loc[:,"tmstamp"].values),freq=FREQUENCY)
        plt.xticks(rotation='vertical') 
        plt.xlabel("hour")
        plt.ylabel("irradiance [W/m2]")        
        filename='crest_roof_180825_irradiances'
        df_results=df
        save(filename,CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax])
        #plotting        
        
    if ("irradiances" not in process) and ("crest roof" in process):
        if "180825" in process:
            setup = Setup(source="crest roof 180825",test=test)  
            filename0 = "crest_roof_180825"
        elif "almost clear days" in process:
            setup = Setup(source="crest roof days",days=_DAYS_CREST_ALMOST_CLEAR,test=test)
            filename0 = "crest_roof_almostclear"               
        if "months" in process:
            filename0 = "crest_roof_months"
            if "months all" in process:
                setup = Setup(source="crest roof months",months=_MONTHS_ALL,test=test)
                filename0=filename0+"_all"
            if "months all" not in process:   
                months = []
                if "May" in process: months.append(5) 
                if "October" in process: months.append(10)
                setup = Setup(source="crest roof months",months=months,test=test)
                for value in list(months): filename0=filename0+"_"+str(value)                       
        #recovering dataframe from setup 
        df= setup.dataframe
        #definition of locations depending on chosen setup 
        location = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)
        df["AOI"] = location.getsolarzenithangle(df["tmstamp"])
        #calculation of values necessary for directional response
        #horizontal components from the pyrheliometers        
        label_suffix1="ventilated horizontal pyranometer"
        df1= mdq.SolarData.irradiance_cosine_error(        
        location=location,
        global_series=df.loc[:,"pyro_cmp11_w07"],
        beam_series=df.loc[:,"pyrheliometer_chp1_02_mms2"],
        diffuse_series=df.loc[:,"pyro_cmp11_w08"],
        time_series=df.loc[:,"tmstamp"],
        aoi_series=df.loc[:,"AOI"])
        
        if "180825" in process: #only locally found data for pyro_cmp11_w08
            label_suffix2="horizontal pyranometer"
            df2= mdq.SolarData.irradiance_cosine_error(        
            location=location,
            global_series=df.loc[:,"pyro_cmp11_w01"],
            beam_series=df.loc[:,"pyrheliometer_chp1_02_mms2"],
            diffuse_series=df.loc[:,"pyro_cmp11_w08"],
            time_series=df.loc[:,"tmstamp"],
            aoi_series=df.loc[:,"AOI"])
            #merging results
            df_results = df1.merge(df2,
            left_on=["time","hour","beam","diffuse","month"],
            right_on=["time","hour","beam","diffuse","month"],
            how="outer",suffixes=[" ventilated"," nonvent"])
        elif "180825" not in process: df_results=df1 
            
       
        if "absolute" in process:
            #ventilated pyranometer
            if "hour" in process: 
                label1 = label_suffix1+" "+"absolute error before 12:00"
                p1,=chart.scatter(CFormat,df1[df1.hour<12]["aoi"],
                df1[df1.hour<12]["abs_err"],label1,marker="^",markersize=markersize)
                label2 = label_suffix1+" "+"absolute error after 12:00"
                p2,=chart.scatter(CFormat,df1[df1.hour>12]["aoi"],
                df1[df1.hour>12]["abs_err"],label2,marker="v",markersize=markersize) 
                if "180825" in process:
                    #non-ventilated pyranometer
                    label1 = label_suffix2+" "+"absolute error before 12:00"
                    p3,=chart.scatter(CFormat,df2[df2.hour<12]["aoi"],
                    df2[df2.hour<12]["abs_err"],label1,marker="^",markersize=markersize)
                    label2 = label_suffix2+" "+"absolute error after 12:00"
                    p4,=chart.scatter(CFormat,df2[df2.hour>12]["aoi"],
                    df2[df2.hour>12]["abs_err"],label2,marker="v",markersize=markersize) 
            if "month" in process:
                for m in list(setup.months):
                #for m in range(1,13):
                    if df1[df1.month==m].empty==False:
                        label1 = label_suffix1+" "+dtt.date(1900, m, 1).strftime('%B')
                        p1,=chart.scatter(CFormat,df1[df1.month==m]["aoi"],
                        df1[df1.month==m]["abs_err"],label1,marker=".",markersize=markersize)
                        if "180825" in process:
                            label2 = label_suffix2+" "+dtt.date(1900, m, 1).strftime('%B')
                            p2,=chart.scatter(CFormat,df2[df2.month==m]["aoi"],
                            df2[df2.month==m]["abs_err"],label2,marker=".",markersize=markersize) 
                #axes = plt.gca()
                #axes.set_ylim([-50,50])  
            plt.xlabel("angle of incidence [degree]")
            plt.ylabel("absolute error [W/m2]")
            filename = filename0 +"_abs"  
            save(filename,CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax]) #save only image 
     
        if "cosine" in process:     
            #ventilated pyranometer
            if "hour" in process:
                label1 = label_suffix1+" "+"cosine error before 12:00"
                p1,=chart.scatter(CFormat,df1[df1.hour<12]["aoi"],
                df1[df1.hour<12]["cos_err"],label1,marker="^",markersize=markersize)
                label2 = label_suffix1+" "+"cosine error after 12:00"
                p2,=chart.scatter(CFormat,df1[df1.hour>12]["aoi"],
                df1[df1.hour>12]["cos_err"],label2,marker="v",markersize=markersize) 
                if "180825" in process:
                    #non-ventilated pyranometer
                    label1 = label_suffix2+" "+"cosine error before 12:00"
                    p3,=chart.scatter(CFormat,df2[df2.hour<12]["aoi"],
                    df2[df2.hour<12]["cos_err"],label1,marker="^",markersize=markersize)
                    label2 = label_suffix2+" "+"cosine error after 12:00"
                    p4,=chart.scatter(CFormat,df2[df2.hour>12]["aoi"],
                    df2[df2.hour>12]["cos_err"],label2,marker="v",markersize=markersize) 
                plt.ylabel('cosine error [%]')  
            if "month" in process:
                for m in list(setup.months):
                    if df1[df1.month==m].empty==False:   
                        label1 = label_suffix1+" "+dtt.date(1900, m, 1).strftime('%B')
                        p1,=chart.scatter(CFormat,df1[df1.month==m]["aoi"],
                        df1[df1.month==m]["cos_err"],label1,marker=".",markersize=markersize)
                        if "180825" in process:
                            label2 = label_suffix2+" "+dtt.date(1900, m, 1).strftime('%B')
                            p2,=chart.scatter(CFormat,df2[df2.month==m]["aoi"],
                            df2[df2.month==m]["cos_err"],label2,marker=".",markersize=markersize)  
            #axes = plt.gca()
            plt.xlabel("angle of incidence [degree]")
            plt.ylabel("cosine error [%]")
            filename = filename0 +"_cos"
            #DEV NOTE 4/3/19: old version mistake not using beam on plane 
            save(filename,CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax]) #save both image and file   

    if "eurac abd error" in process :
        #FM updated on 3/3/19
        #aggregate directional error from tilted and horizontal pyranometer 
        setup = Setup(source=process,test=test)
        #recovering dataframe from setup 
        df = setup.dataframe
        #filter dataframe for positive beam irradiation
        df = df[(df[setup.beam_column]>0)]                
        #definition of locations depending on chosen setup 
        target_CMP11 = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)
        target_CMP11_tilted = target_CMP11
        target_CMP11_tilted.surface_tilt = 30 
        #definition of adjusted aoi for the tilted ABD and new filter   
        df["AOI tilted"]= target_CMP11_tilted.getangleofincidence(df[setup.time_column]) 
       
        
        #first calculating and plotting for the horizontal one 
        df1 = mdq.SolarData.irradiance_cosine_error(        
        location=target_CMP11,
        global_series=df[setup.global_column],
        beam_series=df[setup.beam_column],
        diffuse_series=df[setup.diffuse_column],
        time_series=df[setup.time_column],
        aoi_series=[]
        )

        #then calculating and plotting for the tilted one. 
        df2 = mdq.SolarData.irradiance_cosine_error(        
            location=target_CMP11_tilted,
            global_series=df[setup.tilted_column],
            beam_series=df[setup.beam_column],
            diffuse_series=df[setup.diffuse_column],
            time_series=df[setup.time_column],
            aoi_series=df.loc[:,"AOI tilted"]
            )
                       
        df_results = df1.merge(df2,
          left_on=["time","hour","beam","diffuse"],
          right_on=["time","hour","beam","diffuse"],
          how="outer",
          suffixes=[" CMP11"," CMP11 tilt 30"])

        label_suffix1="horizontal pyranometer"
        label_suffix2="tilted pyranometer" + " " + str(CMP11_174113_TILT) 
        if "2015" in process: filename= "eurac_2015_error"
        elif "2016" in process: filename= "eurac_2016_error"
                
        
        if "absolute" in process:
            
            f = plt.figure(3)                      
            f.add_subplot(111)            
            if "hour" in process:
                label1 = "absolute error before 12:00"+" "+label_suffix1
                p1,=chart.scatter(CFormat,df1[df1.hour<12]["aoi"],
                df1[df1.hour<12]["abs_err"],label1,marker="^",markersize=markersize)
                label2 = "absolute error after 12:00"+" "+label_suffix1
                p2,=chart.scatter(CFormat,df1[df1.hour>12]["aoi"],
                df1[df1.hour>12]["abs_err"],label2,marker="v",markersize=markersize)  
                label1 = "absolute error before 12:00"+" "+label_suffix2
                p3,=chart.scatter(CFormat,df2[df2.hour<12]["aoi"],
                df2[df2.hour<12]["abs_err"],label1,marker="^",markersize=markersize)
                label2 = "absolute error after 12:00"+" "+label_suffix2
                p4,=chart.scatter(CFormat,df2[df2.hour>12]["aoi"],
                df2[df2.hour>12]["abs_err"],label2,marker="v",markersize=markersize)  
            if "month" in process:                
                for m in list(setup.months):
                    if df1[df1.month==m].empty==False:
                        label = label_suffix1+" "+dtt.date(1900, m, 1).strftime('%B')
                        p,=chart.scatter(CFormat,df1[df1.month==m]["aoi"],
                        df1[df1.month==m]["abs_err"],label,marker=".",markersize=markersize)        
            #axes = plt.gca()
            #axes.set_ylim([ymin,ymax])
            plt.xlabel("angle of incidence [degree]")
            plt.ylabel("absolute error [W/m2]")
            save(filename+"_abs",CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax])#save both image and file
        
     
        if "cosine" in process:
            f = plt.figure(3)                       
            f.add_subplot(111) 
            if "hour" in process:
                label1 = "cosine error before 12:00"+" "+label_suffix1
                p1,=chart.scatter(CFormat,df1[df1.hour<12]["aoi"],
                df1[df1.hour<12]["cos_err"],label1,marker="^",markersize=markersize)
                label2 = "cosine error after 12:00"+" "+label_suffix1
                p2,=chart.scatter(CFormat,df1[df1.hour>12]["aoi"],
                df1[df1.hour>12]["cos_err"],label2,marker="v",markersize=markersize)  
                label1 = "cosine error before 12:00"+" "+label_suffix2
                p3,=chart.scatter(CFormat,df2[df2.hour<12]["aoi"],
                df2[df2.hour<12]["cos_err"],label1,marker="^",markersize=markersize)
                label2 = "cosine error after 12:00"+" "+label_suffix2
                p4,=chart.scatter(CFormat,df2[df2.hour>12]["aoi"],
                df2[df2.hour>12]["cos_err"],label2,marker="v",markersize=markersize) 
            if "month" in process:
                for m in list(setup.months):
                    if df1[df1.month==m].empty==False:
                        label = label_suffix1+" "+dtt.date(1900, m, 1).strftime('%B')
                        p,=chart.scatter(CFormat,df1[df1.month==m]["aoi"],
                        df1[df1.month==m]["cos_err"],label,marker=".",markersize=markersize)                
            #axes = plt.gca()
            plt.xlabel("angle of incidence [degree]")
            plt.ylabel("cosine error [%]")
            save(filename+"_cos",CFormat,dataframe=df_results,path_to_output=path_to_output,xlim=[xmin,xmax],ylim=[ymin,ymax])#save both image and file        
    

    return df_results
    
    


#testing 2/11/20
run(process="crest roof 180825 irradiances")

