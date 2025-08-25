# -*- coding: utf-8 -*-
"""
HISTORY
Created on Wed Oct 10 09:21:29 2018
@author: wsfm
DEV NOTE: interesting for automatic import (better check existing?) and prefilte of data
DEV NOTE 18/10/18: overlaps with irradinace comparison to be avoided
"""

""" MODULES """
#importing sys to locate modules in different paths
import sys
#poth to python modules
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_IT_R&D/Python_modules/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)
#import sqla to define variable format 
import sqlalchemy as sqla
#import sql module
import datamanager.sql as sql
#import pandas for dataframe manipulation
import pandas as pd 
#import datetime for time detla
import datetime as dtt
#importing pvlibinterface for solar zenith
import pvdata.solarlibraries.pvlibinterface as pvlibint
#importing meteodataquality for solar data
import numpy as np
#importing matplotlib
import matplotlib.pyplot as plt
#importing meteodataquality for testing
import meteodataquality.meteodataquality as mdq



#parameter for pyranometer measurements 
#position CREST (temporary) ground facility, west from Greenwich 
#Note that M is measured eastward from north.
LATITUDE= 52.762463
#DEV NOTE 29/10/18: if necessary redo graph with negative based on Solar Position Algorithm for Solar Radiation Applications
LONGITUDE = -1.240622
ALTITUDE = 79

crest_location = pvlibint.SolarLibrary(latitude = LATITUDE, longitude = LONGITUDE, altitude = ALTITUDE)



#generic formulation to extract more than one file
#FILES_LIST = [f for f in os.listdir(PATH_TO_FOLDER) if (f.endswith(".csv") & f.startswith(FILES_NAME))]
#path to output
PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"



"""GLOBAL VARIABLES """

PATH_TO_FOLDER = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/2018-08-24 to 2018-08-30 W roof data/"
#files list: pyranometer file only
FILE_NAME = r"test.csv"
FILE_NAME = r"mms2_NL115_CR1000_TableMeasurements_2018-08-25.csv"

PATH_TO_FOLDER_SOLYS =  r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/crest_outdoor_measurements_fm/180824-30_fm/"
FILE_NAME_SOLYS = r"Solys2180412-sunlog-20180829095411-all_181008_fm.csv"

FILE_NAME_GROUND = r"crest_outdoor_180824-9_fm.csv"


LOCAL_CONNECTION="postgresql+psycopg2://postgres@localhost/research_sources" 
LOCAL_SCHEMA = "roof_monitoring"
LOCAL_TABLE = "radiation"
LOCAL_ID_HEADER = "RECNBR"
LOCAL_ID_HEADER_SOLYS = "TMSTAMP"

REMOTE_CONNECTION="postgresql://python_program:P9th0n@crest-vm-ib1/APVDBV3"
REMOTE_SCHEMA = "w_meas"
REMOTE_TABLE = "mms2_met"
REMOTE_HEADERS = [
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





#clear sky list as extracted from a previous study
CLEARSKY_ALL = [
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

CLEARSKY_JUN= ["2015-06-04","2015-06-07","2015-06-11","2015-06-15","2015-06-18","2015-06-30"]
CLEARSKY_JUL = ["2015-07-03"]
CLEARSKY_SEP = ["2015-09-06","2015-09-10","2015-09-11"]
CLEARSKY_NOV = ["2015-11-21"]
CLEARSKY_DEC = ["2015-12-23","2015-12-29","2015-12-31"]
CLEARSKY_JAN = ["2016-01-15"]
CLEARSKY_FEB = ["2016-02-15","2016-02-16","2016-02-18","2016-02-24","2016-02-25"]
CLEARSKY_MAR = ["2016-03-07","2016-03-25","2016-03-31"]
 

CLEARSKY = CLEARSKY_ALL


#not needed 
LOCAL_HEADERS = ["TMSTAMP",
                     "RECNBR",
                     "pyrheliometer_chp1_01_mms2",
                     "heliometer_chp1_02_mms2",	
                     "pyro_cmp11_w07",	
                     "pyro_cmp11_w08",	
                     "pyro_cmp11_w01",	
                     "pyrheliometer_chp1_01_pt100",	
                     "pyrheliometer_chp1_02_pt100"]


        
SQLA_DTYPES = {"TMSTAMP":sqla.types.TIMESTAMP(timezone=False),
             "RECNBR":sqla.types.INTEGER,
             "pyrheliometer_chp1_01_mms2":sqla.types.FLOAT(2,10),
             "pyrheliometer_chp1_02_mms2":sqla.types.FLOAT(2,10),	
             "pyro_cmp11_w07":sqla.types.FLOAT(2,10),	
             "pyro_cmp11_w08":sqla.types.FLOAT(2,10),	
             "pyro_cmp11_w01":sqla.types.FLOAT(2,10),	
             "pyrheliometer_chp1_01_pt100":sqla.types.FLOAT(2,10),	
             "pyrheliometer_chp1_02_pt100":sqla.types.FLOAT(2,10)}


LOCAL_HEADERS_SOLYS = ['TMSTAMP',
             'step_seconds',
             'Intensity',
             'Error size',
             'Errod dir',
             'Azimuth',
             'Zenith'], 


SQLA_DTYPES_SOLYS = {"TMSTAMP":sqla.types.TIMESTAMP(timezone=False),
 "step_seconds":sqla.types.INTEGER,
 "Intensity":sqla.types.FLOAT(2,10),
 "Error size":sqla.types.FLOAT(2,10),	
 "Errod dir":sqla.types.FLOAT(2,10),	
 "Azimuth":sqla.types.FLOAT(2,10),	
 "Zenith":sqla.types.FLOAT(2,10)}


""" TOOL SETUP """
#source = "remote sql"
source = "local sql"
#souce = "" 

filter_str = "cr30 irr50"
#filter_str = "irr50"
filter_str = "cr30 irr50 25/08/18"


resolution = "3600s"
resolution = "60s"
resolution = ""


results_str = "overview"
results_str = "absolute error"
results_str = "relative error"
#results_str = ""


#Pyranometer CMP11_W01 measures global irradiance and is not on a ventilation unit.
#Pyranometer CMP11_W07 measures global irradiance and is on a ventilation unit.
#Pyranometer CMP11_W08 measures diffuse horizontal irradiance and is on a ventilation unit.





""" FUNCTIONS """







def copy_roof_csv_to_sql():
    #time function to transfer CSV file (received by TB) files into local database 
    #file example: mms2_NL115_CR1000_TableMeasurements_2018-08-24
    dataframe = pd.read_csv(PATH_TO_FOLDER+FILE_NAME,header=1)
    dataframe["TMSTAMP"]=pd.DatetimeIndex(dataframe["TMSTAMP"])
    #,format="%d/%m/%y %h:%m:%s")
    
    sql.sql.alchemy.dataframe_to_sql(dataframe = dataframe,
                         connection=LOCAL_CONNECTION,
                         schema=LOCAL_SCHEMA,
                         table=LOCAL_TABLE,
                         id_header=LOCAL_ID_HEADER,
                        sqa_dtypes=SQLA_DTYPES)


def copy_solys_csv_to_sql():
    #one time function to transfer CSV file of solys (slightl modified by fm) into local database#
    #Solys2180412-sunlog-20180829095411-all_181008_fm#
    dataframe = pd.read_csv(PATH_TO_FOLDER_SOLYS+FILE_NAME_SOLYS,
    usecols=['TMSTAMP',
             'step_seconds',
             'Intensity',
             'Error size',
             'Error dir',
             'Azimuth',
             'Zenith'],                       
    header=0)
    
    #dataframe["TMSTAMP"]=pd.DatetimeIndex(dataframe["TMSTAMP"].to_datetime(format="%d/%m/%Y %H:%M:%S"))

    

    sql.sql.alchemy.dataframe_to_sql(dataframe = dataframe,
                     connection=LOCAL_CONNECTION,
                     schema=LOCAL_SCHEMA,
                     table="solys",
                     id_header=LOCAL_ID_HEADER_SOLYS,
                    sqa_dtypes=SQLA_DTYPES_SOLYS)


   

def left_outer_join():
    #DEV NOTE 21/10/18: to be transferred into sql function
    table_left = "solys"
    table_right = "radiation"
    table_left_column = "TMSTAMP"
    table_right_column = "TMSTAMP"
    connstring = LOCAL_CONNECTION
    schema = LOCAL_SCHEMA
    table_left_columns = list(["TMSTAMP","Intensity","Zenith","step_seconds"])
    table_right_columns = list(["pyro_cmp11_w01","pyro_cmp11_w07",
                                "pyrheliometer_chp1_01_mms2","pyrheliometer_chp1_02_mms2",
                                "pyro_cmp11_w08"])
    table_columns=table_left_columns+table_right_columns
    
    select_query = "SELECT "
    started = False
    for value in table_left_columns:
        if started == True: 
            select_query = select_query + ","
        select_query = select_query + "tab_l."+'"'+str(value)+'"'
        started = True
    for value in table_right_columns:
        select_query = select_query + ","
        select_query = select_query + "tab_r."+'"'+str(value)+'"'
    select_query =  select_query +(
    " FROM "+schema+"."+table_left+" AS tab_l LEFT OUTER JOIN "+schema+"."+table_right+
    " AS tab_r ON (tab_l."+'"'+table_left_column+'"'+
    "=tab_r"+"."+'"'+table_right_column+'"'+")")
    
    #print(select_query)
    
    #print(table_columns)
    
    df=pd.DataFrame(sql.sql.alchemy.query(select_query,connstring))
    
    df.columns = table_columns
    
    return df 




#DEV NOTE 22/10.18 to be transferred into irradiance comparison
df_sql = left_outer_join()


df_sql["TMSTAMP"]=pd.DatetimeIndex(df_sql["TMSTAMP"])

df_sql=df_sql[(df_sql["TMSTAMP"]>"25/08/2018 00:00:00")&(df_sql["TMSTAMP"]<"26/08/2018 00:00:00")]


df_select = import_csv()


#print(df_select)

df =df_sql.merge(df_select,left_on='TMSTAMP',right_on='TMSTAMP',how='outer')


#df.to_csv(PATH_TO_OUTPUT+"jointed.csv")



time_column = "TMSTAMP"
global_columns = list(["pyro_cmp11_w01","pyro_cmp11_w07","CMP21_Irrd_Avg"])
beam_columns = list(["pyrheliometer_chp1_01_mms2","pyrheliometer_chp1_02_mms2","Intensity","CHP_Irrd_Avg"])
diffuse_columns = list(["pyro_cmp11_w08","CMP11a_Irrd_Avg"])
aoi_columns = list(["Zenith"])

mdq.SolarData.irradiance_comparison_iteration(crest_location,
                                         dataframe = df,
                                         time_column = time_column,
                                         global_columns = global_columns,
                                         beam_columns = beam_columns,
                                         diffuse_columns = diffuse_columns,
                                         aoi_columns = aoi_columns,
                                         graph_vs_time=True,
                                         graph_x_frequency="h",
                                         graph_deviation_abs=False,
                                         graph_deviation_prc=False,                                  
                                         markersize=2
                                         )



if "local sql" in source:
    
    querystring = "SELECT * FROM "+ LOCAL_SCHEMA +"."+LOCAL_TABLE
    #DEV NOTE: LIMIT 100 to test
    #read sql
    df_msrm = pd.read_sql(querystring,LOCAL_CONNECTION)




if "remote sql" in source:
    #modified version of select_on_list(parameter,table,schema,variable,lista,connection) since day needed
    variable = "day"
    querystring = "SELECT "
    started = False
    for value in list(REMOTE_HEADERS):
        if started == True: 
            querystring = querystring + ","
        querystring = querystring + "tabella." + value
        started = True
    querystring = querystring + ", date_trunc('day', measurementdatetime) AS day "   
    querystring = querystring + " FROM "+str(REMOTE_SCHEMA)+"."+REMOTE_TABLE+" AS tabella WHERE ("
    started = False   
    for value in list(CLEARSKY):
        if started == True: 
            querystring = querystring + " OR "
        querystring = querystring+"date_trunc('day', measurementdatetime)"+"='" + str(value)+"'"
        started = True
    
    querystring = querystring + ") AND extract('second' from measurementdatetime)=0"
    #querystring = querystring + ") AND extract('minute' from measurementdatetime)=0 AND extract('second' from measurementdatetime)=0"
    #read sql
    querystring = querystring + " LIMIT 100"
    start = dtt.datetime.now()
    df = pd.read_sql(querystring,REMOTE_CONNECTION)
    #print(querystring)
    print(dtt.datetime.now()-start)
    #renaming since not working before
    #DEV NOTE: dummy values for pt100
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

    


    
    
if resolution == "5s":
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
    tmstamp = df_5s["df.rename(columns={i:value},inplace=True)"]
    
    
    df_msrm  = pd.DataFrame({
    'pyrheliometer_chp1_01_mms2':pyrheliometer_chp1_01_mms2.mean(),
    'pyrheliometer_chp1_02_mms2':pyrheliometer_chp1_02_mms2.mean(),
    'pyro_cmp11_w07':pyro_cmp11_w07.mean(),
    'pyro_cmp11_w08':pyro_cmp11_w08.mean(),
    'pyro_cmp11_w01':pyro_cmp11_w01.mean(),
    'pyrheliometer_chp1_01_pt100':pyrheliometer_chp1_01_pt100.mean(),
    'pyrheliometer_chp1_02_pt100':pyrheliometer_chp1_02_pt100.mean(),
    "tmstamp":tmstamp.max()})
  
elif (resolution != "5s") and (results_str != ""):
    df = df_msrm
    df_msrm.rename(columns={"TMSTAMP":"tmstamp"},inplace=True)



if results_str != "":     
    df_msrm["AOI"] = crest_location.getsolarzenithangle(df_msrm["tmstamp"])
    #calculation of values necessary for directional response
    df_msrm["chp1_4_hrz_Wm-2"] = np.cos(np.radians(df_msrm.loc[:,"AOI"]))*df_msrm.loc[:,"pyrheliometer_chp1_01_mms2"]
    df_msrm["chp2_4_hrz_Wm-2"] = np.cos(np.radians(df_msrm.loc[:,"AOI"]))*df_msrm.loc[:,"pyrheliometer_chp1_02_mms2"]
    
    df_msrm["cmp1_vs_chp1_abs_Wm-2"] = (df_msrm.loc[:,"pyro_cmp11_w01"]-df_msrm["pyro_cmp11_w08"])-df_msrm["chp1_4_hrz_Wm-2"]     
    df_msrm["cmp7_vs_chp1_abs_Wm-2"] = (df_msrm.loc[:,"pyro_cmp11_w07"]-df_msrm["pyro_cmp11_w08"])-df_msrm["chp1_4_hrz_Wm-2"]       
    df_msrm["cmp1_vs_chp2_abs_Wm-2"] = (df_msrm.loc[:,"pyro_cmp11_w01"]-df_msrm["pyro_cmp11_w08"])-df_msrm["chp2_4_hrz_Wm-2"]
    df_msrm["cmp7_vs_chp2_abs_Wm-2"] = (df_msrm.loc[:,"pyro_cmp11_w07"]-df_msrm["pyro_cmp11_w08"])-df_msrm["chp2_4_hrz_Wm-2"]
       
    df_msrm["cmp1_vs_chp1_prc"] = df_msrm["cmp1_vs_chp1_abs_Wm-2"] / df_msrm.loc[:,"chp1_4_hrz_Wm-2"] * 100
    df_msrm["cmp7_vs_chp1_prc"] = df_msrm["cmp7_vs_chp1_abs_Wm-2"] / df_msrm.loc[:,"chp1_4_hrz_Wm-2"] * 100
    df_msrm["cmp1_vs_chp2_prc"] = df_msrm["cmp1_vs_chp2_abs_Wm-2"] / df_msrm.loc[:,"chp2_4_hrz_Wm-2"] * 100
    df_msrm["cmp7_vs_chp2_prc"] = df_msrm["cmp7_vs_chp2_abs_Wm-2"] / df_msrm.loc[:,"chp2_4_hrz_Wm-2"] * 100



if "cr30" in filter_str:
    #definition of cloud ratio, i.e. diffuse divided by global
    df_msrm["CR"]=df_msrm["pyro_cmp11_w08"]/df_msrm["pyro_cmp11_w07"]    
    #filter on cloud ratio, originally at 0.3
    df_msrm = df_msrm[(df_msrm["CR"]<0.3)]

if "irr50" in filter_str:     
    #TEMPORARY FILTER
    #Global higher than 50 W m-2, most strict BSRN checks
    #df_msrm = df_msrm[(df_msrm["CMP21_Glbl_Avg_Wm-2"]>50)]
    title="Directional error calculated assuming the same diffuse irradiance for the horizontal and tilted pyranometers (filter: "+ filter_str +")"
    df_msrm = df_msrm[((df_msrm["pyrheliometer_chp1_01_mms2"]>50)|(df_msrm["pyrheliometer_chp1_02_mms2"]>50))]


if "25/08/18" in filter_str:     
    df_msrm=df_msrm[(df_msrm["tmstamp"]>"25/08/2018 00:00:00")&(df_msrm["tmstamp"]<"26/08/2018 00:00:00")]



plt.close()

if  "overview" in results_str:        
    DPI = 100 
    FREQUENCY = "h"
    #plotting of functions
    
    if df_msrm.empty == False:
        
        plt.close()
        plt.title("Irradiance measurements from the roof")
        plt.ylabel('irradiance [Wm-2]')
        plt.xlabel('time') 
        
        
        
        p1,=plt.plot(df_msrm["tmstamp"],df_msrm["pyrheliometer_chp1_01_mms2"],".y")
        p2,=plt.plot(df_msrm["tmstamp"],df_msrm["pyrheliometer_chp1_02_mms2"],".r")
        p3,=plt.plot(df_msrm["tmstamp"],df_msrm["pyro_cmp11_w07"],".b")
        p4,=plt.plot(df_msrm["tmstamp"],df_msrm["pyro_cmp11_w08"],".k")
        p5=plt.plot(df_msrm["tmstamp"],df_msrm["pyro_cmp11_w01"],".g")
        
 
   
        #definition of intervals and extreme for x
        xticks = pd.date_range(start=min(df_msrm.loc[:,"tmstamp"].values),
        end=max(df_msrm.loc[:,"tmstamp"].values),freq=FREQUENCY)
        #plotting
        plt.show()


if "relative error" in results_str:

    plt.title("Relative error")
    plt.ylabel('directional response error [%]')
    plt.xlabel('angle of incidence')       
    
    p1,=plt.plot(df_msrm["AOI"],df_msrm["cmp1_vs_chp1_prc"],"y.")
    p2,=plt.plot(df_msrm["AOI"],df_msrm["cmp7_vs_chp1_prc"],"r.")
    p3,=plt.plot(df_msrm["AOI"],df_msrm["cmp1_vs_chp2_prc"],"b.")
    p4=plt.plot(df_msrm["AOI"],df_msrm["cmp7_vs_chp2_prc"],"k.")


    plt.show()



if "absolute error" in results_str:

    plt.title("Absolute error")
    plt.ylabel('directional response deviation [Wm-2]')
    plt.xlabel('angle of incidence')       
    
    p1,=plt.plot(df_msrm["AOI"],df_msrm["cmp1_vs_chp1_abs_Wm-2"],"y.")
    p2,=plt.plot(df_msrm["AOI"],df_msrm["cmp7_vs_chp1_abs_Wm-2"],"r.")
    p3,=plt.plot(df_msrm["AOI"],df_msrm["cmp1_vs_chp2_abs_Wm-2"],"b.")
    p4=plt.plot(df_msrm["AOI"],df_msrm["cmp7_vs_chp2_abs_Wm-2"],"k.")


    plt.show()


                
#output of all measurements temporary removed
if results_str != "":
    df_msrm.to_csv(PATH_TO_OUTPUT+"measurements.csv")

