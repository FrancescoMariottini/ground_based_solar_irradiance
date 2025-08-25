# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:51:30 2018

Yield analsyis and Montecarlo to be implemented into thesis

@author: wsfm
"""


import pandas as pd 
#importing meteodataquality for testing
import meteodataquality.meteodataquality as mdq
#importing pvlibinterface for solar library
import pvdata.solarlibraries.pvlibinterface as pvlibint
#import datetime for extracting month ?
import datetime as dtt



"""GLOBAL VARIABLES FILES"""
PATH_TO_OUTPUT_FOLDER = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"
PATH_TO_YIELD_FOLDER=r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/Yield_181025/input_csv/"
PATH_TO_METEONORM_FOLDER=r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/meteonorm/future_181113/"
PATH_TO_METEONORM_FOLDER=r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/meteonorm/2000_181114/"

FILE_NAME_MMS2=r"w_meas_mms2_met_all_181025.csv"

#Pyranometer CMP11_W01 measures global irradiance and is not on a ventilation unit.
#Pyranometer CMP11_W07 measures global irradiance and is on a ventilation unit.
#Pyranometer CMP11_W08 measures diffuse horizontal irradiance and is on a ventilation unit.

FILE_NAME_MMS1V1MET=r"w_meas_mms1v1met_2015_181026.csv"
FILE_NAME_SW238 = r"Solarworld_238_all_avg.csv"
FILE_NAME_MMS1WIND = r"w_meas_mms1wind_181026.csv"

FILE_ID_METEONORM_CREST = "crest_loc10_181113_fm-"
FILE_ID_METEONORM_EURAC = "abd_approx_181113_fm-"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

#crest starting hour columns
HOUR_STARTING_COLUMN = "startinghour"
HOUR_CLOSER_COLUMN = "closesthour"


"""GLOBAL VARIABLES PARAMETERS"""
#IMPORTANT NOTE ON CONVENTION: 
#Meteonorm: positive if east from Greenwich 
#
#location 18: Solys2 tracker, this is the 2 axis mounting for pyrheliometers and OEM sun sensor
#location 19: Solys2 tracker, this is the horizontal table for pyranometers etc with azimuth rotation and shading balls for some sensors
CREST_ROOF_LATITUDE_LOCATION_18t19 = 52.761103
CREST_ROOF_LONGITUDE_LOCATION_18t19 = 1.24092
CREST_ROOF_ALTITUDE_LOCATION_18t19 = 78
#Location 10: small horizontal sensor rack on W roof above the 45 rack location 9
CREST_ROOF_LATITUDE_LOCATION_10 = 52.7616
CREST_ROOF_LONGITUDE_LOCATION_10 = 1.2407
CREST_ROOF_ALTITUDE_LOCATION_10 = 80
#selected coordinates
CREST_ROOF_LATITUDE = CREST_ROOF_LATITUDE_LOCATION_10
#DEV NOTE 29/10/18: if necessary redo graph with negative based on Solar Position Algorithm for Solar Radiation Applications
CREST_ROOF_LONGITUDE = CREST_ROOF_LONGITUDE_LOCATION_10
CREST_ROOF_ALTITUDE = CREST_ROOF_ALTITUDE_LOCATION_10

EURAC_ABD_LATITUDE = 46.464010
EURAC_ABD_LONGITUDE = -11.330354
EURAC_ABD_ALTITUDE = 0  


CREST_CMP11_05_SENSITIVITY = 8.65
CREST_CMP11_06_SENSITIVITY = 8.8

"""USER DEFINED EXCEPTION"""

"""SETUP"""
class Setup:
    #assign differente values to variables depending on chosen setup 
    #SETUP VARIABLES
    path_to_output_folder: str
    path_to_input_folder: str
    path_to_input_file: str
    latitude: float
    longitude: float 
    altitude: float    
    solar_library: pvlibint.SolarLibrary
    time_column:str
    dataframe: pd.DataFrame  
    
    def __init__(self,source:str):
        if ("mms2" in source) or ("mms1v1met" in source) or ("mms1wind" in source) or ("238" in source):
            header = 0
            skiprows = None
            self.path_to_output_folder = PATH_TO_OUTPUT_FOLDER
            self.path_to_input_folder = PATH_TO_YIELD_FOLDER
            self.latitude = CREST_ROOF_LATITUDE
            self.longitude = CREST_ROOF_LONGITUDE
            self.altitude = CREST_ROOF_ALTITUDE 
            if ("mms2" in source): 
                self.path_to_input_file = FILE_NAME_MMS2
            elif ("mms1v1met" in source):
                self.path_to_input_file = FILE_NAME_MMS1V1MET           
            elif ("238" in source):
                self.path_to_input_file = FILE_NAME_SW238
            elif ("mms1wind" in source):
                self.path_to_input_file = FILE_NAME_MMS1WIND
        if ("meteonorm" in source):
            self.path_to_input_folder = PATH_TO_METEONORM_FOLDER
            if ("crest") in source:
                self.path_to_input_file = FILE_ID_METEONORM_CREST
                self.latitude = CREST_ROOF_LATITUDE
                self.longitude = CREST_ROOF_LONGITUDE
                self.altitude = CREST_ROOF_ALTITUDE 
            elif ("abd") in source:
                self.path_to_input_file = self.path_to_input_file + FILE_ID_METEONORM_EURAC
            if ("hour" in source):
                header = 0
                skiprows = 3
                self.path_to_input_file=self.path_to_input_file+"hour"+".csv"
            elif ("min" in source):
                header = 2
                skiprows = 2
                self.path_to_input_file=self.path_to_input_file+"min"+".dat"
            self.latitude = EURAC_ABD_LATITUDE
            self.longitude = EURAC_ABD_LONGITUDE
            self.altitude = EURAC_ABD_ALTITUDE    
            self.beam_column = "G_Bh"
            self.diffuse_column = "G_Dh"
            self.global_column = "G_Gh"
            
            self.time_column = "timestamp"
            
            self.temp_amb_column = "Ta"




            
        if ("csv" in source) or ("dat" in source):
            self.dataframe = pd.read_csv(filepath_or_buffer=self.path_to_input_folder+self.path_to_input_file,
                                         header=header,skiprows=skiprows)
            if ("mms2" in source) or ("mms1v1met" in source) or ("mms1wind" in source) or ("238" in source):
                self.time_column = HOUR_STARTING_COLUMN
            if ("mms1v1met") in source:
                #applying conversion formula. Note: tilt are 45.6 and 32.5 respectively
                self.dataframe["pyro_cmp11_w05_avg"]=self.dataframe["pyro_cmp11_w05_avg"]*1000/CREST_CMP11_05_SENSITIVITY
                self.dataframe["pyro_cmp11_w06_avg"]=self.dataframe["pyro_cmp11_w06_avg"]*1000/CREST_CMP11_06_SENSITIVITY
            
            
            if ("meteonorm" in source):
                if ("hour" in source):
                    time_df = pd.DataFrame({
                    'year':  self.dataframe[" y"],
                    'month': self.dataframe[" m"],
                    'day': self.dataframe[" dm"],
                    'hour': self.dataframe[" ST"].apply(lambda x: int(divmod(x,1)[0])),
                    'minute': self.dataframe[" ST"].apply(lambda x: int(divmod(x,1)[1]*60)) 
                    })
                    self.dataframe[self.time_column]=pd.to_datetime(time_df)
                    
                                
        #general setup
        self.solar_library=pvlibint.SolarLibrary(
        latitude = self.latitude,
        longitude = self.longitude,
        altitude = self.altitude)


""" SUPPORT FUNCTIONS """
# DEV NOTE 5/11/18: some support functions could be transferred into modules

def extract_meteo_ascii():
    #extract data from some crest db and aggregate them in a df 
    #DEV NOTE 29/10/18: not working due to not matching records ? merge to be checked 
    #opening different setup
    mms1v1met = Setup(source="mms1v1met csv")
    mms2 = Setup(source="mms2 csv")
    #DEV NOTE 29/10/18: wind remove due to not enough data
    #mms1wind = Setup(source="mms1wind csv")
    #converting into dataframe
    df_mms1v1met = pd.DataFrame(mms1v1met.dataframe)
    df_mms2 = pd.DataFrame(mms2.dataframe)
    df_mms2_columns= [x for x in df_mms2.columns if ((x!="date")&(x!="time"))]
    df_mms2=df_mms2[list(df_mms2_columns)]
    #DEV NOTE 29/10/18: wind remove due to not enough data
    #df_mms1wind  = pd.DataFrame(mms1wind.dataframe)
    #df_mms1wind_columns = [x for x in df_mms1wind.columns if ((x!="date")&(x!="time"))]
    #df_mms1wind= df_mms1wind[list(df_mms1wind_columns)]
    #datetimeindex
    df_mms1v1met[mms1v1met.time_column]=pd.DatetimeIndex(df_mms1v1met[mms1v1met.time_column])
    df_mms2[mms2.time_column]=pd.DatetimeIndex(df_mms2[mms2.time_column])
    #DEV NOTE 29/10/18: wind remove due to not enough data
    #df_mms1wind[mms1wind.time_column]=pd.DatetimeIndex(df_mms1wind[mms1wind.time_column])
    #filtering from 17/1/2015 to 17/1/2016
    df_mms1v1met_15 = df_mms1v1met[((df_mms1v1met[mms1v1met.time_column]>"17/01/2015 00:00:00")&(df_mms1v1met[mms1v1met.time_column]<"17/01/2016 00:00:00"))]  
    #mms2 available until 11/4/16
    df_mms2_15 = df_mms2[((df_mms2[mms2.time_column]>"17/01/2015 00:00:00")&(df_mms2[mms2.time_column]<"17/01/2016 00:00:00"))]
    #DEV NOTE 29/10/18: wind remove due to not enough data
    #df_mms1wind_15 = df_mms1wind[(df_mms1wind[mms1wind.time_column]>"17/01/2015 00:00:00")&(df_mms1wind[mms1wind.time_column]<"17/01/2016 00:00:00")]
    #DEV NOTE: momentarily ignoring dataset quality
    #merging dataset    
    df = df_mms1v1met_15.merge(df_mms2_15,left_on=mms1v1met.time_column,right_on=mms2.time_column,how='outer',copy=False)
    #print(df.iloc[1,:])
    #DEV NOTE 29/10/18: wind remove due to not enough data
    #df2 = df.merge(df_mms1wind_15,left_on=mms1v1met.time_column,right_on=mms1wind.time_column,how='outer',copy=False)
    return df

                  
def extract_cmp11_6():
    #extract data for cmp11_6
    mms1v1met = Setup(source="mms1v1met csv")
    df_mms1v1met = pd.DataFrame(mms1v1met.dataframe)
    df_mms1v1met[mms1v1met.time_column]=pd.DatetimeIndex(df_mms1v1met[mms1v1met.time_column])
    df_mms1v1met_15 = df_mms1v1met[((df_mms1v1met[mms1v1met.time_column]>"17/02/2015 00:00:00")&(df_mms1v1met[mms1v1met.time_column]<"17/02/2016 00:00:00"))]  
    return df_mms1v1met_15 


def missing_records(completeness:pd.DataFrame,time="hour"):
    #based on input of data completeness function, extract the list of missing records 
    #to be completed with other types of disconnection
    #create series of missing values
    df_true = completeness[(completeness["dif_set"]>0)|(completeness["dif_rise_t_x"]>0)
    |(completeness["rcv_days"]>0)|(completeness["dsc_days"]>0)|
    (completeness["dif_rise_t_y"]>0)]
    #DEV NOTE 30/10/18: to_datetime not working
    #datetime for miss_start
    df_true.loc[:,"miss_start"] = (
    pd.to_datetime(
    df_true[(df_true.dsc_days>0)]["date"].map(str)+" "+
    df_true[(df_true.dsc_days>0)]["data_off"].map(int).map(str)+":"+
    df_true[(df_true.dsc_days>0)]["data_off"].map(lambda x: str(int(x%1*60))+"0")
    ,format='%Y-%m-%d %I:%M',errors='ignore')
    )
    #datetime for miss end
    df_true.loc[:,"miss_end"] = (
    pd.to_datetime(
    df_true[(df_true.rcv_days>0)]["date"].map(str)+" "+
    df_true[(df_true.rcv_days>0)]["data_on"].map(int).map(str)+":"+
    df_true[(df_true.rcv_days>0)]["data_on"].map(lambda x: str(int(x%1*60))+"0")
    ,format='%Y-%m-%d %I:%M',errors='ignore')
    )
    df_true.loc[:,"miss_end"] =  df_true.loc[:,"miss_end"].shift(-1)
    #first identify index for ending
    start_list = list(df_true[(df_true.dsc_days>0)].index)
    #print(df_true[(df_true.dsc_days>0)][["date","data_on","data_off","miss_start","miss_end"]])
    #print(df_true.columns)
    print(df_true.loc[start_list,:][["date","data_on","data_off","miss_start","miss_end"]])

    
""" PROCESSING EXAMPLES """

process = "uncertainty impact on irradiance"
process = "irradiance generation"
process= "uncertainty impact on irradiance"

if "data completeness" in process:
    df = extract_cmp11_6()
    mms1v1met = Setup(source="mms1v1met csv")
    df_dc = mdq.SolarData.dailycompleteness(SolarLibrary=mms1v1met.solar_library,
                                  datetime_series=df[mms1v1met.time_column],
                                  days_limit=1,
                                  std_coef=3,
                                  dt_format='%d/%m/%Y %I:%M:%p',
                                  start_end_graph=False,
                                  outliers_graph=True,
                                  path_to_output =mms1v1met.path_to_output_folder)    
    missing = missing_records(df_dc)
    df_dc.to_csv(PATH_TO_OUTPUT_FOLDER+"meteo_ascii_6m_checked.csv")


if "irradiance generation" in process: 

    sys_aoi = pd.Series([10,30,30,50])
    sys_az = pd.Series([0,-30,+30,60])
    sys_temp = pd.Series([25,15,20,10])
    irradiance = pd.Series([1000,600,600,200])    
    from scipy.stats import norm
    a,b = norm.rvs(size=len(irradiance)),norm.rvs(size=len(irradiance))
    test_location = mdq.SolarData()
    sensor_uncertainty = mdq.SolarData.Irradiance_Uncertainty(test_location,"mrt",9.25)
    #0: irradiance based on law of uncertainty, 1st Taylor
    irradiance_0 = sensor_uncertainty.get_uncertainty(irradiance)
    #1: irradiance based on law of uncertainty, modelling of 
    
    irradiance_1 = sensor_uncertainty.get_uncertainty(irradiance,sys_temp,sys_az,sys_aoi)
    a = dtt.datetime.now()
    irradiance_2 = sensor_uncertainty.get_uncertainty_mc(irradiance,simulations=10)
    b = dtt.datetime.now() - a
    a = dtt.datetime.now()
    irradiance_3 = sensor_uncertainty.get_uncertainty_mc(irradiance,temperature=sys_temp,azimuth=sys_az,simulations=10000)
    b = dtt.datetime.now() - a
    
    
  
    print(b.seconds)
    print(irradiance_0)
    print(irradiance_1)
    print(irradiance_2)
    print(irradiance_3)
    
if "uncertainty impact on irradiance" in process:
    setup = Setup(source="meteonorm crest hour dat")
    df =setup.dataframe
    location = pvlibint.SolarLibrary(setup.latitude,setup.longitude,setup.altitude)  
    
    #df["aoi"]=location.getsolarzenithangle(df[setup.time_column])
    
   
    sensor_uncertainty_crest = mdq.SolarData.Irradiance_Uncertainty(location=location,
                                                                    method="mrt",
                                                                    sensitivity=CREST_CMP11_06_SENSITIVITY)
    #creating angle of incidence data
    df["aoi"]=pd.Series(location.getangleofincidence(df[setup.time_column]))
    #df = df.loc[12:14,:]
    #filtering datasets
    df0 = df[(df[setup.global_column]>0)&(df.aoi<90)]
    df35 = df[(df[setup.global_column]>35)&(df.aoi<90)]
    df200 = df[(df[setup.global_column]>200)&(df.aoi<90)]
    
   
    
    print("irradiance" )
    deviation0 = sensor_uncertainty_crest.get_uncertainty(df0[setup.global_column],coverage_factor=2)
    print(deviation0.sum(),deviation0.sum()/df0[setup.global_column].sum()*100)
    deviation35 = sensor_uncertainty_crest.get_uncertainty(df35[setup.global_column],coverage_factor=2)
    print(deviation35.sum(),deviation35.sum()/df35[setup.global_column].sum()*100)
    deviation200 = sensor_uncertainty_crest.get_uncertainty(df200[setup.global_column],coverage_factor=2)
    print(deviation200.sum(),deviation200.sum()/df200[setup.global_column].sum()*100)
    
    
    print("irradiance Monte Carlo multiple" )
    simulations = 100
    percentile = [50,90,95.5]
    deviation0 = sensor_uncertainty_crest.get_uncertainty_mc(
        df0[setup.global_column],coverage_factor=2,simulations=simulations,percentile=percentile)
    deviation35 = sensor_uncertainty_crest.get_uncertainty_mc(
        df35[setup.global_column],coverage_factor=2,simulations=simulations,percentile=percentile)
    deviation200 = sensor_uncertainty_crest.get_uncertainty_mc(
        df200[setup.global_column],coverage_factor=2,simulations=simulations,percentile=percentile)    
    irradiation_df = pd.DataFrame(
    columns=percentile,
    index=[">50",">90",">95.5"])

    for index,value in enumerate(percentile):
        print("PERCENTILE "+str(value))
        
        irradiation_df.loc[">50",value]=deviation0.loc[:,value].sum(),deviation0.loc[:,value].sum()/df0[setup.global_column].sum()*100
        irradiation_df.loc[">90",value]=deviation35.loc[:,value].sum(),deviation0.loc[:,value].sum()/df35[setup.global_column].sum()*100
        irradiation_df.loc[">95.5",value]=deviation200.loc[:,value].sum(),deviation200.loc[:,value].sum()/df200[setup.global_column].sum()*100
        
      
    irradiation_df.to_csv(PATH_TO_OUTPUT_FOLDER+"test.csv")
        
    
    #DEV NOTE 27/11/18: works until this point 
    
    """
    
    print("irradiance and angle of incidence")
    #deviation0 = sensor_uncertainty_crest.get_uncertainty(irradiance=df0[setup.global_column],angle_of_incidence=df0["aoi"],coverage_factor=2)
    
    
   
    #deviation35 = sensor_uncertainty_crest.get_uncertainty(irradiance=df35[setup.global_column],angle_of_incidence=df35["aoi"],coverage_factor=2)
    #deviation200 = sensor_uncertainty_crest.get_uncertainty(irradiance=df200[setup.global_column],angle_of_incidence=df200["aoi"],coverage_factor=2)
    
    #print([deviation0.sum(),deviation0.sum()/df[setup.global_column].sum()*100])                                                  
    #print([deviation35.sum(),deviation35.sum()/df[setup.global_column].sum()*100])                                                         
    #print([deviation200.sum(),deviation200.sum()/df[setup.global_column].sum()*100])                                                       
    
   
    
    df_aoi90 = df[df["aoi"]==90]
    
    print(df_aoi90[setup.global_column])
    print(df_aoi90["aoi"])
    
    
    
    deviation_aoi90 = sensor_uncertainty_crest.get_uncertainty(irradiance=df_aoi90[setup.global_column],angle_of_incidence=df_aoi90["aoi"],coverage_factor=2)
    
    
    
    print([deviation_aoi90.sum(),deviation_aoi90.sum()/df_aoi90[setup.global_column].sum()*100])                                   
                                                         

   
    
    
    print("irradiance and azimuth Monte Carlo")
    sys_aoi=location.getangleofincidence(df[setup.time_column])    
    deviation = sensor_uncertainty_crest.get_uncertainty_mc(irradiance=df[df["G_Gh"]>0]["G_Gh"],
                                                         angle_of_incidence=sys_aoi,coverage_factor=2,simulations=10)
    
    
    print(deviation.sum())
    print(deviation.sum()/df[df["G_Gh"]>0]["G_Gh"].sum()*100)
    deviation = sensor_uncertainty_crest.get_uncertainty_mc(irradiance=df[df["G_Gh"]>35]["G_Gh"],
                                                         angle_of_incidence=sys_aoi,coverage_factor=2,simulations=10)    
    print(deviation.sum())
    print(deviation.sum()/df[df["G_Gh"]>35]["G_Gh"].sum()*100)
    deviation = sensor_uncertainty_crest.get_uncertainty_mc(irradiance=df[df["G_Gh"]>200]["G_Gh"],
                                                         angle_of_incidence=sys_aoi,coverage_factor=2,simulations=10)    
    print(deviation.sum())
    print(deviation.sum()/df[df["G_Gh"]>200]["G_Gh"].sum()*100)
    
    """
    

    
"""
    irradiance_3 = sensor_uncertainty_crest.get_uncertainty_mc(irradiance,temperature=sys_temp,azimuth=sys_az,simulations=10000)
    b = dtt.datetime.now() - a
"""


""" #ARCHIVED TESTING  """


"""
df = extract_cmp11_6()
#df = extract_meteo_ascii()

mms1v1met = Setup(source="mms1v1met csv")
#setup = Setup(source="mms1v1met csv")
#setup = Setup(source="mms2 csv")

df2 = df[df[mms1v1met.time_column]=='2015-03-02T12:00:00.000000000']
df2 = df.head(2)

mms1v1met.solar_library.surface_azimuth = 180
mms1v1met.solar_library.surface_tilt = 0

irradiance = df2.loc[:,'cs215ambienttemp_avg']
temperature = df2.loc[:,'pyro_cmp11_w06_avg']
azimuth = mms1v1met.solar_library.getsolarazimuthangle(df2[mms1v1met.time_column])
angle_of_incidence = mms1v1met.solar_library.getangleofincidence(df2[mms1v1met.time_column])
"""



    