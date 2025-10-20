# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 08:30:48 2018
more information in the readme file

@author: wsfm
"""

# -*- coding: utf-8 -*-
"""
@author: Francesco Mariottini 
Created on 14/8/17
more information on the readme file 
9/9/25 some functions could be older version than the ones in other files
9/9/25 example: irradianceflagging vs datesflagging in executable.calibrations_analysis
"""

import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)


""" IMPORTING MAIN MODULES (used in most of the classes)"""
#importing pandas for dataframe
import pandas as pd
#import numpy for the any and all function
import numpy as np
#import matplotlib for plotting options 
import matplotlib.pyplot as plt
#explict poth to python modules
#DEV NOTE 18/10/18: maybe not necessary
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)
#importing "figureshow" for the pie chart (old version)
import dataframeshow_py.figureshow as fshw  #testing new version chart
import dataframeshow_py.chart as chrt
#importing pvlibint to introduce variables in functions definitions
import solarlibraries_py.pvlibinterface as pvlibint
#importing generation of number from a normal distribution 
from scipy.stats import norm
from scipy.optimize import fsolve # for equation solving
import datetime as dt #for datetimedelta object



"""GLOBAL VARIABLES """
PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"

#STANDARD UNCERTAINTY FOR KZ METHOD  
#Estimated standard uncertainties for KZ, see "21sheet" of "Pyranometer_Uncertainties_180410_v2"
#DEV NOTE 5/11/18: sources for different su to be reported  

#max number of decimals to consider in MC
MC_NORMAL_DECIMALS = 3 
#percentile to be considered in MC, for p90 is 1.645
COVERAGE_FACTOR = 1.645
#higher percentile
PERCENTILE_MAX = 95.45
PERCENTILE_MIN = 100-95.45

#uncertainty source for pyranometer iso9060, astmG213, iec61724-1 
TOTAL_ZERO_OFFSET={'uncertainty':'total_zero_offset','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':10,'acceptance_b':21,'acceptance_c':41}
NON_STABILITY={'uncertainty':'non_stability','id':'isoc1','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.8,'acceptance_b':1.5,'acceptance_c':3}
NON_LINEARITY={'uncertainty':'non_linearity','id':'isoc2','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.5,'acceptance_b':1,'acceptance_c':3}
DIRECTIONAL_RESPONSE={'uncertainty':'directional_response','id':'isoc3','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':10,'acceptance_b':20,'acceptance_c':30}
SPECTRAL_ERROR={'uncertainty':'spectral_error','id':'isoc4','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.5,'acceptance_b':1,'acceptance_c':5}
TEMPERATURE_RESPONSE={'uncertainty':'temperature_response','id':'isoc5','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':1,'acceptance_b':2,'acceptance_c':4}
TILT_RESPONSE={'uncertainty':'tilt_response','id':'isoc6','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.5,'acceptance_b':2,'acceptance_c':5}
SIGNAL_PROCESSING={'uncertainty':'signal_processing','id':'isoc7','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':2,'acceptance_b':5,'acceptance_c':10}
#21/5/20 signal processing should be f(voltage) not f(irradiance)
CALIBRATION={'uncertainty':'calibration','id':'astm1','parameter':'sensitivity','distribution':'symmetric','shape':'symmetric','divisor':2,'acceptance_a':5.62,'acceptance_b':5.62,'acceptance_c':5.62}
MAINTENANCE={'uncertainty':'maintenance','id':'astm8','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.3,'acceptance_b':0.3,'acceptance_c':0.3}
ALIGNMENT_ZENITH={'uncertainty':'alignment_zenith','id':'iec6b','parameter':'zenith','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':1,'acceptance_b':1.5,'acceptance_c':2}
ALIGNMENT_AZIMUTH={'uncertainty':'alignment_azimuth','id':'iec6a','parameter':'azimuth','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':2,'acceptance_b':3,'acceptance_c':4}

UNCERTAINTIES=[TOTAL_ZERO_OFFSET,NON_STABILITY,NON_LINEARITY,DIRECTIONAL_RESPONSE,
               SPECTRAL_ERROR,TEMPERATURE_RESPONSE,TILT_RESPONSE,SIGNAL_PROCESSING,
               CALIBRATION,MAINTENANCE,ALIGNMENT_ZENITH,ALIGNMENT_AZIMUTH]
UNCERTAINTIES_DF=pd.DataFrame()

for value in list(UNCERTAINTIES):
    UNCERTAINTIES_DF=UNCERTAINTIES_DF.append(value,ignore_index=True)
UNCERTAINTIES_DF.index=UNCERTAINTIES_DF.loc[:,"uncertainty"]
METHOD="isoa_astm"
UNCERTAINTIES_CMP21={"non_stability":0.5,"non_linearity":0.2,"calibration":1.4} 



""" # 13/5/20 old method based on LR to be removed
UNCERTAINTY_TABLE = pd.DataFrame({ 
"source": ["datalogger","calibration uncertainty","non-stability","non-linearity","temperature response",
"maintenance","zero off-set a","zero off-set b","directional response"],
"parameter":["voltage","sensitivity","sensitivity","sensitivity","sensitivity","sensitivity","irradiance",
             "irradiance","irradiance"],
"parameter unit":["µV","µVW-1m2","µVW-1m2","µVW-1m2","µVW-1m2","µVW-1m2","Wm-2","Wm-2","Wm-2"],
"limit": [10.00,0.15,0.50,0.20,1.00,0.50,-7.00,2.00,2],
"limit unit": ["µV","µVW-1m2","%","%","%","%","W m-2","W m-2","W m-2"],
"distribution": ["rectangular","normal","rectangular","rectangular","rectangular","rectangular",
          "rectangular","rectangular","rectangular"],
"shape": ["symmetric","symmetric","negative","symmetric","symmetric","symmetric","negative","symmetric","symmetric"],
"divisor": [3**0.5,2,2*3**0.5,3**0.5,3**0.5,3**0.5,2*3**0.5,3**0.5,3**0.5]
},
index = ["datalogger","calibration uncertainty","non-stability","non-linearity","temperature response",
"maintenance","zero off-set a","zero off-set b","directional response"])


#parameters used for old method
SU_CALIBRATION_KZ = 0.0735
#S uncertainty, spec: 0.15 µV W-1 m2 (normal) [KZ] for a coverage of 2*su (about 95.5%)
#su obtained from the specification-based value divided by two 
CALIBRATION_COVERAGE_FACTOR = 2
#converage factor to convert into expenaded uncertainty.
#2 for a level of confidence of approx 95% in case of normal distribution. Used for calibration uncertainty
SU_NONSTABILITY_KZ = 0.015155445
#S uncertainty, spec: 0.5 % (rectangular, one-side negative) [KZ]
#su obtained from the specification-based value divided by first dividing by root square of 3 and then 2
SU_NONLINEARITY_KZ = 0.012124356
#S uncertainty, spec: 0.2 % (rectangular) [KZ]
SU_TEMPERATURE_KZ = 0.060621778
#S uncertainty, spec: 1 % (rectangular) [KZ]
#max deviation declared in calibration certificate is 0.54
SU_OFFSET_A_KZ = 2.020725942 
#E uncertainty, spec: -7 W m-2 (rectangular, one-side negative) [KZ]
SU_OFFSET_B_KZ = 1.154700538 
#E uncertainty, spec: 2 W m-2 (rectangular) [KZ]
SU_DIRECTIONAL_KZ = 5.773502692
#E uncertainty, spec: 2 W m-2 (rectangular) [KZ]

SU_DIRECTIONAL_KNN = 5.921304360
#E uncertainty, spec: 1% (rectangular) [KNN]


SU_DATALOGGER_KNN = 5.773502692
#V uncertainty, spec: 10.00 µV (rectangular) from manufacturer  [HBT] 
#[KNN] [MRT] used 10 
SU_DATALOGGER_DRIFT =  0.923760431
#V uncertainty, spec: 0.32 Wm-2(rectangular) [KRT]
SU_DATALOGGER_RESOLUTION = 0.028867513
#V uncertainty, spec: 0.01 Wm-2(rectangular) [KRT]
SU_DATALOGGER_CONVERTER =  0.058312377
#V uncertainty, spec: 0.02 µV (rectangular) [KRT]

SU_MAINTENANCE_KNN = 0.030310889
#S uncertainty, spec: 0.5 % (rectangular) [KNN] [MRT]
SU_MAINTENANCE_HBT = 0.013983712
#S uncertainty, spec: 0.3 % (rectangular) [HBT]
SU_SPECTRAL_HBT =  0.060621778
#S uncertainty, spec: 1 % (rectangular) [HBT]
SU_SPECTRAL_KRT =  0.014433757
#S uncertainty, spec: 0.5 % (rectangular) [KRT]

#Specification limit source (S & E su are calculated depending on irradiance value)
# [HBT] Calibration and Measurement Uncertainty Estimation of Radiometric Data. Habte, Aron at al. 2014 
# [KZ] Kipp & Zonen CMP21 datasheet (higher values possible when extracted from older standards) 
# [KNN] Uncertainty evaluation of measurements with pyranometers and pyrheliometers. Konings, Jörgen and Habte, Aron, 2015
# [KRT] Uncertainty Calculations in Pyranometer Measurements and application. Kratzenberg, M. G. et al 2006
# [MRT] Evaluation of uncertainty sources and propagation from irradiance sensors to PV energy production Evaluation of Uncertainty Sources and Propagation from Irradiance Sensors to PV Energy Production, 2018
"""


#Point values for the calibration of KZ CMP21 090318
TEMPERATURE_DEVIATION_KZ_CMP21 = pd.Series(
[-0.29,0.04,0.11,0.01,0.00,0.06,0.03,0.54],
index=  [-20,-10,0,10,20,30,40,50])
DIRECTIONAL_DEVIATION_KZ_CMP21 = pd.DataFrame(
{"-180": [1.23725,0.49,0.18375,0.21],
 "-90": [3.20675,1.27,0.47625,0.23],
 "0": [2.02,0.8,0.3,0.07],
 "90": [-0.47975,-0.19,-0.07125,-0.05],
 "180": [1.23725,0.49,0.18375,0.21]}, 
index = ["40","60","70","80"])
 #DEV NOTE 14/11/18: angles of incidence correctly changed as also in the paper and poster 
TEMPERATURE_DEVIATION_KZ = TEMPERATURE_DEVIATION_KZ_CMP21
DIRECTIONAL_DEVIATION_KZ = DIRECTIONAL_DEVIATION_KZ_CMP21
TEMPERATURE_DEVIATION_RANGE_KZ = pd.DataFrame({
"mean":[-0.440923077,	0.523076923,	0.471692308,	0,	-0.328307692,	-0.194769231],
"sd":[0.451230769	, 0.318,	0.266615385,	0,	0.394923077,	0.610307692]},
index=[-10,0,10,20,30,40]
)
#representative values used to calculated the overall uncertainty 
#highest t value based on: http://www.kippzonen.com/News/582/The-Importance-of-Pyranometer-Temperature-Response#.W-vo5LFKiM-
TEMPERATURE_REFERENCE_KZ = 50
#option A: angles representative of maximum deviations as proposed in calibration certificate (no interpolation used)
ANGLE_OF_INCIDENCE_REFERENCE_KZ = 80
AZIMUTH_REFERENCE_KZ = 0
#OUTDOOR CALIBRATION INFORMATION
CALIBRATION_READINGS = 21
ITERATIONS_MAXIMUM = 100
#absolute deviation from calibration factor
DEVIATION_MAX = 0.02 #10/1/20 modified from 2
#days flagging 
HOURS_LIMIT=24
COVERAGE_FACTOR=2 #DEV NOTE 13/5/19: 2 instead of 3 (RG) applied for peak irradiance, to be justified why 
CLEARSKY_FREQUENCY=30 #from cleaning requirements but checks also calibration
CLEARSKY_ITERATION_SPEED=1 #speed adapted per parameter. 1 corresponding to speed in FM paper 2018
CLEARSKY_ERROR = 0.03 #GHI A posteriori clear-sky identification methods in solar irradiance time series: Review and preliminary validation using sky imagers
ITERATION_LIMIT = 1000 
#overall criteria
AOI_MAX = 70 # [ISO 9847]
NIGHT_HOURS_MIN=10 # to avoid time conversion issues due to wrong timezone or DST
#series criteria
RESOLUTIONS=[1,60,3600,86400]
SERIES_VALUES_MIN = CALIBRATION_READINGS #BOLZANO  resolution is 15 minutes, thus all valid values in the series
SERIES_COUNT_MIN = 15 # 15 ok if series of 15 minutes analysed [ISO 9847] 
KCS_DEV_LMT = 0.2 
# 0.2: uncertainty 95% of HOURLY values for moderate quality (class C) radiometers [WMO 2017) @fm: enough due to additional filter and uncertainty of mcs
KCS_CV_LMT = 0.02 
# adapted from [ISO 9847]
PWR_CV_LMT = 0.1

#daily criteria
PEARSON_LMT = 0.8 #strong correlation 
#other criteria
IRR_MIN = 20 
#20 to evaluate first switch on #ISO 61724-1:2-2017
#100 for calibration in cloudy conditions 9847
#NOT APPLIED 
IRR_DAY_DEV_LMT = 0.1 #uncertainty 95% of daily values for moderate quality radiometers [WMO)]
#too strict filter in case of alignment problems. Pearson used instead. 
ISO9060_AOI = [0,40,60,70,80,95]
ISO9060_AZIMUTH = [0,90,180,270]
IEC61724_ALIGNMENT_TILT = 1 #1, 1.5, 2
IECAZIMUTH_ALIGNMENT_AZIMUTH = 2 #2, 3, 4
 



"""CLASSES AND FUNCTIONS """

def tdhour(td:np.timedelta64): #td function to get hours
    return (td.total_seconds()/3600)




def datacompleteness(dataframe=pd.DataFrame,
                     datetimecolumn="datetime",
                     checkcolumn="irradiance",
                     timeresolutionmax=1): 
    timerstart = dt.datetime.now()    
    
    datetimeindex_utc = pd.DatetimeIndex(dataframe.loc[:,datetimecolumn].values,ambiguous='NaT',name="datetime")  

    dataframe.index=datetimeindex_utc.to_series(name="index") #26/2/20 .index remove the time 
    
    datetimeindex_utc_all = pd.date_range(start=min(datetimeindex_utc.date)-pd.DateOffset(1),end=max(datetimeindex_utc.date),
    freq=str(timeresolutionmax)+"s",name="datetime_all")
    
    df=pd.merge(left=dataframe,
                       right=datetimeindex_utc_all.to_series(keep_tz=True,index=None,name=datetimecolumn),
                       how="right",on=datetimecolumn,indicator=True,sort=datetimecolumn)

    #df.index=datetimeindex_utc_all.to_series(keep_tz=True,index=None,name=datetimecolumn) #23/2/20 mistmatch not sure why
    df.index = pd.DatetimeIndex(df.loc[:,datetimecolumn])
    df["date"]=df.index.date
    df=df[(df.date>=min(datetimeindex_utc.date))&(df.date<=max(datetimeindex_utc.date))]    
    def my_agg(x):
        #https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with
        names={
        "na": len(x[x["_merge"]=="right_only"]),
        "null":len(x[x[checkcolumn].isnull()]) #to be improved
        }
        return pd.Series(names, index=['na','null'])
    #df_agg = df_grp["_merge"].agg(lambda x: len(x[x=="right_only"]))
    #df_daily= pd.DataFrame({"missing":df_agg.values},index=df_agg.index)
    #df_grp = df.groupby("date")
    df_daily=df.groupby('date').apply(my_agg)  
    df = df.merge(df_daily,how="left",on="date")
    df = df.fillna(value={checkcolumn:0})
    df_daily["date"]=df_daily.index #"21/2/20 date after to avoid merging issues    
    print("data completeness: "+str(dt.datetime.now()-timerstart))
    return df, df_daily


class Irradiance_Uncertainty:        
        def __init__(self,sensitivity:float,coverage_factor=COVERAGE_FACTOR,method=METHOD):#=500):
            #sensitivity in V/Wm-2
            self.uncertainties_df = UNCERTAINTIES_DF
            if "isoa" in method: self.uncertainties_df.loc[:,"acceptance"]=self.uncertainties_df.loc[:,"acceptance_a"]
            elif "isob" in method: self.uncertainties_df.loc[:,"acceptance"]=self.uncertainties_df.loc[:,"acceptance_b"]
            elif "isoc" in method: self.uncertainties_df.loc[:,"acceptance"]=self.uncertainties_df.loc[:,"acceptance_c"]
            if ("cmp21" in method) or ("cmp11" in method):
                for k,v in UNCERTAINTIES_CMP21.items():self.uncertainties_df.loc[k,"acceptance"]=v
            self.method = method
            #DEV NOTE 16/11/18: to be renamed as test, irradiance new 
            self.sensitivity = sensitivity
            self.coverage_factor = coverage_factor        
            """ #13/5/20 to be checked
            elif method == "mrt2":
                df = UNCERTAINTY_TABLE
                parameter_values = dict({
                "irradiance": irradiance,
                "sensitivity": sensitivity})
                for index in list(df.index):
                    if df.loc[index,"parameter unit"] == df.loc[index,"limit unit"]:
                        su_tmp = df.loc[index,"limit"]
                    elif df.loc[index,"parameter unit"] == "%":
                        su_tmp = df.loc[index,"limit"]*parameter_values[df.loc[index,"parameter"]]
                    su_tmp = su_tmp / df.loc[index,"divisor"].values
                self.su_datalogger = SU_DATALOGGER_KNN
                self.su_calibration = SU_CALIBRATION_KZ
                self.su_nonstability = SU_NONSTABILITY_KZ
                self.su_nonlinearity = SU_NONLINEARITY_KZ
                self.su_temperature = SU_TEMPERATURE_KZ 
                self.su_maintenance = SU_MAINTENANCE_KNN
                self.su_offset_a = SU_OFFSET_A_KZ
                self.su_offset_b = SU_OFFSET_B_KZ
                self.su_directional = SU_DIRECTIONAL_KZ
        """

        def get_deviation_on_temperature(self,temperature=None):
            #DEV NOTE 31/10/18: only Series and not series cases covered here
            if self.method == "mrt":
                temp_dev_tab = TEMPERATURE_DEVIATION_KZ
                x = list(temp_dev_tab.index.values)
                f = list(temp_dev_tab.values)
                temperature_reference = TEMPERATURE_REFERENCE_KZ
            #uncertainty due to temperature, estimated interpolation from KZ data  
                def temp_deviation(temperature):
                    #if (temperature < min(x))or(temperature>max(x)):
                    #    raise ValueError("temperature out of the function bounds")
                    #    return None 
                    #elif (temperature >= min(x)) or (temperature<=max(x)):
                    if temperature <= min(x):    
                        temp_deviation = min(f)        
                    elif temperature >= max(x):
                        temp_deviation = max(f)
                    elif (temperature > min(x)) & (temperature < max(x)):                
                        temp_deviation = np.interp(temperature, x, f)
                    return temp_deviation
            #regardless of model used, return single value or series depending on istance             
            if isinstance (temperature,pd.Series) == False: 
                if temperature == None:
                    temperature = temperature_reference
                deviation_on_temperature = temp_deviation(temperature)
            if (isinstance(temperature,pd.Series) == True) and (np.all(temperature) is not None): 
                    #deviation_on_temperature = temperature.apply(lambda x: temp_deviation(x))
                    deviation_on_temperature = temperature.transform(lambda x: temp_deviation(x))
                    #DEV NOTE 5/11/18: old working version
                    #deviation = pd.Series(index=temperature.index)
                    #_t_tmp = temperature[temperature <= -20]
                    #deviation.loc[_t_tmp.index] = np.full(len(_t_tmp),-0.29)
                    #_t_tmp = temperature[temperature >= 50]
                    #deviation.loc[_t_tmp.index] = np.full(len(_t_tmp),0.54)
                    #_t_tmp = temperature[(temperature > -20) & (temperature < 50)]
                    #deviation.loc[_t_tmp.index]=temperature.loc[_t_tmp.index].apply(lambda t: np.interp(t, x, f))
            return deviation_on_temperature
            
                
        def get_deviation_on_direction(self,sys_az=None,sys_aoi=None):
            #directional response in 
            #DEV NOTE 5/11/18: check if interpolation could be done with
            #x,y = np.meshgrid(az,aoi,sparse=True)
            #data = f(*np.meshgrid(az,aoi,indexing='ij',sparse=True))
            #from scipy.interpolate import RegularGridInterpolator
            #my_interpolating_function = RegularGridInterpolator((az,aoi,dev),data)
            if self.method == "mrt":
            #uncertainty due to angle of incidence and azimuth of the radiometer, estimated interpolation from KZ data
                dir_dev_tab = DIRECTIONAL_DEVIATION_KZ 
                az = list(dir_dev_tab.columns.values.astype(int))
                aoi = list(dir_dev_tab.index.values.astype(int))
                azimuth_reference = AZIMUTH_REFERENCE_KZ
                angle_of_incidence_reference = ANGLE_OF_INCIDENCE_REFERENCE_KZ
                #KZ dependency for directional response
                def directional_deviation(sys_az,sys_aoi): 
                    #defining related aoi range              
                    if sys_aoi <= min(aoi):
                        aoi_start = min(aoi)
                        aoi_end = min(aoi)
                    elif sys_aoi >= max(aoi):
                        aoi_start = max(aoi)
                        aoi_end = max(aoi)
                    elif (sys_aoi > min(aoi)) and (sys_aoi < max(aoi)):
                        #if not outside range, iterate to find interpolation interval 
                        for index,value in enumerate(aoi):
                            if value == max(aoi):
                                break
                            elif value <= sys_aoi:
                                #'=' condition if exact value
                                aoi_start = value
                                aoi_end = aoi[index+1]
                    #defining related azimuth range
                    if sys_az <= min(az):
                        az_start = min(az)
                        az_end = min(az)
                    elif sys_az >= max(az):
                        az_start = max(az)
                        az_end = max(az)
                    elif (sys_az > min(az)) and (sys_az < max(az)):
                        #if not outside range, iterate to find interpolation interval 
                        for index,value in enumerate(az):
                            if value == max(az):
                                break
                            elif value <= sys_az:
                                #'=' condition if exact value
                                az_start = value
                                az_end = az[index+1]
                    #combined interpolation
                    az0_aoi = [aoi_start,aoi_end]
                    #retrieving values from kz results 
                    az0_dev = [dir_dev_tab.loc[str(aoi_start),str(az_start)],
                    dir_dev_tab.loc[str(aoi_end),str(az_start)]]
                    az0_val = np.interp(sys_aoi, az0_aoi, az0_dev)                            
                    az1_aoi = [aoi_start,aoi_end]
                    az1_dev = [dir_dev_tab.loc[str(aoi_start),str(az_end)],
                    dir_dev_tab.loc[str(aoi_end),str(az_end)]]
                    az1_val = np.interp(sys_aoi, az1_aoi, az1_dev)                            
                    az_range = [az_start,az_end]
                    dev_range = [az0_val,az1_val]                            
                    return np.interp(sys_az, az_range, dev_range)              
            if (isinstance(sys_az,pd.Series)==False) & (isinstance(sys_aoi,pd.Series)==False):
                #DEV NOTE 29/11/18: temp sol due to '<' not supported between instances of 'NoneType' and 'int'
                if sys_aoi is None:
                    deviation = None
                elif sys_aoi is not None:
                    if sys_aoi < 90:
                        deviation = directional_deviation(sys_az,sys_aoi) 
                    elif sys_aoi >= 90:
                        raise ValueError("angle of incidence not lower than 90 degree")
                        deviation = None 
            elif (isinstance(sys_az,pd.Series)==True) or (isinstance(sys_aoi,pd.Series)==True):
                if (isinstance(sys_aoi,pd.Series)==False):
                    sys_aoi=pd.Series(np.full(shape=len(sys_az),fill_value=angle_of_incidence_reference))
                if (isinstance(sys_az,pd.Series)==False):
                    sys_az=pd.Series(np.full(shape=len(sys_aoi),fill_value=azimuth_reference))
                if (np.any(sys_aoi) is None) or (np.any(sys_az) is None):
                    deviation = None
                elif (np.all(sys_aoi) is not None) or (np.all(sys_az) is not None):
                    if np.all(sys_aoi) < 90 and abs(np.all(sys_az)) <= 180:
                        df_inputs = pd.DataFrame({"sys_az":sys_az,"sys_aoi": sys_aoi})
                        #returning deviation as series to keep same index
                        #deviation = df_inputs.apply(lambda x: directional_deviation(x["sys_az"],x["sys_aoi"]),axis=1)
                        deviation = df_inputs.transform(lambda x: directional_deviation(x["sys_az"],x["sys_aoi"]),axis=1) 
                    elif np.any(sys_aoi) >= 90 and abs(np.all(sys_az)) > 180:
                        deviation = None 
            return deviation

        def get_uncertainty_gum(self,irradiance_total,diffuse_fraction=0,zenith=None,azimuth=None,surface_zenith=None,surface_azimuth=None,temperature=None):
            #not none temperature for certificate uncertainty
            #100% total as conservative estimation of beam (minimum of 80%)
            cit = 1; uit2 = 0 
            cv = 1/self.sensitivity; uv2 = 0
            voltage = self.sensitivity*irradiance_total                
            cs = -(voltage)/(self.sensitivity**2); us2 = 0                           
            for row in list(self.uncertainties_df.index):
                u=self.uncertainties_df.loc[row,"acceptance"]/self.uncertainties_df.loc[row,"divisor"]
                p=self.uncertainties_df.loc[row,"parameter"]
                if p=="irradiance": uit2+=u**2
                elif p=="voltage": uv2+=u**2
                elif p=="sensitivity": us2+=u**2
            us2 = us2 * (self.sensitivity/100)**2 #13/5/20 adapt to percentage values
            comb_stnd_uncr2 = (uv2*cv**2+us2*cs**2+uit2*cit**2)
            if (zenith is not None)&(azimuth is not None)&(surface_zenith is not None)&(surface_azimuth is not None):
                z=zenith/180*np.pi;a=azimuth/180*np.pi;z0=surface_zenith/180*np.pi;a0=surface_azimuth/180*np.pi
                uz = self.uncertainties_df.loc["alignment_zenith","acceptance"]/self.uncertainties_df.loc["alignment_zenith","divisor"]*np.pi/180
                ua = self.uncertainties_df.loc["alignment_azimuth","acceptance"]/self.uncertainties_df.loc["alignment_azimuth","divisor"]*np.pi/180
                ca0=irradiance_total*(1-diffuse_fraction)*(np.cos(a0)*np.sin(a)*np.sin(z)*np.sin(z0)-np.cos(a)*np.sin(a0)*np.sin(z)*np.sin(z0))
                cz0=irradiance_total*(1-diffuse_fraction)*(np.cos(a)*np.cos(a0)*np.cos(z0)*np.sin(z)-np.cos(z)*np.sin(z0)+np.sin(a)*np.sin(a0)*np.cos(z0)*np.sin(z))
                comb_stnd_uncr2+=ca0**2*ua**2+cz0**2*uz**2
            exp_uncr = (comb_stnd_uncr2**0.5)*self.coverage_factor
            return exp_uncr 

        def get_uncertainty(self,
                          irradiance,
                          temperature = None,
                          azimuth = None,
                          angle_of_incidence = None,                                  
                          coverage_factor=COVERAGE_FACTOR,
                          mc_coeff = None):
            if np.all(temperature) is not None:
                #DEV NOTE 15/11/18: positive chosen over negative
                sensitivity = self.sensitivity*(100+self.get_deviation_on_temperature(temperature))/100
                su_temperature = 0 
            elif np.any(temperature) is None:
                sensitivity = self.sensitivity
                su_temperature = self.su_temperature
            if (np.all(azimuth) is not None) or (np.all(angle_of_incidence) is not None):
                #check if deviation not None (e.g. aoi <90 )
                #print(azimuth,angle_of_incidence)
                deviation_directional = self.get_deviation_on_direction(azimuth,angle_of_incidence)
                if deviation_directional is not None:
                    irradiance = irradiance * (100+deviation_directional)/100
                    su_directional = 0
                if deviation_directional is None:
                    su_directional = None
            elif (np.any(azimuth) is None) | (np.any(angle_of_incidence) is None):
                su_directional = self.su_directional  
            #DEV NOTE 9/11/18: MC part currently compatible only with specific iteration
            #estimation of expanded uncertainty with irradiance measurements based on voltage and sensitivity only (negligible infrared contribution)
            #first check if su_temperature and su_directional have been properly calculated
            if (su_directional is not None) and (su_temperature is not None):
                voltage = sensitivity * irradiance
                if np.any(mc_coeff) is None:
                        #calculate uncertainties related to each parameter
                        uv2 = (self.su_datalogger)**2
                        us2 = ((self.su_calibration)**2+(self.su_nonstability)**2+(self.su_nonlinearity)**2+(su_temperature)**2+(self.su_maintenance)**2)
                        ue2 = ((self.su_offset_a)**2+(self.su_offset_b)**2+(su_directional)**2)     
                        #calculate sensitivity coefficient
                        cv = 1/self.sensitivity
                        cs = -(voltage)/(self.sensitivity**2)
                        ce = 1      
                        #calculate combined standard uncertainty
                        comb_stnd_uncr = (cv**2*uv2 + cs**2*us2 +ce**2*ue2)**0.5
                        #calculate combined expanded uncertainty 
                        exp_uncr = comb_stnd_uncr*coverage_factor
                        uncr = exp_uncr
                elif np.all(mc_coeff) is not None:
                    #transformation function from su to
                    def su_to_val_rect(su:float,random_num:float,asymmetric="no"):
                        #random number fom 0 to 1
                        if asymmetric == "no":
                            su_to_val_rect=su*(3**0.5)*(-1+2*random_num)
                        if asymmetric == "negative":
                            su_to_val_rect=-su*(3**0.5)*(random_num)*2
                        return su_to_val_rect
                    def su_to_val_norm(su:float,random_num:float,asymmetric="no"):
                        #random number fom 0 to 1
                        su_to_val_norm=su*(-1+2*random_num)
                        return su_to_val_norm  
                    #deviation with normal distribution
                    dev_calibration=su_to_val_norm(self.su_calibration,mc_coeff[0],asymmetric="no")
                    #deviation with rectangular distribution asymmetric     
                    #deviation with rectangular distribution asymmetric       
                    dev_datalogger=su_to_val_rect(self.su_datalogger,mc_coeff[1],asymmetric="no")
                    dev_nonlinearity=su_to_val_rect(self.su_nonlinearity,mc_coeff[3],asymmetric="no")
                    dev_temperature=su_to_val_rect(self.su_temperature,mc_coeff[4],asymmetric="no")
                    dev_maintenance=su_to_val_rect(self.su_maintenance,mc_coeff[5],asymmetric="no")
                    dev_offset_b=su_to_val_rect(self.su_offset_b,mc_coeff[7],asymmetric="no")
                    dev_directional=su_to_val_rect(self.su_directional,mc_coeff[8],asymmetric="no")
                    #deviation with rectangular distribution negative only  
                    dev_nonstability=su_to_val_rect(self.su_nonstability,mc_coeff[2],asymmetric="negative")          
                    dev_offset_a=su_to_val_rect(self.su_offset_a,mc_coeff[6],asymmetric="negative")
                    #test symmetrical rectangular distribution 
                    #dev_nonstability=su_to_val_rect(self.su_nonstability,mc_coeff[2],asymmetric="no")/2     
                    #dev_offset_a=su_to_val_rect(self.su_offset_a,mc_coeff[6],asymmetric="no")/2
                    #uncertainties per parameter
                    v2 = (voltage+dev_datalogger)/voltage
                    s2 = ((sensitivity+dev_calibration)*
                           (sensitivity+dev_nonstability)*
                           (sensitivity+dev_nonlinearity)*
                           (sensitivity+dev_temperature)*
                           (sensitivity+dev_maintenance)/sensitivity**5)
                    e2 = ((irradiance+dev_offset_a)*
                           (irradiance+dev_offset_b)*
                           (irradiance+dev_directional)/irradiance**3)
                    irr_adj=v2*s2*e2*irradiance
                    uncr=irr_adj-irradiance
            elif (su_directional is None) or (su_temperature is None):
                uncr = None 
            return uncr 
           
        
        def get_uncertainty_mc(self,irradiance: float,
                                simulations: int,
                                angle_of_incidence = None,
                                temperature = None,
                                azimuth = None,
                                percentile = PERCENTILE_MAX,
                                coverage_factor=COVERAGE_FACTOR): 
            #number of uncertainty sources with an assumed gaussian distribution
            us_nrm_n = 1
            #number of uncertainty sources with an assumed rectangular distribution
            us_rct_n = 8
            
            
            #definining sensitivity and voltage as dependent or indipendent from temperature and angle
            #DEV NOTE 8/11/18: to be improved different su for t,a,aoi depending if characterisation or not. 
            #generation of entire dataframe with normal distribution
            arr_nrm = np.random.normal(loc=0,scale=1,size=(len(irradiance),simulations,us_nrm_n))
            arr_rct = np.random.uniform(low=0,high=1,size=(len(irradiance),simulations,us_rct_n))
            #round to 3 decimals 
            #DEV NOTE 9/11/18: to be checked sum 
            arr_mc = np.concatenate((arr_nrm,arr_rct),axis=2)
            
            arr_mc = arr_mc.round(decimals=MC_NORMAL_DECIMALS)
            
            
            #if percentile is not np.ndarray: 
            if isinstance(percentile,np.ndarray) == True:
                exp_results = pd.DataFrame(columns=percentile)
            elif isinstance(percentile,np.ndarray) == False:
                exp_results = []
  
            
            for i in range(0,len(irradiance)-1):
                unc_irr = []
                
                #if len(percentile)==1: 
                #        unc_irr = []
                #elif len(percentile) > 1 :
                #    unc_irr = pd.DataFrame(columns=list(percentile))
                
                for j in range(0,simulations-1):
                    
                    """2
                    a = 0                     
                    for k in range(0,unc_n-1):
                        if np.isnan(arr_mc[i,j,k]) == True:
                            print([i,j,k])
                        if np.isnan(arr_mc[i,j,k]) == False:
                            a = a + arr_mc[i,j,k]
                        print(a)                    
                    """
                    
                    #if all valid temperatures provided analyse responsitivity on temperature
                    if np.any(angle_of_incidence) is None:
                        angle_of_incidence= None
                    elif np.all(angle_of_incidence) is not None:
                        angle_of_incidence = angle_of_incidence[i]
                    if np.any(temperature) is None:
                        temperature = None
                    elif np.all(temperature) is not None:
                        #DEV NOTE 29/11/18: temperature and azimuth without index
                        temperature = temperature
                    if np.any(azimuth) is None:
                        azimuth = None
                    elif np.all(azimuth) is not None:
                        #DEV NOTE 29/11/18: temperature and azimuth without index
                        azimuth = azimuth
                    #calculate uncertainty 
                    unc_sim = self.get_uncertainty(
                            irradiance = irradiance.iloc[i],
                            temperature = temperature,
                            azimuth = azimuth,
                            angle_of_incidence = angle_of_incidence,
                            coverage_factor = coverage_factor,
                            mc_coeff =arr_mc[i,j,:])
                    #testing null by printing it 
                    if unc_sim is None:
                    #if np.isnan(unc_sim) == True:
                        print([i,j])
                        print([unc_sim,irradiance.iloc[i],arr_mc[i,j,:]])
                    
                    
 
                    unc_irr.append(unc_sim)
     
                #if percentile is not np.ndarray: 
                if isinstance(percentile,np.ndarray) == False:
                    #providing series if single percentile
                    exp_results.append((np.percentile(a=unc_irr,q=percentile)))
                    exp_results=pd.Series(exp_results)
                #elif percentile is np.ndarray:
                elif isinstance(percentile,np.ndarray) == True:
                    #providing dataframe if multiple percentile
                    dict_tmp = {}
                    for index,value in enumerate(percentile):
                        dict_tmp[value]=np.percentile(a=unc_irr,q=value)
                    exp_results=exp_results.append(dict_tmp,ignore_index=True)

                return exp_results

class SolarData: #DEV NOTE 5/12/19: class probably not needed, shold be Solar Library
    """function related to data completeness (e.g. missing days in records )"""
    #version 1 created on 14/8/17 by fm
    #version 1.1. now df
    #duplicate function not inserted 
    
    
    class Calibration:
        def __init__(self,location:pvlibint.SolarLibrary,method:str,calibration_factor:float):
            #DEV NOTE 16/11/18: irradiance to be replaced
            self.mth = method
            self.lct = location
            self.calibration_factor= calibration_factor
            
            #DEV NOTE 19/11/18: some values could be transferred from function to class
        
        def get_factor_df2(self,#improved version calculating median of other variables
                       df:pd.DataFrame,
                       reference_column:str, 
                       field_column:str,
                       datetime_column:str,
                       deviation_max=DEVIATION_MAX,
                       readings_min=CALIBRATION_READINGS,#21
                       iterations_max=ITERATIONS_MAXIMUM,
                       log=False,
                       median=False,#if median=True used instead of average
                       ):            
            df=df.rename(columns={reference_column:"vr_ji",field_column:"vf_ji",datetime_column:"datetime"})
            df[datetime_column]=pd.DatetimeIndex(df[datetime_column])
            df=df[(df["vr_ji"]>0)&(df["vf_ji"]>0)]  #filter only for positive values
            df=df.assign(f_ji=self.calibration_factor*df["vr_ji"]/df["vf_ji"]) #calculate factor for each point 
            df_tmp = df #establish temporary df
            valid = True #initialise break condition          
            if ("eurac" in self.mth): #check if distance min-max is not higher than 2%
                #6/1/20: is EURAC method or other?  #DEV NOTE 19/11/18: traditional irradiance checks should be added 
                if ((df.loc["","vr_ji"].max()-df.loc["","vr_ji"].min())/df.loc["","vr_ji"].mean()>deviation_max):
                    valid==False; print("Error: too high percentage deviation from min to max !!")
            if valid==True and ("9847" in self.mth) and ("1a" in self.mth): #ref ISO 9847 1a & others. 
                for i in range(0,iterations_max-1): #artificial limit to ammissible iterations 
                    if len(df_tmp) < readings_min: #close if not enough values 
                        if log==True:         #log always true               
                            print("aborted due to insufficient number of values: "+str(len(df_tmp)))
                            print(df_tmp)
                        valid=False
                        break
                    if median==False: f_j_tmp=self.calibration_factor*df_tmp["vr_ji"].sum()/df_tmp["vf_ji"].sum()#tmp calibration factor
                    elif median==True: f_j_tmp=self.calibration_factor*df_tmp["vr_ji"].median()/df_tmp["vf_ji"].median()               
                    df_tmp=df_tmp.assign(f_dev=df["f_ji"].map(lambda x: abs((x-f_j_tmp)/f_j_tmp))) #10/1/20 removed factor 100    
                    if df_tmp["f_dev"].max()<= deviation_max:             #stop iterations and provide results if value found 
                        if log==True:
                            print("f_j found: "+str(f_j_tmp)); print("std of f_j: "+str(df_tmp["f_dev"].std()))
                            valid=True
                        break
                    elif df_tmp["f_dev"].max()>deviation_max:
                        # new filter f_j 
                        df_tmp = df_tmp[(df_tmp["f_dev"]<=deviation_max)]
                        valid=False
                        #print("too high max deviation of: "+str(df_tmp["f_dev"].max()))                       
            if valid==False:
                #print("not valid series")
                return 1,None,None,None,None,None #16/1/20 updated to 1 for easy counting of errors
            elif valid == True:
                #DEV NOTE 8/1/19: df_tmp["f_dev"].std() not useful, better std of the series
                #return valid,f_j_tmp,df_tmp["f_dev"].std(),len(df_tmp),df_tmp
                df_tmp["r_ji"]=1/df_tmp["f_ji"]
                
                return 0,f_j_tmp,df_tmp["f_ji"].std()/f_j_tmp,df_tmp["r_ji"].std()/f_j_tmp,len(df_tmp),df_tmp #090620 relative variance 
                #valid Boolean, f_j_tmp scalar, 
        
        
        # 7/9/25 function to obtain calibration factor. Better to convert to obtain the responsitivity instead.
        def get_factor_df(self,
                       reference_series:pd.Series, 
                       field_series:pd.Series,
                       time_series:pd.Series,
                       deviation_max=DEVIATION_MAX,
                       readings_min=CALIBRATION_READINGS,#21
                       iterations_max=ITERATIONS_MAXIMUM,
                       log=False,
                       median=False
                       ):
            df = pd.DataFrame(
            {"vr_ji":reference_series,
             "vf_ji":field_series,
             "datetime":pd.DatetimeIndex(time_series)
             })   
            df=df[(df["vr_ji"]>0)&(df["vf_ji"]>0)]  #filter only for positive values
            df=df.assign(f_ji=self.calibration_factor*df["vr_ji"]/df["vf_ji"]) #calculate factor for each point 
            df_tmp = df #establish temporary df
            valid = True #initialise break condition          
            if ("eurac" in self.mth): #check if distance min-max is not higher than 2%
                #6/1/20: is EURAC method or other? 
                #DEV NOTE 19/11/18: traditional irradiance checks should be added 
                if ((reference_series.max()-reference_series.min())/reference_series.mean()>deviation_max):
                    valid==False
                    print("Error: too high percentage deviation from min to max !!")
            if valid==True and ("9847" in self.mth) and ("1a" in self.mth):
                #ref ISO 9847 1a & others. 
                for i in range(0,iterations_max-1): #artificial limit to ammissible iterations 
                    if len(df_tmp) < readings_min: #close if not enough values 
                        if log==True:         #log always true               
                            print("aborted due to insufficient number of values: "+str(len(df_tmp)))
                            print(df_tmp)
                        valid=False
                        break
                    f_j_tmp=self.calibration_factor*df_tmp["vr_ji"].sum()/df_tmp["vf_ji"].sum()
                    #17/1/20 test/compare with median? (@Tom)                    
                    #recalculate calibration if conditions ok 
                    #calculate relative standard deviation from estimated f_j for each point                         
                    #DEV NOTE 8/1/19: deviation from f_j_tmp not the contrary  
                    #df_tmp=df_tmp.assign(f_dev=df["f_ji"].map(lambda x: abs((x-f_j_tmp)/f_j_tmp)*100))   #BEFORE 10/1/20
                    df_tmp=df_tmp.assign(f_dev=df["f_ji"].map(lambda x: abs((x-f_j_tmp)/f_j_tmp))) #10/1/20 removed factor 100    
             #stop iterations and provide results if value found 
                    if df_tmp["f_dev"].max()<= deviation_max:
                        if log==True:
                            print("f_j found: "+str(f_j_tmp))
                            print("std of f_j: "+str(df_tmp["f_dev"].std()))
                            valid=True
                        break
                    elif df_tmp["f_dev"].max()>deviation_max:
                        # new filter f_j 
                        df_tmp = df_tmp[(df_tmp["f_dev"]<=deviation_max)]
                        valid=False
                        #print("too high max deviation of: "+str(df_tmp["f_dev"].max()))                       
            if valid==False:
                #print("not valid series")
                return 1,None,None,None,None #16/1/20 updated to 1 for easy counting of errors
            elif valid == True:
                #DEV NOTE 8/1/19: df_tmp["f_dev"].std() not useful, better std of the series
                #return valid,f_j_tmp,df_tmp["f_dev"].std(),len(df_tmp),df_tmp
                return 0,f_j_tmp,df_tmp["f_ji"].std()/f_j_tmp,len(df_tmp),df_tmp
                #valid Boolean, f_j_tmp scalar, 

    
    
    
    
    
    def timemaxzenithvariation(SolarLibrary:pvlibint.SolarLibrary,datetimeutc_values:np.ndarray,
                               maxangle=IEC61724_ALIGNMENT_TILT): #October 2019 to be improved   
        datetimeindex_tz = pd.DatetimeIndex(datetimeutc_values,tz=SolarLibrary.timezone,ambiguous='NaT',nonexistent='NaT')                                          
        dfout = pd.DataFrame(columns=['datetime_utc','date','aoi','datetime_e','aoi_e','transit'],index= datetimeindex_tz)
        #dfout.loc[:,"datetime_tz"]=datetimeindex_tz 
        dfout.loc[:,"datetime_utc"]=pd.DatetimeIndex(datetimeutc_values)
        dfout.loc[:,"aoi"]=SolarLibrary.getsolarseries(datetimeindex_tz,"aoi")["aoi"]
        dfout.loc[:,"transit"]=dfout.loc[:,"datetime_utc"].apply(lambda x:SolarLibrary.getsunrisesettransit([x])["suntransit"].values[0])
        for index in dfout.index:   
            if dfout.loc[index,"aoi"]<90:
                #Cannot compare tz-naive and tz-aware timestamps
                if dfout.loc[index,"datetime_utc"]<dfout.loc[index,"transit"]:cff=1
                elif dfout.loc[index,"datetime_utc"]>dfout.loc[index,"transit"]:cff=-1
                def timediffmax(deltasecond):
                    f = (SolarLibrary.getsolarseries([index],["aoi"])["aoi"][0]-
                    SolarLibrary.getsolarseries([index+dt.timedelta(seconds=int(deltasecond))],["aoi"])["aoi"][0]-
                    maxangle*cff)
                    return f
                guess = 600
                dm=fsolve(timediffmax,guess)
                dfout.loc[index,"datetime_e"]=dfout.loc[index,"datetime_utc"]+dt.timedelta(seconds=int(dm))
                dfout.loc[index,"aoi_e"]=SolarLibrary.getsolarseries([index+dt.timedelta(seconds=int(dm))],["aoi"])["aoi"][0]     
        return dfout 
    

    def daysflagging(SolarLibrary:pvlibint.SolarLibrary, #only disconnections & outliers, no clear sky flagging
                      daysoverview=pd.DataFrame,
                      hours_limit=HOURS_LIMIT,
                      coverage_factor=COVERAGE_FACTOR): 
        #return df_days_valid         
        surface_zenith = SolarLibrary.surface_zenith
        df_days = daysoverview
        df_days.rename(columns={
        "day":"day",
        "tmpstampcount":"tmpstampcount",
        "tmstampmin":"data_on",
        "tmstampmax":"data_off",
        "irradiancevalid":"irradiancevalid",
        "tmstamppeak":"peak_datetime",
        "irradiancemax":"peak_irr",
        },inplace=True)
        df_days.index=pd.DatetimeIndex(df_days.loc[:,"day"].values,dayfirst=True)
        timerstart=dt.datetime.now()
        df_days_valid=df_days[(df_days.missing==0)&(df_days.na==0)] #first filtering on missing or na values
        
        if surface_zenith==0: 
            sunpos = SolarLibrary.getsunrisesettransit(df_days_valid['day']) #Getting sunrise & sunset from SolarModels class  
            df_days_valid=df_days_valid.assign(sun_on = pd.DatetimeIndex(sunpos['sunrise'],tz=None))
            df_days_valid=df_days_valid.assign(sun_out =  pd.DatetimeIndex(sunpos['sunset'],tz=None))   
            df_days_valid=df_days_valid.assign(sun_peak =  pd.DatetimeIndex(sunpos['suntransit'],tz=None))                
        if surface_zenith!=0:
            sunpos = SolarLibrary.getsuninout(df_days_valid.index,freq="min")                
            df_days_valid=df_days_valid.assign(sun_on = pd.DatetimeIndex(sunpos['sun_on'],tz=None))
            df_days_valid=df_days_valid.assign(sun_out =  pd.DatetimeIndex(sunpos['sun_off'],tz=None))
            df_days_valid=df_days_valid.assign(sun_peak =  pd.DatetimeIndex(sunpos['sun_max'],tz=None))  
            df_days_valid=df_days_valid.assign(aoi_min=  pd.DatetimeIndex(sunpos['aoi_min'],tz=None))  
        parameters = list(["clearskypoa"])
        poa_global=  SolarLibrary.getsolarseries(df_days_valid['sun_peak'],outputs=parameters)
        df_days_valid=df_days_valid.assign(peak_irr_mod=poa_global.loc[:,"poa_global"].values) 
        #print("calculate daily cs-related indicators of delay/anticipation, disconnections & peaks")
        df_days_valid["on_delay"]=df_days_valid.apply(lambda x: tdhour(x["data_on"]-x["sun_on"]),axis=1)
        df_days_valid["off_anticipation"]=df_days_valid.apply(lambda x: tdhour(x["sun_out"]-x["data_off"]),axis=1)
        #df_days_valid["peak_delay"]=df_days_valid.apply(lambda x: 0.5*tdhour(x["peak_datetime_max"]-x["sun_peak"])+
        #0.5*tdhour(x["peak_datetime_min"]-x["sun_peak"]),axis=1)
        df_days_valid["peak_delay"]=df_days_valid.apply(lambda x: tdhour(x["peak_datetime"]-x["sun_peak"]),axis=1)#24/2/20 replaced only one value
        
        df_days_valid["peak_irr_dif"]=df_days_valid.apply(lambda x: x["peak_irr"]-x["peak_irr_mod"],axis=1)
        df_days_valid["after_off_hours"] = df_days_valid["data_on"].shift(-1)- df_days_valid["data_off"] #28/3/19 DEV no shift immediately
        df_days_valid["before_on_hours"] = df_days_valid["data_on"] - df_days_valid["data_off"].shift(1)            
        #df_days_valid["after_off_hours"]=df_days_valid["after_off_hours"].apply(lambda x: tdhour(x))
        df_days_valid["after_off_hours"]=df_days_valid["after_off_hours"].transform(lambda x: tdhour(x))
        #df_days_valid["before_on_hours"]=df_days_valid["before_on_hours"].apply(lambda x: tdhour(x)) #End of parameters calculation
        df_days_valid["before_on_hours"]=df_days_valid["before_on_hours"].transform(lambda x: tdhour(x))
        #LIMITS FOR DISCONNECTIONS:distance from expected values is higher than standard deviation *3 (Ralph also requested)   
        on_delay_limit = (df_days_valid.loc[df_days_valid.on_delay>0,"on_delay"].median()+
                df_days_valid.loc[df_days_valid.on_delay>0,"on_delay"].std()*coverage_factor)  #identification of suspect outliers
        off_anticipation_limit = (df_days_valid.loc[df_days_valid.off_anticipation>0,"off_anticipation"].median()+
                df_days_valid.loc[df_days_valid.off_anticipation>0,"off_anticipation"].std()*coverage_factor)
        peak_delay_limit = (df_days_valid.loc[:,"peak_delay"].median()+ df_days_valid.loc[:,"peak_delay"].std()*coverage_factor)
        peak_irr_dif_limit = (df_days_valid.loc[df_days_valid.peak_irr_dif>0,"peak_irr_dif"].median()+
                df_days_valid.loc[df_days_valid.peak_irr_dif>0,"peak_irr_dif"].std()*coverage_factor)            
        #DEV NOTE 4/4/19: median could be a systematic effect     
        if hours_limit is None: #otherwise if too small all classified as big
            onoffdiff = df_days_valid["sun_out"] - df_days_valid["sun_on"]
            #onoffdiff = onoffdiff.apply(lambda x: tdhour(x))
            onoffdiff = onoffdiff.transform(lambda x: tdhour(x))
            hours_limit = 24 - onoffdiff.min()
        #print("flagging for daily cs-related indicators")
        df_days_valid.loc[(df_days_valid["after_off_hours"] > hours_limit),"anticipation_type"]= 3 # 17/2/20 "long" before
        df_days_valid.loc[(df_days_valid["before_on_hours"] > hours_limit),"delay_type"]= 3         
        df_days_valid.loc[((df_days_valid["off_anticipation"] > off_anticipation_limit) &
        (df_days_valid["on_delay"].shift(-1) > on_delay_limit) &
        (df_days_valid["after_off_hours"] <= hours_limit)),"anticipation_type"]= 2 # 17/2/20 "overnight" before
        df_days_valid.loc[((df_days_valid["on_delay"] > on_delay_limit) &
        (df_days_valid["off_anticipation"].shift(1) > off_anticipation_limit) &
        (df_days_valid["before_on_hours"] <= hours_limit)),"delay_type"]= 2  #overnight disconnections end
        df_days_valid.loc[(( df_days_valid["off_anticipation"] > off_anticipation_limit) &
        (df_days_valid ["on_delay"].shift(-1) <= on_delay_limit) &
        (df_days_valid["after_off_hours"] <= hours_limit)),"anticipation_type"]= 1 # 17/2/20 "small before" #anticipation only end                                                   
        df_days_valid.loc[(( df_days_valid["on_delay"] > on_delay_limit) &  #yesterday value for off_anticipation. 
        (df_days_valid["off_anticipation"].shift(1) <= off_anticipation_limit) &
        (df_days_valid["before_on_hours"] <= hours_limit)),"delay_type"]= 1 #delayed only end          
        df_days_valid.loc[(df_days_valid["peak_delay"] > peak_delay_limit),"peak_delay_outlier"]= 1 #peak delay outliers end
        df_days_valid.loc[(df_days_valid["peak_irr_dif"] > peak_irr_dif_limit),"irradiance_outlier"]= 1 #peak irradiance outliers end            
                       
        print("daily disconnections & outliers flagging: "+str(dt.datetime.now()-timerstart))

        return df_days_valid
         
            
           
    
    def stabilityflagging(SolarLibrary:pvlibint.SolarLibrary,#8/5/20 reviewed #17/2/20: DOES NOT included disconnections analysis
                          datetimeindex_utc:pd.DatetimeIndex, #timezone:str,
                          irradiance_values:np.ndarray,                      
                          resolutions:np.ndarray,
                          periods:np.ndarray,
                          counts:np.ndarray,
                          irr_min=IRR_MIN,
                          aoi_max=AOI_MAX,
                          kcs_range=[1-KCS_DEV_LMT,1+KCS_DEV_LMT], #kcs max deviation
                          kcs_cv_max=KCS_CV_LMT, #used also for the power
                          pwr_cv_max=PWR_CV_LMT,
                          pearson_min=PEARSON_LMT, #7/5/20: not working removed #Pearson only for higher resolution 
                          beam=False,#provide beam irradiance for the flagging
                          power_values=None,
                          keep_invalid=False #if True filter out only irradiances < irr_min 
                          ):
        #return df,dfsout #return original df flagged 
        
        tb_rpc=pd.DataFrame({"resolution":resolutions,"period":periods,"count":counts})
        if (power_values is None) or (len(power_values)!=len(irradiance_values)): pwr_chk=False
        df  = pd.DataFrame({'datetime': datetimeindex_utc,'irradiance': irradiance_values},index= datetimeindex_utc)
        if pwr_chk!=False: df.loc[:,"power"]=power_values
        #print("filter on max. kcs deviation of "+str(kcs_range)+" & min. irradiance of"+str(IRR_MIN))
        df_vld = df #initialising cleaned dataframe
        df_vld.loc[:,"irradiance"] = df_vld.loc[:,"irradiance"].fillna(0)
        if pwr_chk!=False: df_vld.loc[:,"power"]=df_vld.loc[:,"power"].fillna(0)
        if keep_invalid==True:df_vld = df_vld.loc[(df_vld.loc[:,"irradiance"]>IRR_MIN)]          
        print("theoretical clear sky irradiance estimation")
        datetimeindex_utc=df_vld.loc[:,"datetime"] #using only filtered datetime       
        if beam==False: parameters = list(["clearskypoa","aoi"]) #define requested parameters
        elif beam==True: parameters = list(["clearsky","aoi"])
        df_par =  SolarLibrary.getsolarseries(datetimeindex_utc,parameters) #modelling parameters   
        if beam==False: df_vld.loc[:,"irradiance_cs"]=df_par.loc[:,"poa_global"]
        elif beam==True: df_vld.loc[:,"irradiance_cs"]=df_par.loc[:,"dni"]
        df_vld.loc[:,"extraradiation"]=df_par.loc[:,"extraradiation"]
        df_vld.loc[:,"airmassrelative"]=df_par.loc[:,"airmassrelative"]
        df_vld.loc[:,"aoi"]=df_par.loc[:,"aoi"]
        df_vld.loc[:,"kcs"]=df_vld.apply(lambda x:x["irradiance"]/x["irradiance_cs"] if x["irradiance_cs"]>0 else 0,axis=1) #DEV NOTE 2/5/19 tmp solution # *** transforms cannot produce aggregated results
        
        #len_old=len(df)     #7/5/2020 not needed if wrong stop anyway
        #df_merged=df.merge(right=df_vld,left_on=['datetime','irradiance'],right_on=['datetime','irradiance'],how='left',sort=True) #parameters added for check  
        #if len(df_merged)>len_old: print([min(df.loc[:,"datetime"]),max(df.loc[:,"datetime"])]), sys.exit("df increased")
        #elif len(df_merged)==len_old: 
            #df = df_merged
            #df.index=df.datetime
        df_vld = df_vld.loc[((df_vld.kcs>kcs_range[0])&(df_vld.kcs<kcs_range[1])&(df_vld.aoi<aoi_max))]     
        df.loc[(df.loc[:,"datetime"].isin(df_vld["datetime"])==True),"invalid"]=0 #flagging of analysed values in native df
        df.loc[(df.loc[:,"datetime"].isin(df_vld["datetime"])==False),"invalid"]=1                
        df.loc[:,"i_unstable"]=0 #initialise stability columns as True
        if pwr_chk!=False: df.loc[:,"p_unstable"]=0
        if len(df_vld)==0: 
            print("no valid records")
            dfsout=dict() #empty
        elif len(df_vld)!=0:
            resolutions=tb_rpc.loc[:,"resolution"]
            resolutions=resolutions.drop_duplicates()       
            dfs4rls=dict() #dfs obtained by averaging 7/1/20 what about median instead ?
            dfsout=dict() #dfs merging df with grouping results
            print("Averaging values for different resolutions")
            for rsl in resolutions: #creating dfs for the different requested resolutions
                if rsl!=1:
                    print("averaging for resolution "+str(rsl)+"s")                    
                    if keep_invalid==True: grp = df.groupby([pd.Grouper(freq=str(rsl)+"s", key='datetime')])#identifying peak before cs series to distinguish morning/afternoon
                    if keep_invalid==False: grp = df_vld.groupby([pd.Grouper(freq=str(rsl)+"s", key='datetime')])#identifying peak before cs series to distinguish morning/afternoon
                    if pwr_chk!=False:  
                        agg = grp.agg({'airmassrelative':[np.median],'extraradiation':[np.median],'aoi': [np.median],
                        'irradiance':[np.median],'irradiance_cs':[np.median],'datetime': [max],'power':[np.median]})
                        df_rsl=pd.DataFrame({"airmassrelative":agg.loc[:,("airmassrelative","mean")].values,
                           "extraradiation":agg.loc[:,("extraradiation","mean")].values,
                           "aoi":agg.loc[:,("aoi","mean")].values,
                           "irradiance":agg.loc[:,("irradiance","mean")].values,
                           "irradiance_cs":agg.loc[:,("irradiance_cs","mean")].values,
                           "power":agg.loc[:,("power","mean")].values,
                           "datetime":agg.loc[:,("datetime","max")].values})
                    elif pwr_chk==False:
                        agg = grp.agg({'airmassrelative':[np.median],'extraradiation':[np.median],'aoi': [np.median],
                        'irradiance':[np.median],'irradiance_cs':[np.median],'datetime': [max]})
                        df_rsl=pd.DataFrame({"airmassrelative":agg.loc[:,("airmassrelative","mean")].values,
                           "extraradiation":agg.loc[:,("extraradiation","mean")].values,
                           "aoi":agg.loc[:,("aoi","mean")].values,
                           "irradiance":agg.loc[:,("irradiance","mean")].values,
                           "irradiance_cs":agg.loc[:,("irradiance_cs","mean")].values,
                           "datetime":agg.loc[:,("datetime","max")].values})           
                    #df_rsl["kcs"]=df_rsl.transform(lambda x:x["irradiance"]/x["irradiance_cs"] if x["irradiance_cs"]>0 else 0,axis=1)#DEV NOTE 2/5/19 tmp solution                 
                    df_rsl["kcs"]=df_rsl.apply(lambda x:x["irradiance"]/x["irradiance_cs"] if x["irradiance_cs"]>0 else 0,axis=1)#DEV NOTE 2/5/19 tmp solution                 
    
                    
                    """ #4/1/20 replaced with pearson to avoid sqr error
                    df_rsl["i_cs"] = df_rsl["irradiance"] * df_rsl["irradiance_cs"]
                    df_rsl["i2"] = df_rsl["irradiance"] * df_rsl["irradiance"]
                    df_rsl["cs2"] = df_rsl["irradiance_cs"] * df_rsl["irradiance_cs"] """
                    dfs4rls[rsl]=df_rsl.dropna(how='any',subset=["datetime"]) #clean nan first# https://github.com/pandas-dev/pandas/issues/9697           
            print("Stabilisation analysis for different periods")
            for index in tb_rpc.index:            
                srs_rsl= tb_rpc.loc[index,"resolution"]
                srs_prd= tb_rpc.loc[index,"period"]
                counts= tb_rpc.loc[index,"count"]
                prs_chk=False
                #if ((srs_rsl>=60) and (srs_prd>=srs_rsl*counts)): prs_chk=True #7/5/20 pearson check removed
                srs_cnt= tb_rpc.loc[index,"count"]
                print("Calculation for resolution "+str(srs_rsl)+" period"+str(srs_prd)+" count"+str(srs_cnt))
                if srs_rsl!=1:df_srs=dfs4rls[srs_rsl] 
                elif srs_rsl==1 and keep_invalid==False: df_srs=df_vld #1s assumed lowest
                elif srs_rsl==1 and keep_invalid==True: df_srs=df #1s assumed lowest 
                print("grouping for "+str(srs_prd)+"s period stability")
                
                #df_srs.loc[:,"datetime"]=df_srs.loc[:,"datetime"].apply(lambda x:x-np.timedelta64(int(srs_prd/2),'s'))#8/5/20 shift to get stability middle: higher AOI (75?) required for &)
                
                grp = df_srs.groupby([pd.Grouper(freq=str(srs_prd)+"s", key='datetime')])#identifying peak before cs series to distinguish morning/afternoon
                parameters_index= ['kcs','kcs_std','srs_len','aoi','irradiance','irradiance_cs','datetime']
                def parameters(x,parameters_index):
                    d={}
                    d['srs_len']=x['irradiance'].count()
                    d["kcs"]=x["kcs"].mean()
                    d["kcs_std"]=x["kcs"].std()
                    d["aoi"]=x["aoi"].mean()
                    d["irradiance"]=x["irradiance"].mean()
                    d["irradiance_cs"]=x["irradiance_cs"].mean()
                    d["datetime"]=x["datetime"].max()
                    if prs_chk!=False: #pearson only if enough low resolution (cs minute -> prs_d_cs ==0) #https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
                        d['prs_n']=(x['irradiance'].count()*(x['irradiance']*x['irradiance_cs']).sum()-x['irradiance'].sum()*x['irradiance_cs'].sum())
                        d['prs_d']=(x['irradiance'].count()*(x['irradiance']*x['irradiance']).sum())-(x['irradiance'].sum())**2
                        d['prs_d_cs']=(x['irradiance_cs'].count()*(x['irradiance_cs']*x['irradiance_cs']).sum())-(x['irradiance_cs'].sum())**2
                        parameters_index=parameters_index.extend(['prs_n','prs_d','prs_d_cs'])
                    if pwr_chk!=False: 
                        d["power"]=x["power"].mean()
                        d["power_std"]=x["power"].std()
                        parameters_index=parameters_index.extend(['power','power_std'])
                    return pd.Series(d,index=parameters_index)              
                df_grp=grp.apply(parameters,parameters_index) #6/1/20 could apply be replaced with faster transform? 
                df_grp.loc[:,"kcs_cv"]=df_grp.apply(lambda x: (x["kcs_std"]/x["kcs"]) if x["kcs"]!=0 else np.nan,axis=1)
                if prs_chk!=False:df_grp.loc[:,"pearson"]=df_grp.apply(lambda x: (x["prs_n"]/((x["prs_d"]**0.5)/(x["prs_d_cs"]**0.5))) if (x["prs_d"]>0 and x["prs_d_cs"]>0) else np.nan, axis=1)
                if pwr_chk!=False:df_grp.loc[:,"power_cv"]=df_grp.apply(lambda x: (x["power_std"]/x["power"]) if x["power"]!=0 else np.nan,axis=1)
                df_grp.index=df_grp.datetime
                msg="identify stable series based on kcs std, valid values count, power std"
                df_i_stable = df_grp.loc[((df_grp.kcs_cv>-kcs_cv_max)&(df_grp.kcs_cv<kcs_cv_max)& #Check conditions for each series
                                     (df_grp.kcs_cv!=0)&
                                     (df_grp.srs_len>=srs_cnt)
                                     ),:]
                if prs_chk!=False: 
                    df_i_stable = df_i_stable.loc[(df_i_stable.pearson>=pearson_min),:]
                    msg=msg+" & Pearson"
                    print(msg)
                datetimes_i_stable = df_i_stable.loc[:,"datetime"].values
                df_grp.loc[(df_grp.loc[:,"datetime"].isin(datetimes_i_stable)==True),"i_unstable"]=0
                df_grp.loc[(df_grp.loc[:,"datetime"].isin(datetimes_i_stable)==False),"i_unstable"]=1           
                if pwr_chk!=False: 
                    datetimes_p_stable = df_grp.loc[((df_grp.power_cv>-pwr_cv_max)&(df_grp.power_cv<pwr_cv_max)),
                    "datetime"].values
                    df_grp.loc[(df_grp.loc[:,"datetime"].isin(datetimes_p_stable)==True),"p_unstable"]=0
                    df_grp.loc[(df_grp.loc[:,"datetime"].isin(datetimes_p_stable)==False),"p_unstable"]=1 
                if srs_prd==86400:srs_lbl=str("1d") #simplifying labels
                elif int(srs_prd/3600)>0 and srs_prd<86400:srs_lbl=str(int(srs_prd/3600))+"h"
                elif int(srs_prd/60)>0 and srs_prd<3600:srs_lbl=str(int(srs_prd/60))+"m"
                # print("adding flagging to the summary file")
                if rsl!=1: grp4df=df.groupby([pd.Grouper(freq=str(srs_prd)+"s", key='datetime')]) #grouping original               
                elif rsl==1: grp4df=grp            
                df.loc[:,"datetime"+srs_lbl]=grp4df["datetime"].transform(max) #series ending assignment    
                df.loc[(df.loc[:,"datetime"+srs_lbl].isin(datetimes_i_stable)==True),"i_unstable"+srs_lbl]= 0
                df.loc[(df.loc[:,"datetime"+srs_lbl].isin(datetimes_i_stable)==False),["i_unstable"+srs_lbl,"i_unstable"]]=1
                if pwr_chk!=False: 
                    df.loc[(df.loc[:,"datetime"+srs_lbl].isin(datetimes_p_stable)==True),"p_unstable"+srs_lbl]=0
                    df.loc[(df.loc[:,"datetime"+srs_lbl].isin(datetimes_p_stable)==False),["p_unstable"+srs_lbl,"p_unstable"]]=1  
                    #if rsl!=1:
                grp4df_srs=df_srs.groupby([pd.Grouper(freq=str(srs_prd)+"s", key='datetime')]) #grouping original               
                #elif rsl==1: grp4df=grp
                df_srs.loc[:,"datetime"+srs_lbl]=grp4df_srs["datetime"].transform(max)#series ending 
                df_srs.loc[(df_srs.loc[:,"datetime"+srs_lbl].isin(datetimes_i_stable)==True),"i_unstable"+srs_lbl]=0
                df_srs.loc[(df_srs.loc[:,"datetime"+srs_lbl].isin(datetimes_i_stable)==False),"i_unstable"+srs_lbl]=1
                if pwr_chk!=False: 
                    df_srs.loc[(df_srs.loc[:,"datetime"+srs_lbl].isin(datetimes_p_stable)==True),"p_unstable"+srs_lbl]=0
                    df_srs.loc[(df_srs.loc[:,"datetime"+srs_lbl].isin(datetimes_p_stable)==False),"p_unstable"+srs_lbl]=1                           
                df_grp=df_grp.add_suffix(srs_lbl)    
                if pwr_chk==False: 
                    df_anl=df_srs.merge(right=df_grp, #merge larger specific df with shorter grouping df
                    left_on=['datetime'+srs_lbl,"i_unstable"+srs_lbl],
                    right_on=['datetime'+srs_lbl,"i_unstable"+srs_lbl],
                    how='left',
                    sort=True)
                if pwr_chk!=False: 
                    df_anl=df_srs.merge(right=df_grp,
                    left_on=['datetime'+srs_lbl,"i_unstable"+srs_lbl,"p_unstable"+srs_lbl],
                    right_on=['datetime'+srs_lbl,"i_unstable"+srs_lbl,"p_unstable"+srs_lbl],
                    how='left',
                    sort=True)   
                dfsout[str(srs_rsl)+"_"+str(srs_prd)+"_"+str(srs_cnt)]=df_anl
                #5/12/19 future dev note: averages for next resolution could also be used 
        return df,dfsout #return original df flagged 
    
    
    
    
    
    
    
    

    def datesflagging(SolarLibrary:pvlibint.SolarLibrary, 
                      datetimeutc_values:np.ndarray, #timezone:str,
                      irradiance_values:np.ndarray, #no index, len should be same as datetimeindex,
                      Irradiance_Uncertainty=None,
                      timeresolution=60, #12/5/20 name error in np.timedelta64 to be solved
                      hours_limit=HOURS_LIMIT,
                      coverage_factor=COVERAGE_FACTOR,#Ralph asked coverage factor of 3 but no reason to do it
                      series_count=SERIES_COUNT_MIN,
                      series_values=CALIBRATION_READINGS-1, #20
                      irr_min=IRR_MIN,
                      aoi_max=AOI_MAX,
                      kcs_dev_lmt=KCS_DEV_LMT,
                      pearson_min=PEARSON_LMT, #Pearson only for higher resolution
                      uncertainty=True
                      ):
        # return df_vld, df_srs, df_days 
        # 12/5/20 some df_days grouped results are based on series, other on valid values
          
        #DEV NOTE 25/11/19  replace average with spot value, add as option            
        #DEV NOTE 25/11/19 daysaving passage to be added 
        srs_prd = (int(timeresolution*series_values/60)+((timeresolution*series_values)%60>0))*60 #seconds #DEV NOTE 19/2/20 to be reviewed        
        surface_zenith = SolarLibrary.surface_zenith
        #surface_azimuth = SolarLibrary.surface_azimuth #DEV NOTE 15/4/19 not needed for now (always south)                        
        df_input  = pd.DataFrame({   
        'datetime': datetimeutc_values,
        'irradiance': irradiance_values})    
        df_completeness, df_days_completeness=datacompleteness(dataframe=df_input,timeresolutionmax=timeresolution)  
        df_vld=df_completeness.loc[df_completeness.irradiance>irr_min] #12/5/20 filter needed to avoid negative, more meaningful pearson, irradiation comparison and delay/premature
        if len(df_vld)!=0:  
            timerstart = dt.datetime.now() 
            parameters = list(["clearskypoa","aoi"]) #define requested parameters
            datetimeindex_utc_vld = pd.DatetimeIndex(df_vld.loc[:,"datetime"].values,ambiguous='NaT')            
            df_par =  SolarLibrary.getsolarseries(datetimeindex_utc_vld,parameters) #modelling parameter
            df_vld  = pd.DataFrame({   
            'datetime': df_vld.loc[:,"datetime"].values,
            'date':  df_vld.loc[:,"date"].values,
            'irradiance': df_vld.loc[:,"irradiance"].values, 
            'irradiance_cs': df_par.loc[:,"poa_global"],
            'extraradiation': df_par.loc[:,"extraradiation"],
            'airmassrelative': df_par.loc[:,"airmassrelative"],
            'aoi':df_par.loc[:,"aoi"], 
            'azimuth':df_par.loc[:,"azimuth"],
            'zenith':df_par.loc[:,"zenith"]                
            },index=datetimeindex_utc_vld)  
            df_vld["kcs"]=df_vld.apply(lambda x:x["irradiance"]/x["irradiance_cs"] if x["irradiance_cs"]>0 else 0,axis=1)#DEV NOTE 21/5/19: irradiance difference used instead                
            df_vld["irradiance_difference"] = (df_vld["irradiance"]-df_vld["irradiance_cs"])
            df_vld["i_cs"] = df_vld["irradiance"] * df_vld["irradiance_cs"]
            df_vld["i2"] = df_vld["irradiance"] * df_vld["irradiance"]
            df_vld["cs2"] = df_vld["irradiance_cs"] * df_vld["irradiance_cs"]   
            print("importing solar parameters & calculating parameters for Pearson: "+str(dt.datetime.now()-timerstart))#12/5/20 Pearson for all valid values regardless of aoi
            timerstart = dt.datetime.now() 
            grp_day = df_vld.groupby([pd.Grouper(freq="1D", key='datetime')])#identifying peak before cs series to distinguish morning/afternoon
            i2 = grp_day["i2"]
            cs = grp_day["irradiance_cs"]
            cs2 = grp_day["cs2"]
            i_cs = grp_day["i_cs"]  
            irradiance = grp_day['irradiance']  
            rows = irradiance.count()
            pearson_day = ((rows*i_cs.sum()-irradiance.sum()*cs.sum())/rows/
            np.sqrt((i2.sum()-irradiance.sum()**2/rows)*
            (cs2.sum()-cs.sum()**2/rows))) 
            date = grp_day['date'] 
            df_p = df_vld.loc[:,["datetime","irradiance","date"]]
            grp_p_day = df_p.groupby('date',as_index=True)
            datetime_p = grp_p_day["datetime"]
            irradiance_p = grp_p_day['irradiance']   
            df_peak_d = grp_p_day.apply(lambda x: x[x["irradiance"]==x["irradiance"].max()]) 
            grp_peak_d = df_peak_d.groupby([pd.Grouper(freq="1D",key="datetime")])
            datetime_peak = grp_peak_d['datetime'] 
            #print("filter by defined acceptable deviation of Kcs from 1")
            if Irradiance_Uncertainty is not None:
                def unc(irradiance_total,zenith,azimuth):                 
                    u=Irradiance_Uncertainty.get_uncertainty_gum(irradiance_total=irradiance_total,
                                                                 diffuse_fraction=0,
                                                                 zenith=zenith,
                                                                 azimuth=azimuth,
                                                                 surface_zenith=SolarLibrary.surface_zenith,
                                                                 surface_azimuth=SolarLibrary.surface_azimuth,
                                                                 temperature=None)
                    return u    
                df_vld["irradiance_uncertainty"] = np.vectorize(unc)(df_vld['irradiance'], df_vld['zenith'], df_vld["azimuth"])
                df_clear = df_vld.loc[(abs(df_vld.irradiance_difference)<=df_vld.irradiance_uncertainty+df_vld.irradiance*CLEARSKY_ERROR)&(df_vld.aoi<aoi_max),:]
            elif Irradiance_Uncertainty is None:
                df_clear = df_vld.loc[(df_vld.aoi<=aoi_max)&
                                      (df_vld.kcs<=1+kcs_dev_lmt)& #g2/6/20 generic filter added
                                      (df_vld.kcs>=1-kcs_dev_lmt),
                                      :]
            #["irradiance","datetime","kcs","aoi","date","azimuth","zenith","irradiance_uncertainty"]] #data for series grouping but also irradiance uncertainty kept            
            #print("group by series of defined period")
            df_clear.loc[:,"datetime"]=df_clear.loc[:,"datetime"].apply(lambda x:x-np.timedelta64(int(srs_prd/2),'s')) #shift to center in one minute before groupby 21/5/20 issue later when mergeing with df_vld
    
            grp_srs = df_clear.groupby([pd.Grouper(freq=str(srs_prd)+"s", key='datetime',label='right')])#label left
            
            
            agg_srs = grp_srs.agg({
                    'kcs': [np.std,np.median,len], #all mean replaced with median
                    'irradiance_difference': [np.std,np.median],
                    'aoi': [min],
                    'irradiance':[np.median],
                    'date': [max],
                    'azimuth':[np.median],
                    'zenith':[np.median]
                    })
            agg_srs = agg_srs.dropna(thresh=2) #11/5/20 threeshold for na values
            df_srs = pd.DataFrame({
                   "kcs_s_median": agg_srs.loc[:,("kcs","median")].values,
                   "kcs_s_std":agg_srs.loc[:,("kcs","std")].values,
                   "kcs_s_len":agg_srs.loc[:,("kcs","len")].values, 
                   "irr_dif_s_std":agg_srs.loc[:,("irradiance_difference","std")].values,
                   "irr_dif_s_median":agg_srs.loc[:,("irradiance_difference","median")].values,                    
                   "aoi":agg_srs.loc[:,("aoi","min")].values, #23/2/20 minimum used for filtering but mean could be more correct
                   "date": agg_srs.loc[:,("date","max")].values,
                   "irradiance":agg_srs.loc[:,("irradiance","median")].values,
                   "datetime":agg_srs.index,
                   "azimuth":agg_srs.loc[:,("azimuth","median")].values,
                   "zenith":agg_srs.loc[:,("zenith","median")].values
                   })
            df_srs.index=df_srs.datetime
         
            df_srs_columns=["datetime","kcs_s_std","kcs_s_median","kcs_s_len","date","irradiance","aoi","azimuth","zenith","irr_dif_s_std","irr_dif_s_median"]#27/11/19    
            
            df_clear.loc[:,"datetime_s"]=grp_srs["datetime"].transform(max)
            df_clear.loc[:,"datetime_s"]=df_clear.loc[:,"datetime_s"].apply(lambda x:x+np.timedelta64(int(timeresolution),'s')) #tmp solution
            
                        
            df_calibration=df_clear
            df_len=0 #initialise
            for i in range(0,ITERATIONS_MAXIMUM-1):
                if len(df_calibration)==df_len: break
                elif len(df_calibration)!=df_len: 
                    df_len=len(df_calibration)
                    grp_srs_calibration = df_calibration.groupby([pd.Grouper(freq=str(srs_prd)+"s", key='datetime',label='right')])#label left
                    df_calibration.loc[:,"kcs_s_len"]=grp_srs_calibration["kcs"].transform(len)
                    #df_calibration.loc[:,"kcs_s_median"]=grp_srs_calibration["kcs"].transform(np.median) 
                    df_calibration.loc[:,"kcs_s_median"]=grp_srs_calibration["kcs"].transform(np.median)
                    df_calibration=df_calibration.loc[(df_calibration.kcs_s_len>=series_values)&
                                          (abs(1-(df_calibration.kcs/df_calibration.kcs_s_median))<=KCS_CV_LMT),
                                          :]
            
            agg_srs_calibration = grp_srs_calibration.agg({'kcs': [np.median,len]})#1/6/20 tmp solution
            agg_srs_calibration = agg_srs_calibration.dropna(thresh=2)
            df_srs_clear = df_srs[df_srs.loc[:,"datetime"].isin(agg_srs_calibration.index)]

            grp_s_day = df_srs.loc[:,df_srs_columns].groupby([pd.Grouper(freq="1D", key='datetime')]) #2/6/20 grouping including also not clear series
            #grp_s_day = df_srs_clear.loc[:,df_srs_columns].groupby([pd.Grouper(freq="1D", key='datetime')])
            
            kcs_s_valid_std = grp_s_day["kcs_s_std"]
            kcs_s_valid_median = grp_s_day["kcs_s_median"]
            irr_dif_s_valid_std = grp_s_day["irr_dif_s_std"]
            irr_dif_s_valid_median = grp_s_day["irr_dif_s_median"] 
            #print("series grouping & cs identification: "+str(dt.datetime.now()-timerstart)); timerstart = dt.datetime.now()
            df_days = pd.DataFrame({'data_on': datetime_p.min().dropna(), #Group per day, accounting median parameters of series
            'data_off':datetime_p.max().dropna(),
            'peak_datetime_min':datetime_peak.min().dropna(),
            'peak_datetime_max':datetime_peak.max().dropna(),       
            'peak_irr':irradiance_p.max().dropna(),           
            'date':date.min().dropna(),
            'pearson':pearson_day.dropna(), #used for clear sky 
            'kcs_std_s_d':kcs_s_valid_std.median().dropna(),
            'kcs_median_s_d':kcs_s_valid_median.median().dropna(),
            'irr_dif_s_std':irr_dif_s_valid_std.median().dropna(),
            'irr_dif_s_median':irr_dif_s_valid_median.median().dropna(),            
            'series_d': kcs_s_valid_median.count().dropna(), 
            'irradiation_d':irradiance.sum().dropna()*timeresolution/3600,
            'irradiation_cs':cs.sum().dropna()*timeresolution/3600
            },
            index=date.min().dropna())#dropna to match different len & for getsunrisetransit    
            #df_days=df_days.transform(lambda x: x[x["irradiation_d"]==x["kcs_std"].min()])
            print("estimating irradiance peak and datetimes for sunrise, sunset and peak")
            if surface_zenith==0: 
                sunpos = SolarLibrary.getsunrisesettransit(df_days['date']) #Getting sunrise & sunset from SolarModels class  
                df_days=df_days.assign(sun_on = pd.DatetimeIndex(sunpos['sunrise'],tz=None))
                df_days=df_days.assign(sun_out =  pd.DatetimeIndex(sunpos['sunset'],tz=None))   
                df_days=df_days.assign(sun_peak =  pd.DatetimeIndex(sunpos['suntransit'],tz=None))                
            if surface_zenith!=0:
                sunpos = SolarLibrary.getsuninout(df_days.loc[:,'date'],freq="min")                
                df_days=df_days.assign(sun_on = pd.DatetimeIndex(sunpos['sun_on'],tz=None))
                df_days=df_days.assign(sun_out =  pd.DatetimeIndex(sunpos['sun_off'],tz=None))
                df_days=df_days.assign(sun_peak =  pd.DatetimeIndex(sunpos['sun_max'],tz=None))  
                df_days=df_days.assign(aoi_min=  pd.DatetimeIndex(sunpos['aoi_min'],tz=None))  
            parameters = list(["clearskypoa"])
            poa_global=  SolarLibrary.getsolarseries(df_days['sun_peak'],outputs=parameters)
            df_days=df_days.assign(peak_irr_mod=poa_global.loc[:,"poa_global"].values) 
            #print("calculating daily cs-related indicators of delay/anticipation, disconnections & peaks")
            df_days["on_delay"]=df_days.apply(lambda x: tdhour(x["data_on"]-x["sun_on"]),axis=1)
            df_days["off_anticipation"]=df_days.apply(lambda x: tdhour(x["sun_out"]-x["data_off"]),axis=1)
            df_days["peak_delay"]=df_days.apply(lambda x: 0.5*tdhour(x["peak_datetime_max"]-x["sun_peak"])+
            0.5*tdhour(x["peak_datetime_min"]-x["sun_peak"]),axis=1)
            df_days["peak_difference"]=df_days.apply(lambda x: x["peak_irr"]-x["peak_irr_mod"],axis=1)
            df_days["after_off_hours"] = df_days["data_on"].shift(-1)- df_days["data_off"] #28/3/19 DEV no shift immediately
            df_days["before_on_hours"] = df_days["data_on"] - df_days["data_off"].shift(1)            
            df_days["after_off_hours"]=df_days["after_off_hours"].transform(lambda x: tdhour(x))
            df_days["before_on_hours"]=df_days["before_on_hours"].transform(lambda x: tdhour(x))
            df_days["irradiation_difference"]=df_days.apply(lambda x: x["irradiation_d"]-x["irradiation_cs"],axis=1)
            print("defining limits for outliers")#12/5/20 relative and not absolute to cope with possible misalignment/local conditions (surrounding mountains)
            on_delay_lmt = (df_days.loc[df_days.on_delay>0,"on_delay"].median()+
                    df_days.loc[df_days.on_delay>0,"on_delay"].std()*coverage_factor)  #identification of suspect outliers
            off_anticipation_lmt = (df_days.loc[df_days.off_anticipation>0,"off_anticipation"].median()+
                    df_days.loc[df_days.off_anticipation>0,"off_anticipation"].std()*coverage_factor)
            peak_delay_lmt_max = (df_days.loc[df_days.peak_delay!=0,"peak_delay"].median()+ df_days.loc[:,"peak_delay"].std()*coverage_factor)
            peak_delay_lmt_min = (df_days.loc[df_days.peak_delay!=0,"peak_delay"].median()- df_days.loc[:,"peak_delay"].std()*coverage_factor)
            peak_difference_lmt_max = df_days.loc[df_days.peak_difference!=0,"peak_difference"].median()+df_days.loc[:,"peak_difference"].std()*coverage_factor
            peak_difference_lmt_min = df_days.loc[df_days.peak_difference!=0,"peak_difference"].median()-df_days.loc[:,"peak_difference"].std()*coverage_factor
            irradiation_difference_lmt_max = df_days.loc[df_days.irradiation_difference!=0,"irradiation_difference"].median()+df_days.loc[:,"irradiation_difference"].std()*coverage_factor
            irradiation_difference_lmt_min = df_days.loc[df_days.irradiation_difference!=0,"irradiation_difference"].median()-df_days.loc[:,"irradiation_difference"].std()*coverage_factor
            #if hours_limit is None: #otherwise if too small all classified as big #not used since default value
            #    onoffdiff = df_days["sun_out"] - df_days["sun_on"]
            #         onoffdiff = onoffdiff.transform(lambda x: tdhour(x))
            #   hours_lmt = 24 - onoffdiff.min()
            print("flagging for daily cs-related indicators") #higher value higher flagging
            df_days.loc[(df_days["after_off_hours"] > hours_limit),"anticipation_type"]= 3 # 17/2/20 "long" before
            df_days.loc[(df_days["before_on_hours"] > hours_limit),"delay_type"]= 3         
            df_days.loc[((df_days["off_anticipation"] > off_anticipation_lmt) &
            (df_days["on_delay"].shift(-1) > on_delay_lmt) &
            (df_days["after_off_hours"] <= hours_limit)),"anticipation_type"]= 2 # 17/2/20 "overnight" before
            df_days.loc[((df_days["on_delay"] > on_delay_lmt) &
            (df_days["off_anticipation"].shift(1) > off_anticipation_lmt) &
            (df_days["before_on_hours"] <= hours_limit)),"delay_type"]= 2  #overnight disconnections end
            df_days.loc[(( df_days["off_anticipation"] > off_anticipation_lmt) &
            (df_days ["on_delay"].shift(-1) <= on_delay_lmt) &
            (df_days["after_off_hours"] <= hours_limit)),"anticipation_type"]= 1 # 17/2/20 "small before" #anticipation only end                                                   
            df_days.loc[(( df_days["on_delay"] > on_delay_lmt) &  #yesterday value for off_anticipation. 
            (df_days["off_anticipation"].shift(1) <= off_anticipation_lmt) &
            (df_days["before_on_hours"] <= hours_limit)),"delay_type"]= 1 #delayed only end          
            df_days.loc[(df_days["peak_delay"] > peak_delay_lmt_max),"peak_delay_outlier"]= 1 #peak delay outliers end
            df_days.loc[(df_days["peak_delay"] < peak_delay_lmt_min),"peak_delay_outlier"]= -1
            df_days.loc[(df_days["peak_difference"] > peak_difference_lmt_max),"peak_difference_outlier"]= 1 #peak delay outliers end
            df_days.loc[(df_days["peak_difference"] < peak_difference_lmt_min),"peak_difference_outlier"]= -1
            df_days.loc[(df_days["irradiation_difference"] > irradiation_difference_lmt_max),"irradiation_difference_outlier"]= 1 #peak delay outliers end
            df_days.loc[(df_days["irradiation_difference"] < irradiation_difference_lmt_min),"irradiation_difference_outlier"]= -1

            df_days_clear2 = df_days.loc[((df_days.anticipation_type.isna())&
                                         (df_days.delay_type.isna())&
                                         (df_days.peak_delay_outlier.isna())&
                                         (df_days.peak_difference_outlier.isna())&
                                         (df_days.irradiation_difference_outlier.isna())&
                                         (df_days.pearson>=0.8)),:]#no series count min imposed

            #df_days_clear = df_days_clear2.loc[(df_days.pearson>=0.8),:]#290520 tmp solution, two steps
            df_days_clear = df_days.loc[(df_days.pearson>=0.8),:]#290520 tmp solution, two steps
            
            """   #12/5/20 best month filter not required
            if df_d_clear.empty == True:
                df_days=df_days.assign(cloudy=np.full(len(df_days),1)) #17/2/20 True before
                df_days=df_days.assign(csmonthly=np.full(len(df_days),0)) #17/2/20 False before
                df_srs=df_srs.assign(cloudy=np.full(len(df_srs),1)) #17/2/20 True before
                df_srs=df_srs.assign(csmonthly=np.full(len(df_srs),0)) #17/2/20 False before
            elif (len(df_d_clear) > 0): #if not empty
                dates_cs = df_d_clear.loc[:,"date"].values
                df_days.loc[(df_days.loc[:,"date"].isin(dates_cs)==False),"cloudy"]=1 #"False" before flag cloudy in compare 
                df_srs.loc[(df_srs.loc[:,"date"].isin(dates_cs)==False),"cloudy"]=1
                grp_cs_m=df_days.groupby(pd.Grouper(freq='MS',key="data_off"))#looking for best monthly cs         
                df_series_max=grp_cs_m.apply(lambda x: x[x["series"]==x["series"].max()])
                #df_series_max=grp_cs_m.transform(lambda x: x[x["series"]==x["series"].max()])
                grp_cs_m=df_series_max.groupby(pd.Grouper(freq='MS',key="data_off"))  
                df_d_clearmonthly=grp_cs_m.apply(lambda x: x[x["kcs_std"]==x["kcs_std"].min()])
                #df_d_clearmonthly=grp_cs_m.transform(lambda x: x[x["kcs_std"]==x["kcs_std"].min()])#20/5/19: std instead of pearson
                dates_csmonthly = df_d_clearmonthly["date"]
                df_days.loc[(df_days.loc[:,"date"].isin(dates_csmonthly)==True),"csmonthly"]=1 #True before
                df_srs.loc[(df_srs.loc[:,"date"].isin(dates_csmonthly)==True),"csmonthly"]=1
            #print("return days overview, valid series in clear days and valid series")
            """
            
            df_days.index.name=None 
            df_days=df_days.merge(df_days_completeness.loc[:,["missing","na"]],how="right",right_index=True,left_index=True)
            
            
            df_days.loc[(df_days.loc[:,"date"].isin(df_days_clear.loc[:,"date"].values)==False),"cloudy_d"]=1            
            df_srs.index.name=None 
            df_days.index.name=None
            df_srs=df_srs.merge(right=df_days.loc[:,["date","on_delay","off_anticipation","peak_delay","peak_difference","irradiation_difference",
                                                     "pearson","anticipation_type","delay_type","peak_delay_outlier","peak_difference_outlier","irradiation_difference_outlier",
                                                     "cloudy_d"]]
                              ,left_on=["date"],right_on=["date"],how="left",sort=True)
            
            

            
            df_srs.loc[(df_srs.loc[:,"datetime"].isin(df_srs_clear.loc[:,"datetime"].values)==False),"cloudy_s"]=1   
            #df_srs2.loc[(df_srs2.loc[:,"datetime"].isin(df_srs2_clear.loc[:,"datetime"].values)==False),"cloudy_s"]=1  
            
            df_clear.index.name=None 
            df_clear=df_clear.loc[:,#tmp solution
                                  ["datetime","date","irradiance","irradiance_cs",
                                  "extraradiation","airmassrelative","aoi","azimuth","zenith",
                                  "kcs","irradiance_difference","i_cs","i2","cs2","datetime_s"]]
                                  #no kcs_s_len & kcs_s_median
           
            df_srs.rename(columns={"datetime":"datetime_s"},inplace=True)            
            df_clear = df_clear.merge(right=df_srs.loc[:,["date","on_delay","off_anticipation","peak_delay","peak_difference","irradiation_difference",
                                                     "pearson","anticipation_type","delay_type",
                                                     "kcs_s_median","kcs_s_std","kcs_s_len","datetime_s","irr_dif_s_median","irr_dif_s_std",
                                                     "cloudy_d","cloudy_s"]],left_on=["datetime_s","date"],right_on=["datetime_s","date"],how="left")    
            
            df_clear.loc[:,"datetime"]=df_clear.loc[:,"datetime"].apply(lambda x:x+np.timedelta64(int(srs_prd/2),'s'))#datetime back to solve merging with df_vld
            
            df_vld_lr=["datetime","date","irradiance","irradiance_cs","extraradiation",
                       "airmassrelative","aoi","azimuth","zenith","kcs","irradiance_difference",
                       "i_cs","i2","cs2"]
            
            df_vld = df_vld.merge(right=df_clear,
                                  left_on=df_vld_lr, #df_vld.columns.tolist(),
                                  right_on=df_vld_lr, #df_vld.columns.tolist(), uncertainty missing
                                  how="left")
            
            df_vld.loc[(df_vld.loc[:,"datetime"].isin(df_clear.loc[:,"datetime"].values)==False),"cloudy_m"]=1 
            df_vld.loc[(df_vld.loc[:,"date"].isin(df_days_clear.loc[:,"date"].values)==False),"cloudy_d"]=1            
            df_vld.loc[(df_vld.loc[:,"datetime_s"].isin(df_srs_clear.loc[:,"datetime"].values)==False),"cloudy_s"]=1   
            
        
        if len(df_vld)==0:   
          df_vld=pd.DataFrame()
          df_srs=pd.DataFrame()
          df_days=df_days_completeness

        return df_vld, df_srs, df_days 
    


    def irradianceflagging(SolarLibrary:pvlibint.SolarLibrary,
                          datetimeutc_values:np.ndarray, #timezone:str,
                          irradiance_values:np.ndarray, #no index, len should be same as datetimeindex,
                          timeresolution:int, #seconds
                          hours_limit=HOURS_LIMIT,
                          coverage_factor=COVERAGE_FACTOR,
                          series_count=SERIES_COUNT_MIN,
                          series_values=SERIES_VALUES_MIN,
                          irr_min=IRR_MIN,
                          aoi_max=AOI_MAX,
                          kcs_dev_lmt=KCS_DEV_LMT
                          ):
            #DEV NOTE 25/11/19 to be split into more indipendent parts while keeping effective comprenhesive solution
            #DEV NOTE 25/11/19  replace average with spot value, add as option            
            #DEV NOTE 25/11/19 daysaving passage to be added 
            #print("calculating srs_period and remove initial and last values to center average later")
            srs_prd = (int(timeresolution*series_values/60)+((timeresolution*series_values)%60>0))*60 #seconds
            #datetimeutc_values=datetimeutc_values[int(srs_prd/2):(len(datetimeutc_values)-1-int(srs_prd/2)):1] #27/11/19 beware removed some values doing intersect after
            #irradiance_values=irradiance_values[int(srs_prd/2):(len(irradiance_values)-1-int(srs_prd/2)):1]
            srs_cnt = series_values  #as reference minimum 15 of 5m (unstable sky conditions)              
            surface_zenith = SolarLibrary.surface_zenith
            #surface_azimuth = SolarLibrary.surface_azimuth #DEV NOTE 15/4/19 not needed for now (always south)                        
            datetimeindex_utc = pd.DatetimeIndex(datetimeutc_values,ambiguous='NaT') #tz="utc"            
            print("calcuating solar irradiance parameters")
            parameters = list(["clearskypoa","aoi"]) #define requested parameters
            df_par =  SolarLibrary.getsolarseries(datetimeindex_utc,parameters) #modelling parameters   
            df  = pd.DataFrame({   
            'datetime': datetimeindex_utc,
            'date':  datetimeindex_utc.date,
            'irradiance': irradiance_values, 
            'irradiance_cs': df_par.loc[:,"poa_global"],
            'extraradiation': df_par.loc[:,"extraradiation"],
            'airmassrelative': df_par.loc[:,"airmassrelative"],
            'aoi':df_par["aoi"] #, datetimeindex_utc.minute//15) #used for groupby
            },index= datetimeindex_utc)  
            irradiance_series = df.loc[:,"irradiance"]
            irradiance_series = irradiance_series.fillna(0) #replacing negative and NaN with 0 for calculation purpose
            irradiance_series = irradiance_series.where(irradiance_series>0,0) 
            df.loc[:,"irradiance"] = irradiance_series  
            print("calculating kcs & values for Pearson estimation from positive irradiance values")
            #df["kcs"]=df.apply(lambda x:x["irradiance"]/x["irradiance_cs"] if x["irradiance_cs"]>0 else 0,axis=1)#DEV NOTE 2/5/19 tmp solution                 
            df["kcs"]=df.apply(lambda x:x["irradiance"]/x["irradiance_cs"] if x["irradiance_cs"]>0 else 0,axis=1)#DEV NOTE 2/5/19 tmp solution                 
            df["irradiance_difference"] = (df["irradiance"]-df["irradiance_cs"])
            df["i_cs"] = df["irradiance"] * df["irradiance_cs"]
            df["i2"] = df["irradiance"] * df["irradiance"]
            df["cs2"] = df["irradiance_cs"] * df["irradiance_cs"]          
            
            
            print("group by day for daily stability") #only to get pearson for all values
            grp_day = df.groupby([pd.Grouper(freq="1D", key='datetime')])#identifying peak before cs series to distinguish morning/afternoon
            i2 = grp_day["i2"]
            cs = grp_day["irradiance_cs"]
            cs2 = grp_day["cs2"]
            i_cs = grp_day["i_cs"]  
            irradiance = grp_day['irradiance']  
            rows = irradiance.count()
            pearson_day = ((rows*i_cs.sum()-irradiance.sum()*cs.sum())/rows/
            np.sqrt((i2.sum()-irradiance.sum()**2/rows)*
            (cs2.sum()-cs.sum()**2/rows))) 
            date = grp_day['date'] 
            
            
            df_p = df.loc[((df.irradiance>irr_min)&(df.aoi<aoi_max)),["datetime","irradiance","date"]]      
            grp_p_day = df_p.groupby('date',as_index=True)
            datetime_p = grp_p_day["datetime"]
            irradiance_p = grp_p_day['irradiance']   
            df_peak_d = grp_p_day.apply(lambda x: x[x["irradiance"]==x["irradiance"].max()]) 
            grp_peak_d = df_peak_d.groupby([pd.Grouper(freq="1D",key="datetime")])
            datetime_peak = grp_peak_d['datetime'] 
            print("filter by defined acceptable deviation of Kcs from 1")
            df_vld = df.loc[((df.kcs>1-KCS_DEV_LMT)&(df.kcs<1+KCS_DEV_LMT)),["irradiance","datetime","kcs","aoi","date"]] #series grouping             
            print("group by series of defined period")
            
            df_vld.loc[:,"datetime"]=df_vld.loc[:,"datetime"].apply(lambda x:x-np.timedelta64(int(srs_prd/2),'s')) #shift to center in one minute before groupby
            
            
            grp_srs = df_vld.groupby([pd.Grouper(freq=str(srs_prd)+"s", key='datetime',label='right')])#label left
            agg_srs = grp_srs.agg({
                    'kcs': [np.std,np.median,len],
                    'aoi': [min],
                    'irradiance':[np.median],
                    'date': [max]})
            df_s = pd.DataFrame({
                   "kcs_s_median": agg_srs.loc[:,("kcs","median")].values,
                   "kcs_s_std":agg_srs.loc[:,("kcs","std")].values,
                   "kcs_s_len":agg_srs.loc[:,("kcs","len")].values, 
                   "aoi":agg_srs.loc[:,("aoi","min")].values,
                   "date": agg_srs.loc[:,("date","max")].values,
                   "irradiance":agg_srs.loc[:,("irradiance","median")].values,
                   "datetime":agg_srs.index})
            df_s.index=df_s.datetime
            
            path_to_output=r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/filtering/"
            df_s.to_csv(path_to_output+"irradiance_grouped"+'.csv') #flagging
            
            #df_s.loc[:,"irradiance"]=df.loc[(df.loc[:,"datetime"].isin(df_s.datetime)),"irradiance"] #27/11/19 spot values    
            print("identify clear series based on kcs std, valid values count and maximum AOI")
            df_s_clear = df_s.loc[((df_s.kcs_s_std>-KCS_CV_LMT)&(df_s.kcs_s_std<KCS_CV_LMT)& #Check conditions for each series
                                 (df_s.kcs_s_std!=0)&
                                 (df_s.kcs_s_len>=SERIES_VALUES_MIN)&
                                 (df_s.aoi<aoi_max)), #calibration condition 
                                ["datetime","kcs_s_std","kcs_s_median","kcs_s_len","date","irradiance"]]#27/11/19    
            grp_s_day = df_s_clear.groupby([pd.Grouper(freq="1D", key='datetime')])    
            kcs_s_valid_std = grp_s_day["kcs_s_std"]
            kcs_s_valid_median = grp_s_day["kcs_s_median"] 
            #'peak_datetime_max':df_day_peak.max().values, 
            #DEV NOTE 15/4/19 drop index since ambiguous
            #df_peak = df_peak.reset_index(drop=True)
            #df_day_peak=df_peak.groupby(['date'], as_index=True)['datetime'] 
            print("Calculate cs-related parameters: peaks, daily deviation, data on/off, Pearson & series")
            df_days = pd.DataFrame({'data_on': datetime_p.min(), #Group per day, accounting median parameters of series
            'data_off':datetime_p.max(),
            'peak_datetime_min':datetime_peak.min(),
            'peak_datetime_max':datetime_peak.max(),       
            'peak_irr':irradiance_p.max(),           
            'date':date.min(),
            'pearson':pearson_day, #used for clear sky 
            'kcs_std':kcs_s_valid_std.median(),
            'kcs_median':kcs_s_valid_median.median(),
            'series': kcs_s_valid_median.count(),      
            'irradiance_difference_d':(irradiance.sum()-cs.sum())/cs.sum() #used for clear sky only
            },
            index=date.min())    
            #df_days.to_csv( _PATH_TO_OUTPUT+"test_days.csv")
            #DEV NOTE 3/10/18: outliers anakysis seems to cover only internal outliers not outside 
            print("estimate irradiance peak and datetimes for sunrise, sunset and peak")
            if surface_zenith==0: 
                sunpos = SolarLibrary.getsunrisesettransit(df_days['date']) #Getting sunrise & sunset from SolarModels class  
                df_days=df_days.assign(sun_on = pd.DatetimeIndex(sunpos['sunrise'],tz=None))
                df_days=df_days.assign(sun_out =  pd.DatetimeIndex(sunpos['sunset'],tz=None))   
                df_days=df_days.assign(sun_peak =  pd.DatetimeIndex(sunpos['suntransit'],tz=None))                
            if surface_zenith!=0:
                sunpos = SolarLibrary.getsuninout(df_days.loc[:,'date'],freq="min")                
                df_days=df_days.assign(sun_on = pd.DatetimeIndex(sunpos['sun_on'],tz=None))
                df_days=df_days.assign(sun_out =  pd.DatetimeIndex(sunpos['sun_off'],tz=None))
                df_days=df_days.assign(sun_peak =  pd.DatetimeIndex(sunpos['sun_max'],tz=None))           
            parameters = list(["clearskypoa"])
            poa_global=  SolarLibrary.getsolarseries(df_days['sun_peak'],outputs=parameters)
            df_days=df_days.assign(peak_irr_mod=poa_global.loc[:,"poa_global"].values) 
            print("calculate daily cs-related indicators of delay/anticipation, disconnections & peaks")
            df_days["on_delay"]=df_days.apply(lambda x: tdhour(x["data_on"]-x["sun_on"]),axis=1)
            df_days["off_anticipation"]=df_days.apply(lambda x: tdhour(x["sun_out"]-x["data_off"]),axis=1)
            df_days["peak_delay"]=df_days.apply(lambda x: 0.5*tdhour(x["peak_datetime_max"]-x["sun_peak"])+
            0.5*tdhour(x["peak_datetime_min"]-x["sun_peak"]),axis=1)
            df_days["peak_irr_dif"]=df_days.apply(lambda x: x["peak_irr"]-x["peak_irr_mod"],axis=1)
            df_days["after_off_hours"] = df_days["data_on"].shift(-1)- df_days["data_off"] #28/3/19 DEV no shift immediately
            df_days["before_on_hours"] = df_days["data_on"] - df_days["data_off"].shift(1)            
            #df_days["after_off_hours"]=df_days["after_off_hours"].apply(lambda x: tdhour(x))
            df_days["after_off_hours"]=df_days["after_off_hours"].transform(lambda x: tdhour(x))
            #df_days["before_on_hours"]=df_days["before_on_hours"].apply(lambda x: tdhour(x)) #End of parameters calculation
            df_days["before_on_hours"]=df_days["before_on_hours"].transform(lambda x: tdhour(x))
            #LIMITS FOR DISCONNECTIONS:distance from expected values is higher than standard deviation *3 (Ralph also requested)   
            on_delay_limit = (df_days.loc[df_days.on_delay>0,"on_delay"].median()+
                    df_days.loc[df_days.on_delay>0,"on_delay"].std()*coverage_factor)  #identification of suspect outliers
            off_anticipation_limit = (df_days.loc[df_days.off_anticipation>0,"off_anticipation"].median()+
                    df_days.loc[df_days.off_anticipation>0,"off_anticipation"].std()*coverage_factor)
            peak_delay_limit = (df_days.loc[:,"peak_delay"].median()+ df_days.loc[:,"peak_delay"].std()*coverage_factor)
            peak_irr_dif_limit = (df_days.loc[df_days.peak_irr_dif>0,"peak_irr_dif"].median()+
                    df_days.loc[df_days.peak_irr_dif>0,"peak_irr_dif"].std()*coverage_factor)            
            #DEV NOTE 4/4/19: median could be a systematic effect     
            if hours_limit is None: #otherwise if too small all classified as big
                onoffdiff = df_days["sun_out"] - df_days["sun_on"]
                #onoffdiff = onoffdiff.apply(lambda x: tdhour(x))
                onoffdiff = onoffdiff.transform(lambda x: tdhour(x))
                hours_limit = 24 - onoffdiff.min()
            print("flagging for daily cs-related indicators")
            df_days.loc[(df_days["after_off_hours"] > hours_limit),"anticipation_type"]= 3 # 17/2/20 "long" before
            df_days.loc[(df_days["before_on_hours"] > hours_limit),"delay_type"]= 3         
            df_days.loc[((df_days["off_anticipation"] > off_anticipation_limit) &
            (df_days["on_delay"].shift(-1) > on_delay_limit) &
            (df_days["after_off_hours"] <= hours_limit)),"anticipation_type"]= 2 # 17/2/20 "overnight" before
            df_days.loc[((df_days["on_delay"] > on_delay_limit) &
            (df_days["off_anticipation"].shift(1) > off_anticipation_limit) &
            (df_days["before_on_hours"] <= hours_limit)),"delay_type"]= 2  #overnight disconnections end
            df_days.loc[(( df_days["off_anticipation"] > off_anticipation_limit) &
            (df_days ["on_delay"].shift(-1) <= on_delay_limit) &
            (df_days["after_off_hours"] <= hours_limit)),"anticipation_type"]= 1 # 17/2/20 "small before" #anticipation only end                                                   
            df_days.loc[(( df_days["on_delay"] > on_delay_limit) &  #yesterday value for off_anticipation. 
            (df_days["off_anticipation"].shift(1) <= off_anticipation_limit) &
            (df_days["before_on_hours"] <= hours_limit)),"delay_type"]= 1 #delayed only end          
            df_days.loc[(df_days["peak_delay"] > peak_delay_limit),"peak_delay_outlier"]= 1 #peak delay outliers end
            df_days.loc[(df_days["peak_irr_dif"] > peak_irr_dif_limit),"irradiance_outlier"]= 1 #peak irradiance outliers end            
            
         
            df_d_clear = df_days[(df_days.pearson>=PEARSON_LMT)&                               
                               (df_days.series>=srs_cnt)&
                               (df_days.anticipation_type.isnull())&
                               (df_days.delay_type.isnull())&
                               (df_days.peak_delay_outlier.isnull())] #30/5/20: to be checked before was not working
            
            
            if df_d_clear.empty == True:
                df_days=df_days.assign(cloudy=np.full(len(df_days),1)) #17/2/20 True before
                df_days=df_days.assign(csmonthly=np.full(len(df_days),0)) #17/2/20 False before
                df_sd_clear = pd.DataFrame()
            elif (len(df_d_clear) > 0): #if not empty
                dates_cs = df_d_clear.loc[:,"date"].values
                df_days.loc[(df_days.loc[:,"date"].isin(dates_cs)==False),"cloudy"]=1 #"False" before flag cloudy in compare 
                grp_cs_m=df_days.groupby(pd.Grouper(freq='MS',key="data_off"))#looking for best monthly cs         
                df_series_max=grp_cs_m.apply(lambda x: x[x["series"]==x["series"].max()])
                grp_cs_m=df_series_max.groupby(pd.Grouper(freq='MS',key="data_off"))  
                df_d_clearmonthly=grp_cs_m.apply(lambda x: x[x["kcs_std"]==x["kcs_std"].min()]) #18/2/20 removed apply to be sure
                #df_d_clearmonthly=grp_cs_m.transform(lambda x: x[x["kcs_std"]==x["kcs_std"].min()])#20/5/19: std instead of pearson
                dates_csmonthly = df_d_clearmonthly["date"]
                df_days.loc[(df_days.loc[:,"date"].isin(dates_csmonthly)==True),"csmonthly"]=1 #True before
                df_sd_clear=df_s_clear.loc[(df_s_clear.loc[:,"date"].isin(dates_cs))] 
            print("return days overview, valid series in clear days and valid series")
            return df_days, df_sd_clear, df_s_clear #
            #grp_cs_m=df_compare.groupby(pd.Grouper(freq='MS',key="data_off"))            
            #df_compare2=grp_cs_m.apply(lambda x: x[x["series"]==x["series"].max()])
            #grp_cs_m3=df_compare2.groupby(pd.Grouper(freq='MS',key="data_off"))  
            #df_compare3=grp_cs_m3.apply(lambda x: x[x["pearson"]==x["pearson"].max()])
            
       
    
    def irradiance_cosine_error(
            location:pvlibint.SolarLibrary,
            global_series:pd.Series,
            beam_series:pd.Series,
            diffuse_series:pd.Series,
            time_series:pd.Series,
            aoi_series=None):
            df=pd.DataFrame({
            "global": global_series,
            "beam": beam_series,
            "diffuse": diffuse_series})
            #function for defining and plotting cosine error only
            #DEV NOTE 6/11/18: by not plotting immediately the graph, sovrapposition of graph good for iteration    
            time_series=pd.DatetimeIndex(time_series)
            df.loc[:,"datetime"]=time_series
            df.loc[:,"hour"]=time_series.hour 
            df.loc[:,"month"]=time_series.month
            #DEV NOTE 6/11/18: changed from the original one, only one AOI accepted & used
            if isinstance(aoi_series,pd.Series) == True: 
                df.loc[:,"aoi"] = aoi_series
            elif isinstance(aoi_series,pd.Series) == False:
                #calculated aoi from NREL 
                df_par=location.getsolarseries(pd.DatetimeIndex(time_series),outputs=["zenith"])
                df = df.merge(df_par[["zenith","datetime"]],left_on="datetime",right_on="datetime",how='left')
                df = df.rename(columns={"zenith":"aoi"})
                #df.loc[:,"aoi"] =  location.getsolarzenithangle(time_series)   
            
            
            #filter for AOI not higher than 90
            df = df[df["aoi"]<90]
            #calculation for the estimated aoi
            #estimate direct irradiance component for the measured global irradiance from the beam one 
            df["beam in plane"] = np.cos(np.radians(df.loc[:,"aoi"]))*beam_series
            #calculate measured direct irradiance component for the global sensor only
            df["global minus diffuse"] =  df["global"] - df["diffuse"]
            #calculate absolute beam deviation
            #DEV NOTE 2/3/19: absolute beam deviation renamed as abs_err       
            df["abs_err"] = df["global minus diffuse"] - df["beam in plane"] 
            #calculate percentage of beam deviation
            df["cos_err"] = df["abs_err"]/df["beam in plane"]*100  
            #plot deviation
            
            #DEV 23/2/19: graphs sould be created jointly intestead            
            """
            #DEV NOTE 24/2/19: previous before splitting calculation & representation
            f = plt.figure(3)                       
            f.add_subplot(111)
            plt.xlabel("angle of incidence")
            plt.ylabel("percentage beam deviation [%]")
            if label_suffix != None:
                #if suffix provided add to suffix (for comparison of results)
                label1 = "percentage beam deviation before 12:00 "+label_suffix
                label2 = "percentage beam deviation after 12:00 "+label_suffix
            CFormat=chrt.Format()  
            p1,=chrt.lineplot(CFormat,df[df.hour<12]["angle of incidence"],
            df[df.hour<12]["percentage beam deviation"],label1,marker="^",ylim=(-30,70))
            p2,=chrt.lineplot(CFormat,df[df.hour>12]["angle of incidence"],
            df[df.hour>12]["percentage beam deviation"],label2,marker="v",ylim=(-30,70))           
            chrt.save(CFormat,filename="test")
            """            
            return df 
        

    def irradiance_horizontal_comparison(location:pvlibint.SolarLibrary,
                                         global_series,
                                         beam_series,
                                         diffuse_series,
                                         time_series,
                                         aoi_series=[],
                                         graph_vs_time=True,
                                         graph_deviation_abs=False,
                                         graph_deviation_prc=False,                                    
                                         graph_x_frequency="h",
                                         markersize=2
                                         ):
        df=pd.DataFrame({
        "global": global_series,
        "beam": beam_series,
        "diffuse": diffuse_series})
        time_series=pd.DatetimeIndex(time_series)
        df["time"]=time_series
        df["hour"]=time_series.hour 
        if aoi_series != []:
            df["angle of incidence provided"] = aoi_series
            #calculation for the estimated angle of incidence
            #estimate direct irradiance component for the measured global irradiance from the beam one 
            df["beam in plane (AoI inp)"] = np.cos(np.radians(df.loc[:,"angle of incidence calculated"]))*beam_series
            #calculate measured direct irradiance component for the global sensor only
            df["global minus diffuse (AoI inp)"] =  df["global"] - df["diffuse"]
            #calculate absolute beam deviation
            df["absolute beam deviation (AoI inp)"] = df["global minus diffuse (AoI inp)"] - df["beam in plane (AoI inp)"] 
            #calculate percentage of beam deviation
            df["percentage beam deviation (AoI inp)"] = df["absolute beam deviation (AoI inp)"]/df["beam in plane (AoI inp)"]*100  
            #filter for AOI not higher than 90
            df = df[df["angle of incidence provided"]<90]
        #calculated angle of incidence from NREL  
        df["angle of incidence NREL"] =  location.getsolarzenithangle(time_series) 
        #calculation for the measured angle of incidence
        #estimate direct irradiance component for the measured global irradiance from the beam one 
        df["beam in plane (AoI NREL)"] = np.cos(np.radians(df.loc[:,"angle of incidence NREL"]))*beam_series
        #calculate measured direct irradiance component for the global sensor only
        df["global minus diffuse (AoI NREL)"] =  df["global"] - df["diffuse"]
        #calculate absolute beam deviation
        df["absolute beam deviation (AoI NREL)"] = df["global minus diffuse (AoI NREL)"] - df["beam in plane (AoI NREL)"] 
        #calculate percentage of beam deviation
        df["percentage beam deviation (AoI NREL)"] = df["absolute beam deviation (AoI NREL)"]/df["beam in plane (AoI NREL)"]*100                                  
        #filter for angle of incidence not higher than 90
        df = df[df["angle of incidence NREL"]<90]
        #plot deviations
        #plotting all irradiance measurements against time
        if graph_vs_time == True:            
            #plt.subplot(1,1,1) 
            df1 = df.loc[:,("time",
                            "beam",
                            "beam in plane",
                            "global minus diffuse",
                            "diffuse",
                            "global")]            
            fshw.plotvstime(dataframe=df1,
                            time_label="time",
                            frequency="h",
                            merged_y="irradiance [Wm-2]",
                            title=None,
                            legend_anchor=-0.4,
                            markersize=markersize,
                            path_to_output=None)     
       #plotting absolute deviation of beam irradiance 
        if graph_deviation_abs == True:                          
            f2 = plt.figure(2)
            #subplot to setup and plot on something instead of nothing ?
            ax2 = f2.add_subplot(111)
            plt.xlabel("angle of incidence")
            plt.ylabel("absolute beam deviation [Wm-2]")
            ax2.scatter(df["angle of incidence"],df["absolute beam deviation"])
        #plotting percentage deviation of beam irradiance 
        if graph_deviation_prc == True:    
            f3 = plt.figure(3)                       
            f3.add_subplot(111)
            plt.xlabel("angle of incidence")
            plt.ylabel("percentage beam deviation [%]")   
            #p5,= plt.plot(df["angle of incidence provided"],df["percentage beam deviation (AoI inp)"],".",markersize=markersize)  
            #p7,= plt.plot(df["angle of incidence calculated"],df["percentage beam deviation (AoI NREL)"],".",markersize=markersize)                          
            p5,= plt.plot(df[df.hour<12]["angle of incidence NREL"],
            df[df.hour<12]["percentage beam deviation (AoI NREL)"],"^",
            label="percentage beam deviation before 12:00 (AoI NREL)",markersize=markersize)  
            p6,= plt.plot(df[df.hour>12]["angle of incidence NREL"],
            df[df.hour>12]["percentage beam deviation (AoI NREL)"],"v",
            label="percentage beam deviation after 12:00 (AoI NREL)",markersize=markersize)
            if aoi_series != []:
                p7,= plt.plot(df[df.hour<12]["angle of incidence provided"],
                df[df.hour<12]["percentage beam deviation (AoI inp)"],"^",
                label="percentage beam deviation before 12:00 (AoI NREL)",markersize=markersize)  
                p8,= plt.plot(df[df.hour>12]["angle of incidence provided"],
                df[df.hour>12]["percentage beam deviation (AoI inp)"],"v",
                label="percentage beam deviation after 12:00 (AoI NREL)",markersize=markersize)
            plt.show()                           
        return df 
    
    def characterisation_outdoor_alternating(location:pvlibint.SolarLibrary,
                                 datetimeindex_utc:pd.DatetimeIndex, #timezone:str,
                                 hemispherical_values:np.ndarray,
                                 temperature_values:np.ndarray, #temperature of sensor for irradiance correction also for pyrheliometer?
                                 beam_values:np.ndarray,
                                 phase_values:np.ndarray,
                                 resolutions:np.ndarray,
                                 periods:np.ndarray,
                                 counts:np.ndarray,
                                 deviation_max=DEVIATION_MAX,
                                 tilt=float,
                                 irr_min=IRR_MIN,
                                 aoi_max=AOI_MAX,
                                 kcs_range=[1-KCS_DEV_LMT,1+KCS_DEV_LMT],
                                 kcs_cv_max=KCS_CV_LMT, #used also for the power
                                 pwr_cv_max=PWR_CV_LMT,
                                 pearson_min=PEARSON_LMT,
                                 count_characterisation=SERIES_VALUES_MIN,
                                 keep_invalid=False): #generic parameters for distinguishing phases to be introduced
        
        #stability flagging should be performed at resolution level and only for measurements of phases 
        
        # 6/9/25 stability at 
        df,dfsout=SolarData.stabilityflagging(SolarLibrary=location,
                          datetimeindex_utc=datetimeindex_utc, #timezone:str,
                          irradiance_values=beam_values,                      
                          resolutions=resolutions,
                          periods=periods,
                          counts=counts,
                          irr_min=irr_min,
                          aoi_max=aoi_max,
                          kcs_range=kcs_range,
                          kcs_cv_max=kcs_cv_max, 
                          pwr_cv_max=pwr_cv_max,
                          pearson_min=pearson_min,
                          power_values=None,
                          beam=True,
                          keep_invalid=keep_invalid
                          )  
        df=df.rename(columns={"aoi":"zenith"})
        
        
        timedelta_su=60*2 #intervals between the end of two measurements phases
        df.loc[:,"hemispherical"]=hemispherical_values
        df.loc[:,"phase"]=phase_values
        df.loc[:,"beam"]=beam_values
        
        df_srs=df
        #df_srs=df.loc[(df.i_unstable==0)] #stability for analysed resolutions (no Pearson low level)
        #test dt_stable=df.loc[(df.i_stable==True),"datetime"]
        #test df_srs=df #13/1/20 test remove stability requirement
        
        calibration_series_summary = pd.DataFrame()
        if len(df_srs)!=0: 
            sqn_ends=df_srs.loc[(df_srs.loc[:,"phase"]!=df_srs.loc[:,"phase"].shift(-1)),["datetime","phase"]]
            sqn_ends.index=sqn_ends.loc[:,"datetime"]
            sqn_ends.loc[:,"sequence"]=range(0,len(sqn_ends))
            def sequence_id(x):
                i=sqn_ends.index.get_loc(x,method="bfill")
                sqn=sqn_ends.iloc[i,sqn_ends.columns.get_loc("sequence")]
                return sqn
            df_srs.loc[:,"sequence"]=df_srs.loc[:,"datetime"].transform(sequence_id)
            # 6/9/25 defining diffuse dataframe including only shaded series
            df_dff=df_srs.loc[(df_srs.loc[:,"phase"]==2),:]
            #DUMMY TEST CODE
            #df_dff.loc[:,"hemispherical"]=np.random.randint(low=20, high=200, size=len(df_dff)) #test dummy
            grp_dff = df_dff.groupby("sequence")
            agg_dff_tmp = grp_dff.agg({
                    'datetime':[max,len],
                    'hemispherical':[np.median,np.var],
                    'beam':[np.median,np.var]
                    })            
            agg_dff=pd.DataFrame({"sequence":agg_dff_tmp.index,
            "hemispherical":agg_dff_tmp.loc[:,("hemispherical","median")].values,
            "hemispherical_var":agg_dff_tmp.loc[:,("hemispherical","var")].values,
            "datetime":agg_dff_tmp.loc[:,("datetime","max")].values,
            "count":agg_dff_tmp.loc[:,("datetime","len")].values,
            },index=agg_dff_tmp.loc[:,("datetime","max")].values)
            
            #12/1/20 improvements: filtering without removing 
            #dt_dff = agg_dff.loc[(agg_dff.loc[:,"hemispherical_var"]<deviation_max)&(agg_dff.loc[:,"hemispherical_var"]>-deviation_max)&(agg_dff.loc[:,"count"]>=count_characterisation),"datetime"] #condition on shading                  
            dt_dff = agg_dff.loc[:,"datetime"] #condition on shading                  
            # 6/9/25 defining hemispherical dataframe including only unshaded series            
            df_hms=df_srs.loc[(df_srs.loc[:,"phase"]==5),:]  
            #DUMMY TEST CODE
            #df_hms.loc[:,"hemispherical"]=np.random.randint(low=100, high=500, size=len(df_hms)) #test dummy
            #df_hms.loc[:,"aoi"]=np.random.randint(low=60, high=70, size=len(df_hms)) #test dummy
            #df_hms.loc[:,"beam"]=np.random.randint(low=500, high=700, size=len(df_hms))     
            grp_hms = df_hms.groupby("sequence")
            agg_hms_tmp = grp_hms.agg({
                    'datetime':[max,len],
                    'hemispherical':[np.median,np.var],
                    'beam':[np.median,np.var],
                    "zenith":[np.median], #not required for resolution <= 60s
                    "i_unstable":[sum]
                    })            
            agg_hms=pd.DataFrame({"sequence":agg_hms_tmp.index,
            "hemispherical":agg_hms_tmp.loc[:,("hemispherical","median")].values,
            "hemispherical_var":agg_hms_tmp.loc[:,("hemispherical","var")].values,
            "beam":agg_hms_tmp.loc[:,("beam","median")].values,
            "beam_var":agg_hms_tmp.loc[:,("beam","var")].values,
            "datetime":agg_hms_tmp.loc[:,("datetime","max")].values,
            "zenith":agg_hms_tmp.loc[:,("zenith","median")],
            "count":agg_hms_tmp.loc[:,("datetime","len")].values,
            "i_unstable_cnt":agg_hms_tmp.loc[:,("i_unstable","sum")].values
            },index=agg_hms_tmp.index) #use sequence as identifier (datetime for shading)
            agg_hms.loc[:,"dt_dff_bfr"]=agg_hms.loc[:,"datetime"].transform(lambda x:x-dt.timedelta(seconds=timedelta_su))
            agg_hms.loc[:,"dt_dff_aft"]=agg_hms.loc[:,"datetime"].transform(lambda x:x+dt.timedelta(seconds=timedelta_su))
            agg_hms_vld=agg_hms.loc[((agg_hms.loc[:,"dt_dff_bfr"].isin(dt_dff))&(agg_hms.loc[:,"dt_dff_aft"].isin(dt_dff))),:]
            agg_hms_vld.loc[:,"aoi_tilt"]=agg_hms_vld.loc[:,"zenith"]-tilt
            if len(agg_hms_vld)!=0:
                agg_hms_vld.loc[:,"diffuse_bfr"]=agg_hms_vld.apply(lambda x: agg_dff.loc[x.loc["dt_dff_bfr"],"hemispherical"],axis=1)
                agg_hms_vld.loc[:,"diffuse_aft"]=agg_hms_vld.apply(lambda x: agg_dff.loc[x.loc["dt_dff_aft"],"hemispherical"],axis=1)    
                agg_hms_vld.loc[:,"diffuse"]=agg_hms_vld.apply(lambda x:np.median([x["diffuse_bfr"],x["diffuse_aft"]]),axis=1)
                #12/01/20 diffuse reated parameters for further analysis
                agg_hms_vld.loc[:,"dff_aft_var"]=agg_hms_vld.apply(lambda x:agg_dff.loc[x.loc["dt_dff_aft"],"hemispherical_var"],axis=1)   
                agg_hms_vld.loc[:,"dff_bfr_var"]=agg_hms_vld.apply(lambda x:agg_dff.loc[x.loc["dt_dff_bfr"],"hemispherical_var"],axis=1)   
                agg_hms_vld.loc[:,"dff_aft_len"]=agg_hms_vld.apply(lambda x:agg_dff.loc[x.loc["dt_dff_aft"],"count"],axis=1)   
                agg_hms_vld.loc[:,"dff_bfr_len"]=agg_hms_vld.apply(lambda x:agg_dff.loc[x.loc["dt_dff_bfr"],"count"],axis=1)
              #13/01/20 required calculated parameters 
                df_hms_vld=df_hms.loc[(df_hms.loc[:,"sequence"].isin(agg_hms_vld.index)),:]#screening out valid of not valid series
                df_hms_vld.loc[:,"diffuse"]=df_hms_vld.apply(lambda x:agg_hms_vld.loc[x.loc["sequence"],"diffuse"],axis=1) #9/1/2020 provided as check
                df_hms_vld.loc[:,"aoi_tilt"]=df_hms_vld.loc[:,"zenith"]-tilt
                # 6/9/25 calculation of the measured beam, the theoretical one is from the pyrheliometer
                df_hms_vld.loc[:,"hemispherical_beam"]=df_hms_vld.apply(lambda x:
                (x["hemispherical"]-x["diffuse"])/np.cos(np.deg2rad(x["aoi_tilt"])),axis=1) #aoi is for horizontal initially 
                calibration = SolarData.Calibration(location=location, method="9847 1a",calibration_factor=1)#initialise characterisation (as calibration with reference factor 1)    
                print("Characterisation of series with max deviation of "+str(deviation_max)+" & min "+str(count_characterisation)+" valid points")
                for i in agg_hms_vld.index:
                    df_hms_tmp=df_hms_vld.loc[(df_hms_vld.loc[:,"sequence"]==i)]
                    
                    valid,f_j,f_j_std,len_j,results_j_df=calibration.get_factor_df(
                            reference_series=df_hms_tmp.loc[:,"beam"],
                            field_series=df_hms_tmp.loc[:,"hemispherical_beam"],
                            time_series=df_hms_tmp.loc[:,"datetime"],
                            deviation_max=deviation_max,
                            readings_min=count_characterisation,
                            iterations_max=ITERATIONS_MAXIMUM,
                            log=False
                            )
                    #aggregated parameters 
                    datetime_max=agg_hms_vld.loc[i,"dt_dff_bfr"]
                    aoi_tilt=agg_hms_vld.loc[i,"aoi_tilt"]
                    zenith=agg_hms_vld.loc[i,"zenith"]
                    dff_aft_var=agg_hms_vld.loc[i,"dff_aft_var"]
                    dff_bfr_var=agg_hms_vld.loc[i,"dff_bfr_var"]
                    dff_aft_len=agg_hms_vld.loc[i,"dff_aft_len"]
                    dff_bfr_len=agg_hms_vld.loc[i,"dff_bfr_len"]
                    dff_aft=agg_hms_vld.loc[i,"diffuse_aft"]
                    dff_bfr=agg_hms_vld.loc[i,"diffuse_bfr"]
                    beam=agg_hms_vld.loc[i,"beam"]
                    beam_var=agg_hms_vld.loc[i,"beam_var"]
                    unstable_cnt=agg_hms_vld.loc[i,"i_unstable_cnt"]
    
                    results_j=pd.DataFrame( # parameters from COMPLETE series before calibration process !!
                        { "datetime":datetime_max,
                         "tilt":tilt, #depend on file
                         "aoi_tilt":aoi_tilt,
                         "zenith":zenith,
                         "diffuse_aft":dff_aft,
                         "diffuse_bfr":dff_bfr,
                         "dff_aft_var":dff_aft_var,
                         "dff_bfr_var":dff_bfr_var,
                         "dff_aft_len":dff_aft_len,
                         "dff_bfr_len":dff_bfr_len,
                         "beam":beam,
                         "beam_var":beam_var,
                         "i_unstable_cnt":unstable_cnt,
                         "c_invalid":valid},
                        index=[i]) 
                    if valid == 0:             
                        #create array with series id
                        #results_j_df.assign(id_s=np.full(len(results_j_df),df_hms_vld.loc[i,"id_s"]))
                        #concatenate series df with previous one
                        #DEV NOTE 8/1/19: sort=False removed but not clear why necessary
                        #calibration_series = pd.concat([calibration_series,results_j_df])
                        #define main results for series 
                        results_j.loc[i,"cosine error"]=1-f_j
                        results_j.loc[i,"abs_err"]=(1-f_j)*beam
                        results_j.loc[i,"cos_err_std"]=f_j_std
                        # 7/9/25 updated formulas. Better to redefine fir sensitivity instead of calibration factor ?
                        results_j.loc[i,"cosine error"]=1/f_j -1
                        # 7/9/25 this is actually deviation, not absolute error, but ket 
                        results_j.loc[i,"abs_err"]=(1/f_j -1)*beam
                        results_j.loc[i,"cos_err_std"]=f_j_std                        
                        results_j.loc[i,"len_j"]=len_j
                        
                            
                        #append results
                    elif valid == False: results_j.loc[i,"c_invalid"]=False
                    
                    calibration_series_summary=calibration_series_summary.append(results_j) 
       
        return calibration_series_summary,df,dfsout
    


      
            
    




           
     
           
"""
#FUTURE IMPROVEMENTS FOR STABILISATION: externalisation of graph format in the reporttemplate module?          
title=("Analysis of sensor daily first connections and last disconnections based on a "
+ str(coverage_factor) + "x Standard Deviation limits"+"\n" + "(from "+str(df["date"].min())
+" to "+str(df["date"].max())+")")
#wrap into a single line
import textwrap as twrp
title="\n".join(twrp.wrap(title,70))
plt.title(title)
plt.ylabel('hour')
plt.xlabel('date')
plt.xticks(rotation='vertical')
#consecutive plot 
p1,= plt.plot(df_compare['date'],df_compare['sunrise'],'yo')
p2,= plt.plot(df_compare['date'],df_compare['sunset'],'ko')
p3,= plt.plot(dfdel["date"],dfdel["data_on"],'y2')
p4,= plt.plot(dfant["date"],dfant["data_off"],'k1')
p5,= plt.plot(dfrcvsml["date"],dfrcvsml["data_on"],'b^')
p6,= plt.plot(dfdscnight["date"],dfdscnight["data_off"],'mv')
p7,= plt.plot(dfrcvbig["date"],dfrcvbig["data_on"],'gx')
p8,= plt.plot(dfdscbig["date"],dfdscbig["data_off"],'rx')                  
#DEV NOTE 31/8/18: to be clarified why subplot necessary
subplot = plt.subplot()
#legend                     
plt.legend([p1,p2,p3,p4,p5,p6,p7,p8],
["sunrise","sunset","delayed switch-on","premature switch-off",
"suspected disconnection end","suspected disconnection start",
"day(s)-long disconnection end","day(s)-long disconnection start"],
bbox_to_anchor=(0,-1,1,-1), loc=8, mode="expand", ncol=1,
borderaxespad=0.)
#set limit for axes 
plt.Axes.set_ylim(subplot,bottom=0,top=24)                
fn = 'DisconnectAnalysisStd'+ str(coverage_factor) 
#DEV NOTE: not necessary ?
#plt.show()
#saving of graph at the end, could be optional too. 
    
    
elif str(np.dtype(datetimeindex_utc)) != "datetime64[ns]":  
    print("Error: not possible to format the time series")
    df_compare = pd.DataFrame().empty                
"""    


