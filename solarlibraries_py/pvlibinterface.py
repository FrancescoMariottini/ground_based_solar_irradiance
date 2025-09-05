# -*- coding: utf-8 -*-
"""
@author: Francesco Mariottini 
Created on 14/8/17
more information on the readme file 
getparameter 

"""

# -*- coding: utf-8 -*-
"""
@author: Francesco Mariottini 
Created on 14/8/17
more information on the readme file 
#DEV NOTE 05/12/19: should pvlib & meteodataquality be harmonised?
#DEV NOTE 28/11/18: naming 'datetime' & 'datetime_series' to be harmonised
 
"""

"""SETTING MODULES FOLDER & IMPORTING MODULES"""
#importing sys to locate modules in different paths
import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)



import numpy as np
import pandas as pd

#import pvlib.irradiance as irr
#import pvlib_python.pvlib.solarposition as sp
#import pvlib_python.pvlib.atmosphere as atm
#import pvlib_python.pvlib.clearsky as csky

from pvlib import irradiance as irr 
from pvlib import solarposition as sp
from pvlib import atmosphere as atm
from pvlib import clearsky as csky

from pvlib import location as lct

import datetime as dtt
#used for mnanipulation of getsunrisesettransit2 version 2



"""GLOBAL VARIABLES (default values)"""
#position A
LATITUDE=51+27/60+14.4/3600
LONGITUDE=3+23/60+14.3/3600
ALTITUDE=79
TIMEZONE = "utc"

#surface
surface_zenith=0
SURFACE_AZIMUTH=180
#BEWARE:
# surface_azimuth must be >=0 and <=360. The azimuth convention is dAPPARENT_ZENITH_MODELefined as degrees east of north 
#(e.g. North = 0, South=180 East = 90, West = 270).
#Source: perez(surface_zenith, surface_azimuth, dhi, dni, dni_extra,solar_zenith, solar_azimuth, airmass,model='allsitescomposite1990', return_components=False)          
            

#atmosphere
PRESSURE=101325.
TRANSMITTANCE=0.5
#modeling
ALBEDO = 0.25
APPARENT_ZENITH_MODEL="kastenyoung1989"  #used for atm module 
diffuse_model="allsitescomposite1990"
poa_model="perez"
PACKAGE="pvlib"  #DEV NOTE 3/5/19 could be eliminated
POA_MODEL="isotropic"
#POA_MODEL="perez" DEV NOTE 3/19 run error


#setup
class SolarLibrary:
    _parameters = ["airmassabsolute","airmassrelative","aoi","aoiprojection","apparentzenith",
                   "azimuth","clearsky","clearskypoa","dayofyear","extraradiation","linketurbidity","zenith"]
    #deve note rename add 
    def __init__(self,latitude=LATITUDE,longitude=LONGITUDE,altitude=ALTITUDE,
        surface_zenith=surface_zenith,surface_azimuth=SURFACE_AZIMUTH,package=PACKAGE,
        poa_model=poa_model,transmittance=TRANSMITTANCE,pressure=PRESSURE,
        albedo=ALBEDO,diffuse_model=diffuse_model,timezone=TIMEZONE):
        #initiliase solar libraries
        #version 1 created on 25/1/18 by fm based on previous SolarModels created on 16/8/17
        self.albedo = albedo
        self.diffuse_model = diffuse_model
        self.poa_model = poa_model
        self.package = package  
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.surface_zenith = surface_zenith
        self.surface_azimuth  = surface_azimuth   #180 North
        self.transmittance = transmittance
        self.pressure = pressure
        self.timezone = timezone
        
    def getsolarseries(self,datetimeindex_utc:pd.DatetimeIndex,outputs:list,
                       linketurbidity=None):
        datetimeindex_utc = pd.DatetimeIndex(datetimeindex_utc,ambiguous='NaT')
        #datetimeindex_utc = datetimeindex_tz.tz_convert(None) 
        df=pd.DataFrame({'datetime':datetimeindex_utc},index=datetimeindex_utc) 
        #parameters & related pvlib functions
        csky_ineichen_poa = ["clearskypoa"]
        csky_ineichen = csky_ineichen_poa + ["clearsky"]
        airmassabsolute = csky_ineichen + ["airmassabsolute"]
        airmassrelative = airmassabsolute + ["airmassrelative"]
        aoi =["aoi"]
        aoi_projection = ["aoiprojection"]    
        zenith = airmassrelative + aoi + aoi_projection + ["zenith"]
        apparent_zenith = csky_ineichen + ["apparentzenith"]
        azimuth = csky_ineichen_poa + aoi + aoi_projection + ["azimuth"]
        linke_turbidity = csky_ineichen + ["linketurbidity"]
        extraradiation = csky_ineichen + ["extraradiation"]
        dayofyear = extraradiation + ["dayofyear"]        
        spa_python = apparent_zenith + azimuth + zenith + aoi_projection  
        #calculation of needed parameters 
        if any(i in outputs for i in dayofyear):
            df["dayofyear"] = datetimeindex_utc.dayofyear
        if any(i in outputs for i in extraradiation):
            df["extraradiation"] = irr.extraradiation(df["dayofyear"])
        if any(i in outputs for i in linke_turbidity):
            if linketurbidity == None: df["linketurbidity"] = csky.lookup_linke_turbidity(datetimeindex_utc,self.latitude,self.longitude)
        if any(i in outputs for i in spa_python):
            spa = sp.spa_python(time=datetimeindex_utc,latitude=self.latitude,longitude=self.longitude,altitude=self.altitude)
            """[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
            applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
            [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
            solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007."""
            if any(i in outputs for i in apparent_zenith):
                df["apparentzenith"] = spa["apparent_zenith"] #df["apparentzenith"] changed on 19/11/19
            if any(i in outputs for i in zenith):
                df["zenith"] = spa["zenith"]                
                if any(i in outputs for i in airmassrelative):
                    df["airmassrelative"] = atm.relativeairmass(df["zenith"],model=APPARENT_ZENITH_MODEL)
                    if any(i in outputs for i in airmassabsolute):
                        df["airmassabsolute"] = atm.absoluteairmass(df["airmassrelative"],self.pressure)
            if any(i in outputs for i in azimuth):
                df["azimuth"] = spa["azimuth"]
            if any(i in outputs for i in aoi):
                df["aoi"] = irr.aoi(self.surface_zenith,self.surface_azimuth,df["zenith"],df["azimuth"])
                df.loc[df.aoi>90,"aoi"]=90                
            if any(i in outputs for i in aoi_projection):
                df["aoiprojection"] = irr.aoi_projection(self.surface_zenith,self.surface_azimuth,df["zenith"],df["azimuth"])
        if any(i in outputs for i in csky_ineichen):
            csky_ineichen = csky.ineichen(df["apparentzenith"],df["airmassabsolute"],df["linketurbidity"],self.altitude,df["extraradiation"]) 
            df["dhi"] = csky_ineichen["dhi"]
            df["dni"] = csky_ineichen["dni"]
            df["ghi"] = csky_ineichen["ghi"] #dhi = ghi - dni*cos_zenith
            if any(i in outputs for i in csky_ineichen_poa): 
                #check if air mass absolute or relative 
                #print("Poa model: "+str(self.poa_model)) #test line
                #if self.poa_model=="perez": print("Perez diffuse model: "+str(self.diffuse_model)) #test line
                total_irrad = irr.total_irrad(surface_tilt=self.surface_zenith,
                                              surface_azimuth=self.surface_azimuth,
                                              apparent_zenith=df["apparentzenith"],
                                              azimuth=df["azimuth"],
                                              dni=df["dni"],
                                              ghi=df["ghi"],
                                              dhi=df["dhi"],
                                              dni_extra=df["extraradiation"],
                                              airmass=df["airmassrelative"],
                                              albedo=self.albedo,
                                              surface_type=None, 
                                              model=self.poa_model, 
                                              model_perez=self.diffuse_model)
                df["poa_global"]=total_irrad["poa_global"] #W/m^2
                df["poa_direct"]=total_irrad["poa_direct"]
                df["poa_diffuse"]=total_irrad["poa_diffuse"]
                # diffuse = sky + ground # total = beam + diffuse 
        return df 
    


    def getirradiancediffuseskyinplane(self,datetime_series):
        #added on 5/10/18 to transpose irradiance diffuse in plane from horizontal:
        if self.package == 'pvlib' and self.poa_model == 'klucher1979':
            #extracted from pvlib-pyhton function klucher in irradiance module
            #Determine diffuse irradiance from the sky on a tilted surface using Klucher's 1979 model#
            dhi = self.getirradiance(datetime_series)["dhi"]
            ghi = self.getirradiance(datetime_series)["ghi"]
            sza = self.getsolarzenithangle(datetime_series)
            saa = self.getsolarazimuthangle(datetime_series)
            return irr.klucher(self.surface_zenith,self.surface_azimuth ,dhi,ghi,sza,saa)
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
        if self.package == 'pvlib' and self.poa_model == 'perez1990':
            #extracted from pvlib-pyhton function perez in irradiance module
            #Determine diffuse irradiance from the sky on a tilted surface using one of the Perez models
            dhi = self.getirradiance(datetime_series)["dhi"]
            dni = self.getirradiance(datetime_series)["dni"]
            amr = self.getairmassrelative(datetime_series)
            sza = self.getsolarzenithangle(datetime_series)
            saa = self.getsolarazimuthangle(datetime_series)
            dnxt = self.getextraterrestrialradiation(datetime_series)
            return irr.perez(self.surface_zenith,self.surface_azimuth ,dhi,dni, dnxt,sza,saa,amr,
            model=self.diffuse_model, return_components=False)

    def getirradianceinplane(self,datetime_series):
        #DEV NOTE 16/4/19: compare with clear sky 
        if self.package == 'pvlib':
            #extracted from pvlib-pyhton function globalinplane in irradiance module
            #determine the three components on in-plane irradiance  
            aoi = self.getangleofincidence(datetime_series)
            dni = self.getirradiance(datetime_series)["dni"]
            poa_sky_diffuse = self.getirradiancediffuseskyinplane(datetime_series)
            poa_ground_diffuse = self.getirradiancediffusegroundinplane(datetime_series)            
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
            
    def getirradiancediffusegroundinplane(self,datetime_series):
        #extracted from pvlib-pyhton function grounddiffuse in irradiance module
        #Estimate diffuse irradiance from ground reflections given irradiance, albedo, and surface tilt
        if self.package == 'pvlib':
            ghi = self.getirradiance(datetime_series)["ghi"]
            #albedo originally 0.25 in pvlib. put 0.15 as in Meteonorm
            return irr.grounddiffuse(self.surface_zenith,ghi,albedo=0.25,surface_type=None)
        
   
    
    def getsuninout(self,date_utc:pd.DatetimeIndex,freq="min"): #26/2/20 date with 00:00:00 otherwise timedelta into next day
        #freq: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        
        #DEV NOTE 6/5/19: could be improved with minimum according to irradiance value        
        """ estimate when sun is in or out a tilted surface """
        #Created on 4/4/19 by wsfm 
        #datetimeindex_utc 00:00. freq at: 
        #https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        #dt = pd.DataFrame({'start':date_utc}) #day start
        dt = pd.DataFrame({'start':date_utc.values},index=date_utc.values) #day start
        dt["end"]=dt.apply(lambda x: x["start"]+dtt.timedelta(hours=23,minutes=59,seconds=59),axis=1) #day end  
        utc = pd.concat([pd.DataFrame(pd.date_range(start=dt.loc[index,"start"],
        end=dt.loc[index,"end"],freq=freq),columns=["utc"]) for index in list(dt.index)],ignore_index=True)        
        utc_dti = pd.DatetimeIndex(utc.utc,tz=None)
        spa=sp.spa_python(utc.utc,self.latitude,self.longitude).loc[:,['azimuth','zenith']]               
        
        df = pd.DataFrame({'aoi':irr.aoi(self.surface_zenith,self.surface_azimuth,
        spa.loc[:,'zenith'], spa.loc[:,'azimuth']),'date':utc_dti.date,'datetime':utc_dti},index=utc_dti)        
        
    
        
    
        """
        df_test = pd.DataFrame({'aoi':irr.aoi(self.surface_zenith,self.surface_azimuth,
        spa.loc[:,'zenith'], 
        spa.loc[:,'azimuth']),
        'zenith':spa.loc[:,'zenith'],
        'azimuth':spa.loc[:,'azimuth'], An
        'date':utc.date,
        'datetime':utc},index=utc)
        """
                
        df_valid=df[(df.aoi>0)&(df.aoi<90)]       
        
        df_valid.loc[:,"day"]=df_valid.loc[:,"date"].copy(deep=True)
        
        df_valid_day=df_valid.groupby(by=['date'], as_index=True) #group df_valid per dayofyear and identify min & max hour per day
        datetime = df_valid_day['datetime']
        dt_min = datetime.min() #first daily connections in hour and minutes
        dt_max = datetime.max() #last daily connection in hour and minutes        
        date = df_valid_day['date'].min()
    
        grp_aoi_min = df_valid_day.apply(lambda x: x[x["aoi"]==x["aoi"].min()])
        grp_peak = grp_aoi_min.groupby(by="day") 
        #grp_peak = grp_aoi_min.groupby([pd.Grouper(freq="1D",key="datetime")]) #27/2/20 Grouper will create additional not available dates
        datetime_peak = grp_peak["datetime"]
        
        results = pd.DataFrame({'sun_on': dt_min,
        'sun_off':dt_max,  
        'sun_max': datetime_peak.min(), 
        'aoi_min': df_valid_day["aoi"].min(), #23/2/20 added 
         'date':date})  #include both min & max in a new df_valid and plot the result

        #results = results.merge(df["date"],how="right",on="date")
             
        
        return results
              
    def getsunrisesettransit(self,datetimeindex_utc:pd.DatetimeIndex):
        #TO BEC CHECKED
        """ calculate sunrise, sunset & transit according to the proposed package"""
        #DEV NOTE 3/4/19: dateimeindex and "times must be localized"
        #Returns pandas.DataFrame
        #index is the same as input `times` argument
        #columns are 'sunrise', 'sunset', and 'transit'
        #https://pvlib-python.readthedocs.io/en/latest/_modules/pvlib/solarposition.html        
        #version 1 created on 30/10/17 by fm      
        #identify datetime components for the calculation        
        #Function information (extracted on 30/10/17):
        #Calculate the sunrise, sunset, and sun transit times using the NREL SPA algorithm described in
        #Reda, I., Andreas, A., 2003. Solar position algorithm for solar radiation applications. 
        #Technical report: NREL/TP-560- 34302. Golden, USA, http://www.nrel.gov.
        if self.package == 'pvlib':
            datetimeindex_tz = pd.DatetimeIndex(datetimeindex_utc,tz=self.timezone,ambiguous='NaT')                               
            sunpos = sp.get_sun_rise_set_transit(datetimeindex_tz, self.latitude, self.longitude)               
            spa = sp.spa_python(datetimeindex_utc,self.latitude,self.longitude)            
            #DEV NOTE 3/4/19: not clear why dateindex necessary again
            datetime = pd.DatetimeIndex(datetimeindex_tz)            
            df_out=pd.DataFrame({'date':datetime.date,        
            'sunrise': sunpos.loc[:,'sunrise'].values,
            'sunset': sunpos.loc[:,'sunset'].values,
            'suntransit': sunpos.loc[:,'transit'].values,
            'datetime':datetimeindex_utc,
            'datetime_tz':datetimeindex_tz,
            'zenith':spa.loc[:,"zenith"].values #23/2/20 added for aoi 
            },
            index = datetimeindex_tz)
            #index = datetime.date)
        #else: 
        #    raise ValueError('times must be localized')
        return df_out
    
      
    