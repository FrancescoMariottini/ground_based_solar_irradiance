# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:46:18 2018

@author: wsfm
"""

import pandas as pd
import numpy as np

#import solar libraries
from pvlib import irradiance as irr
from pvlib import solarposition as sp
from pvlib import atmosphere as atm
from pvlib import clearsky as csky


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
        