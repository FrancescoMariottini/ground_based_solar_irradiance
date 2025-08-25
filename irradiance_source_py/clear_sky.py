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
#DEV NOTE 5/1/20: df with datetimeindex progressively adding parameters
#DEV NOTE 05/12/19: should pvlib & meteodataquality be harmonised?
#DEV NOTE 28/11/18: naming 'datetime' & 'datetime_series' to be harmonised

"""

"""SETTING MODULES FOLDER & IMPORTING MODULES"""
# DEV NOTE: declaring a method only importing still load the entire module
#https://softwareengineering.stackexchange.com/questions/187403/import-module-vs-from-module-import-function

# importing sys to locate modules in different paths
#commented not used

# no need to import single method explicitely since entire library imported anyway
# however if only a few methods imported could be useful (e.g. if library really needed)
from os.path import join, dirname
from os import getcwd

# widely used pandas
import pandas as pd
# numpy importe for some mathematical operations and definitions
import numpy as np

import itertools as ittl

#20/5/21 import date_operation to avoid replicating functions 
#20/5/21function from nb should be transfered
import data_operations as dtop

# commented not used
# from pandas import concat

pd.set_option('display.max_columns', 10)

# import pvlib.irradiance as irr
# import pvlib_python.pvlib.solarposition as sp
# import pvlib_python.pvlib.atmosphere as atm
# import pvlib_python.pvlib.clearsky as csky

from pvlib import irradiance as irr
from pvlib import spa as spnrel #SNL python implementation of NREL algorithm
# solarposition includeds different approaches. sun_rise_set_transit_spa imported being based on NREL
from pvlib import atmosphere as atm
from pvlib import clearsky as csky
import pvlib #for solarposition

from pvlib import location as lct
from datetime import datetime, timedelta, timezone

# used for mnanipulation of getsunrisesettransit2 version 2

from typing import List, Dict

import decorators



"""GLOBAL VARIABLES (default values)"""
#using set to automatically remove duplicates (if any) and order
#apparent zenith includes atmospheric correction
_OUTPUTS = set(["irradiancetotalpoa", "irradiancebeampoa", "irradiancediffusepoa", "angleofincidence", "irradiancebeampoa",
                "irradiancediffusepoa", "angleofincidence", "irradiancediffusegroundpoa", "irradiancediffuseskypoa",
                "irradiancebeam", "zenith", "azimuth", "irradiancetotal", "irradiancediffuse", "extraradiation", "airmassrelative",
                "angleofincidence", "airmassabsolute", "apparentzenith", "linketurbidity", "dayofyear",
                "angleofincidenceprojection", "equation_of_time", "elevation", "apparentelevation"])
_OUTPUTS = list(_OUTPUTS)
PACKAGE = "pvlib"  # DEV NOTE 3/5/19 could be eliminated
ASSETS_FOLDER = join(dirname(getcwd()) + r"/assets/")
CSVS_FOLDER = join(dirname(getcwd()) + r"/outputs/csvs/")


# position A
#https://digimap.edina.ac.uk/
LATITUDE = 51 + 27 / 60 + 14.4 / 3600
LATITUDE = 52.7616
# Also, the observer’s geographical longitude is considered positive west, or negative east from Greenwich,
# while for solar radiation applications, it is considered negative west, or positive east from Greenwich
LONGITUDE = 3 + 23 / 60 + 14.3 / 3600
LONGITUDE = - (1.2406)
ALTITUDE = 79
TIMEZONE = "utc"


# surface
SURFACE_ZENITH = 0
SURFACE_AZIMUTH = 180
# BEWARE:
# surface_azimuth must be >=0 and <=360. The azimuth convention is APPARENT_ZENITH_MODEL defined as degrees east of north
# (e.g. North = 0, South=180 East = 90, West = 270).
# Source: perez(surface_zenith, surface_azimuth, dhi, dni, dni_extra,solar_zenith, solar_azimuth, airmass,model='allsitescomposite1990', return_components=False)


# atmosphere parameters [pvlib.solar_position]
# DEV 31/1/21: 'these args are the same for each thread'
PRESSURE = 101325.
TEMPERATURE = 12
DELTA_T = 67.0
ATMOS_REFRACT = 0.5667
TRANSMITTANCE = 0.5
#numthread included for completeness but may not be necessary depending on used function
NUMTHREADS = 4


# modeling
ALBEDO = 0.25
_APPARENT_ZENITH_MODELS: List = ['kastenyoung1989', 'kasten1966', 'simple', 'pickering2002', 'youngirvine1967', 'young1994',
                          'gueymard1993']
APPARENT_ZENITH_MODEL = "kastenyoung1989"  # used for atm module
_CLEAR_SKY_MODELS: List = ["liujordan", 'ineichenperez']
CLEAR_SKY_MODEL = "liujordan"
#10/5/21 replaced liujordan with Pere seems better and working now
CLEAR_SKY_MODEL = "ineichenperez"

#perez diffuse model for PoA irradiance
DIFFUSE_MODELS: List = ['1990', 'allsitescomposite1990', #(same as '1990')
                        'allsitescomposite1988', 'sandiacomposite1988', 'usacomposite1988', 'france1988',
                        'phoenix1988', 'elmonte1988', 'osage1988', 'albuquerque1988', 'capecanaveral1988', 'albany1988']
DIFFUSE_MODEL = "allsitescomposite1990"
# POA_MODEL="perez" DEV NOTE 3/19 run error
POA_MODELS: List = ["perez", "klucher1979"]
POA_MODEL = "perez" #isotropic
_EXTRARADIATION_METHODS: List = ['pyephem', 'spencer', 'asce', 'nrel']
EXTRARADIATION_METHOD = 'spencer'
SOLAR_CONSTANT = 1366.1 #for extraradiation
#1364 in https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.clearsky.ineichen.html

EPOCH_YEAR = 2014 #The year in which a day of year input will be calculated.
# Only applies to day of year input used with the pyephem or nrel methods.
# require installation of tables module
_LINKE_TURBIDITIES_PATH = join(dirname(getcwd())+r"/data/LinkeTurbidities.h5")


"""STABILITY PARAMETERS"""
AOI_MAX = 70  # [ISO 9847]
# 0.2: uncertainty 95% of HOURLY values for moderate quality (class C) radiometers [WMO 2017)
# @fm: enough due to additional filter and uncertainty of mcs
KCS_UNC = 0.2
# 23/5/21 quite strict for low values, maybe better to use absolute values G_UNC
# 23/5/21 still due to time shift could be very restrictive
# 23/5/21 uncertainty analysis could be more precise with PhD02 but PhD03 treated indipendently
# 3% (hourly class A) on 1000 also directional response ONLY on class C
G_UNC = 30



# adapted from 2% variation from average in [ISO 9847]
KCS_CV_MAX = 0.02
PWR_CV_MAX = 0.1
# daily criteria
PEARSON_MIN = 0.8  # strong correlation
# other criteria
# daylight hours IEC 61724-1-2017
G_MIN = 20



"""
22/12/20 Datetime processing for testing
"""
from typing import List

#pd.options.display.max_columns = None
#options.display.max_rows = None
#pd.options.display.max_colwidth = None

CSVS_STARTWITH = 'ch'
DATETIME_COLUMN = 'tmstamp'
COLUMNS_TO_EXCLUDE = ['pt1000_ppuk_sensol']


#folder for py file testing
ISC_IRRADIANCE_FILES_FOLDER = join(dirname(getcwd())+r"/assets/isc_irradiance_files/")



# setup
class SolarLibrary:
    # deve note rename add
    def __init__(self, latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE,
                 surface_zenith=SURFACE_ZENITH, surface_azimuth=SURFACE_AZIMUTH, package=PACKAGE,
                 apparent_zenith_model=APPARENT_ZENITH_MODEL,
                 poa_model=POA_MODEL, transmittance=TRANSMITTANCE, pressure=PRESSURE,
                 albedo=ALBEDO, diffuse_model=DIFFUSE_MODEL, timezone=TIMEZONE, temperature=TEMPERATURE,
                 delta_t=DELTA_T, atmos_refract=ATMOS_REFRACT, solar_constant=SOLAR_CONSTANT,
                 extraradiation_method = EXTRARADIATION_METHOD, epoch_year=EPOCH_YEAR,
                 clear_sky_model = CLEAR_SKY_MODEL, numthreads = NUMTHREADS):
        # https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.solarposition.get_solarposition.html
        """

        :param latitude: Latitude in decimal degrees. Positive north of equator, negative to south
        :param longitude: Longitude in decimal degrees. Positive east of prime meridian, negative to west.
        :param altitude:
        :param surface_zenith:
        :param surface_azimuth:
        :param package:
        :param apparent_zenith_model:
        :param poa_model:
        :param transmittance:
        :param pressure:
        :param albedo:
        :param diffuse_model:
        :param timezone:
        :param temperature:
        :param delta_t:
        :param atmos_refract:
        :param solar_constant:
        :param extraradiation_method:
        :param epoch_year:
        :param clear_sky_model:
        :param numthreads:
        """

        # initiliase solar libraries
        # version 1 created on 25/1/18 by fm based on previous SolarModels created on 16/8/17
        if apparent_zenith_model not in _APPARENT_ZENITH_MODELS:
            raise ValueError('%s is not a valid model for relativeairmass', apparent_zenith_model)
        if extraradiation_method not in _EXTRARADIATION_METHODS:
            raise ValueError('%s is not a valid method for extraradiation', extraradiation_method)
        if clear_sky_model not in _CLEAR_SKY_MODELS:
            raise ValueError('%s is not a valid model for clear sky', clear_sky_model)

        self.albedo = albedo
        self.diffuse_model = diffuse_model
        self.poa_model = poa_model
        self.apparent_zenith_model = apparent_zenith_model
        self.package = package
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.surface_zenith = surface_zenith
        self.surface_azimuth = surface_azimuth  # 180 North
        self.transmittance = transmittance
        self.pressure = pressure #avg. yearly air pressure in Pascals.
        self.timezone = timezone
        self.temperature = temperature #avg. yearly air temperature in degrees C
        self.delta_t = delta_t
        self.atmos_refract = atmos_refract #The approximate atmospheric refraction (in degrees) at sunrise and sunset.
        self.solar_constant = solar_constant
        self.extraradiation_method = extraradiation_method
        self.epoch_year= epoch_year
        self.clear_sky_model = clear_sky_model
        self.numthreads = numthreads

        self._outputs = _OUTPUTS

    # 23/1/25 algorithm previously used to test all the step in the spa algorithm
    # 23/1/25 later replaced with pvlib.solarposition.get_solarposition
    def solar_position(self, datetime_object: datetime, observer_longitude= None, observer_latitude= None,
                       observer_elevation= None, delta_t= None, local_pressure= None, local_temp=None,
                       atmos_refract=None,  numthreads=None):
        """

        :param datetime_object:
        :param observer_longitude: Positive east of prime meridian, negative to west.
        :param observer_latitude: Positive north of equator, negative to sout
        :param observer_elevation:
        :param delta_t:
        :param local_pressure:
        :param local_temp:
        :param atmos_refract:
        :param numthreads:
        :return:
        """
        # step-by-step implementation of Solar Position Algorithm for Solar Radiation Applications
        # used as example not in calculation
        if observer_longitude is None:
            observer_longitude = self.longitude
        if observer_latitude is None:
            observer_latitude = self.latitude
        if observer_elevation is None:
            observer_elevation = self.altitude
        # delta T is the difference between the Earth rotation time and the Terrestrial Time (TT
        if delta_t is None:
            delta_t = self.delta_t
        if observer_elevation is None:
            observer_elevation = self.altitude
        if local_pressure is None:
            local_pressure = self.pressure
        if local_temp is None:
            local_temp = self.temperature
        if atmos_refract is None:
            atmos_refract = self.atmos_refract
        if numthreads is None:
            numthreads = self.numthreads

        unixtime = dtop.datetime_to_utc(datetime_object)

        julian_day = spnrel.julian_day(unixtime)

        julian_ephemeris_day = spnrel.julian_ephemeris_day(julian_day, delta_t)

        julian_century = spnrel.julian_century(julian_day)

        julian_ephemeris_century = spnrel.julian_ephemeris_century(julian_ephemeris_day)

        julian_ephemeris_millennium = spnrel.julian_ephemeris_millennium(julian_ephemeris_century)
        # renaming
        jme = julian_ephemeris_millennium

        heliocentric_longitude = spnrel.heliocentric_longitude(jme)

        heliocentric_latitude = spnrel.heliocentric_latitude(jme)

        heliocentric_radius_vector = spnrel.heliocentric_radius_vector(jme)

        geocentric_longitude = spnrel.geocentric_longitude(heliocentric_longitude)

        geocentric_latitude = spnrel.geocentric_latitude(heliocentric_latitude)

        mean_elongation = spnrel.mean_elongation(julian_ephemeris_century)

        mean_anomaly_sun = spnrel.mean_anomaly_sun(julian_ephemeris_century)

        mean_anomaly_moon = spnrel.mean_anomaly_moon(julian_ephemeris_century)

        moon_argument_latitude = spnrel.moon_argument_latitude(julian_ephemeris_century)

        moon_ascending_longitude = spnrel.moon_ascending_longitude(julian_ephemeris_century)
        # TBC
        x0, x1, x2, x3, x4 = mean_elongation, mean_anomaly_sun, mean_anomaly_moon, moon_argument_latitude, moon_ascending_longitude

        longitude_nutation = spnrel.longitude_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4)

        obliquity_nutation = spnrel.obliquity_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4)

        mean_ecliptic_obliquity = spnrel.mean_ecliptic_obliquity(julian_ephemeris_millennium)

        true_ecliptic_obliquity = spnrel.true_ecliptic_obliquity(mean_ecliptic_obliquity, obliquity_nutation)
        # TBC, renaming
        earth_radius_vector = heliocentric_radius_vector

        aberration_correction = spnrel.aberration_correction(earth_radius_vector)

        apparent_sun_longitude = spnrel.apparent_sun_longitude(geocentric_longitude, longitude_nutation,
                                                               aberration_correction)

        mean_sidereal_time = spnrel.mean_sidereal_time(julian_day, julian_century)

        apparent_sidereal_time = spnrel.apparent_sidereal_time(mean_sidereal_time, longitude_nutation,
                                                               true_ecliptic_obliquity)

        geocentric_sun_right_ascension = spnrel.geocentric_sun_right_ascension(apparent_sun_longitude,
                                                                               true_ecliptic_obliquity,
                                                                               geocentric_latitude)

        geocentric_sun_declination = spnrel.geocentric_sun_declination(apparent_sun_longitude, true_ecliptic_obliquity,
                                                                       geocentric_latitude)
        # renaming
        sun_right_ascension = geocentric_sun_right_ascension

        local_hour_angle = spnrel.local_hour_angle(apparent_sidereal_time, observer_longitude,
                                                   sun_right_ascension)

        equatorial_horizontal_parallax = spnrel.equatorial_horizontal_parallax(earth_radius_vector)

        uterm = spnrel.uterm(observer_latitude)
        # renaming
        u = uterm

        xterm = spnrel.xterm(u, observer_latitude, observer_elevation)

        yterm = spnrel.yterm(u, observer_latitude, observer_elevation)

        parallax_sun_right_ascension = spnrel.parallax_sun_right_ascension(xterm, equatorial_horizontal_parallax,
                                                                           local_hour_angle, geocentric_sun_declination)
        #TBC not used
        topocentric_sun_right_ascension = spnrel.topocentric_sun_right_ascension(geocentric_sun_right_ascension,
                                                                                 parallax_sun_right_ascension)

        topocentric_sun_declination = spnrel.topocentric_sun_declination(
            geocentric_sun_declination,
            xterm,
            yterm,
            equatorial_horizontal_parallax,
            parallax_sun_right_ascension,
            local_hour_angle)

        topocentric_local_hour_angle = spnrel.topocentric_local_hour_angle(local_hour_angle,
                                                                           parallax_sun_right_ascension)

        topocentric_elevation_angle_without_atmosphere = spnrel.topocentric_elevation_angle_without_atmosphere(
            observer_latitude,
            topocentric_sun_declination,
            topocentric_local_hour_angle
            )
        # renaming
        topocentric_elevation_angle_wo_atmosphere = topocentric_elevation_angle_without_atmosphere

        topocentric_zenith_angle_wo_atmosphere = spnrel.topocentric_zenith_angle(topocentric_elevation_angle_without_atmosphere)

        atmospheric_refraction_correction = spnrel.atmospheric_refraction_correction(local_pressure, local_temp,
                                                                                     topocentric_elevation_angle_wo_atmosphere,
                                                                                     atmos_refract)

        topocentric_elevation_angle = spnrel.topocentric_elevation_angle(
            topocentric_elevation_angle_without_atmosphere,
            atmospheric_refraction_correction)

        topocentric_zenith_angle = spnrel.topocentric_zenith_angle(topocentric_elevation_angle)

        topocentric_astronomers_azimuth = spnrel.topocentric_astronomers_azimuth(topocentric_local_hour_angle,
                                                                                 topocentric_sun_declination,
                                                                                 observer_latitude)
        #Azimuth limit to the range from 0/ to 360/. Note that azimuth is measured eastward from north.
        topocentric_azimuth_angle = spnrel.topocentric_azimuth_angle(topocentric_astronomers_azimuth)

        sun_mean_longitude = spnrel.sun_mean_longitude(julian_ephemeris_millennium)

        equation_of_time = spnrel.equation_of_time(sun_mean_longitude, geocentric_sun_right_ascension,
                                                   longitude_nutation, true_ecliptic_obliquity)

        # equivalent to
        # Calculate the solar position assuming unixtime is a numpy array
        # theta: topocentric_zenith_angle
        # theta0: topocentric_zenith_angle
        # e: topocentric_elevation_angle
        # e0
        # gamma = topocentric_astronomers_azimuth
        # phi = topocentric_azimuth_angle

        #theta, theta0, e, e0, phi, eot = spnrel.solar_position_numpy(unixtime, observer_latitude, observer_longitude,
        #observer_elevation, local_pressure, local_temp, delta_t, atmos_refract, numthreads, sst=False, esd=False)

        return topocentric_zenith_angle, topocentric_zenith_angle_wo_atmosphere, topocentric_elevation_angle, topocentric_elevation_angle_wo_atmosphere, topocentric_azimuth_angle, equation_of_time

    #values different from other solar position algorithm to be verified
    #@decorators.timer
    def getsolardataframe(self, datetimeindex_utc: pd.DatetimeIndex, outputs: List[str],
                          solardataframe: pd.DataFrame = None, poa_model: str = None, clear_sky_model: str = None,
                          inplace: bool = False) -> pd.DataFrame:
        # in place to save last used models
        # any not used since printed after
        #if any(o in _OUTPUTS for o in outputs):
        _ = [str(o) for o in outputs if o not in self._outputs]
        if len(_) > 0:
            raise ValueError(f"{','.join(_)} not among the available outputs.")
        datetimeindex_utc = pd.DatetimeIndex(datetimeindex_utc, ambiguous='NaT')
        # datetimeindex_utc = datetimeindex_tz.tz_convert(None)
        if solardataframe is None:
            df = pd.DataFrame({'datetime': datetimeindex_utc}, index=datetimeindex_utc)
        elif all(pd.DatetimeIndex(solardataframe.index) == datetimeindex_utc):
            # simple renaming
            df = solardataframe
        columns_to_drop = []
        #adding outputs based on dependencies
        if "irradiancetotalpoa" in outputs:
            outputs += ["irradiancebeampoa", "irradiancediffusepoa", "angleofincidence"]
            columns_to_drop += ["irradiancebeampoa", "irradiancediffusepoa"]
        if "irradiancediffusepoa" in outputs:
            outputs += ["irradiancediffusegroundpoa", "irradiancediffuseskypoa"]
            columns_to_drop += ["irradiancediffusegroundpoa", "irradiancediffuseskypoa"]
        if "irradiancediffuseskypoa" in outputs:
            outputs += ["irradiancebeam", "zenith", "azimuth"]
            if poa_model is None:
                poa_model = self.poa_model
            elif inplace:
                self.poa_model = poa_model
            if poa_model == 'klucher1979':
                outputs += ["irradiancetotal"]
            elif poa_model == 'perez1990':
                outputs += ["irradiancediffuse", "extraradiation", "airmassrelative"]
                columns_to_drop += ["irradiancetotal", "irradiancebeam", "irradiancediffuse"]
        if "irradiancediffusegroundpoa" in outputs:
            outputs += ["irradiancetotal"]
            columns_to_drop += ["irradiancetotal", "irradiancebeam"]
        if "irradiancebeampoa" in outputs:
            outputs += ["irradiancetotal", "angleofincidence"]
            columns_to_drop += ["irradiancetotal"]
        if "irradiancebeam" in outputs or "irradiancetotal" in outputs or "irradiancediffuse" in outputs:
            outputs += ["airmassabsolute", "extraradiation"]
            if clear_sky_model is None:
                clear_sky_model = self.clear_sky_model
            elif inplace:
                self.clear_sky_model = clear_sky_model
            if clear_sky_model == 'ineichenperez':
                outputs += ["apparentzenith", "airmassabsolute", "linketurbidity", "extraradiation"]
            elif clear_sky_model =='liujordan':
                outputs += ["zenith"]
        if "extraradiation" in outputs:
            outputs.append("dayofyear")
        if "airmassabsolute" in outputs:
            outputs.append("airmassrelative")
        if "airmassrelative" in outputs:
            outputs.append("zenith")
        if "angleofincidence" in outputs or "angleofincidenceprojection" in outputs:
            outputs += ["azimuth", "zenith"]

        # dropping all columns to be recalculate if conditions changed
        if len(df.columns.tolist()) > 1:
            columns_subset = []
            for c in df.columns.tolist():
                if c in columns_to_drop:
                    columns_subset += [c]
            df.drop(columns=columns_subset, inplace=True)

        #starting calculation based on outputs
        def notindf(parameter: str) -> bool:
            #if solardataframe is not None and parameter in solardataframe.columns.tolist():
            if parameter in df.columns.tolist():
                return False
            else:
                return True

        if "dayofyear" in outputs and notindf("dayofyear"):
            df["dayofyear"] = datetimeindex_utc.dayofyear
        if "extraradiation" in outputs and notindf("extraradiation"):
            df["extraradiation"] = irr.get_extra_radiation(df["dayofyear"], self.solar_constant, self.extraradiation_method,
                                      self.epoch_year)
        if "linketurbidity" in outputs and notindf("linketurbidity"):
            #tables library needs to be installed
            #http://www.pytables.org/
            df["linketurbidity"] = csky.lookup_linke_turbidity(datetimeindex_utc, self.latitude, self.longitude,
                                                               filepath=_LINKE_TURBIDITIES_PATH)
        parameters = ["apparentzenith", "zenith", "azimuth", "equation_of_time", "apparentelevation", "elevation"]
        if any(o in parameters for o in outputs) and any(notindf(o) for o in parameters):
        #if ("apparentzenith" in outputs or "zenith" in outputs or "azimuth" in outputs) and:
            #[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
            #applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
            #[2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
            #solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.
            #spa function requiring unixtime
            #OLD VERSION
            """app_zenith, zenith, app_elevation, elevation, azimuth, eot = spnrel.solar_position(
            dtop.datetimeindex_to_utc(datetimeindex_utc),  self.latitude, self.longitude,
            self.altitude, self.pressure, self.temperature, self.delta_t, self.atmos_refract, numthreads=4)"""
            #DEV NOTE 5/5/21 using utc although not strictly necessary:  Must be localized or UTC will be assumed.
            #https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/solarposition.html#get_solarposition
            #[1] I. Reda and A. Andreas, Solar position algorithm for solar radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
            #[2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.
            #[3] NREL SPA code: http://rredc.nrel.gov/solar/codesandalgorithms/spa/
            #latitude (float) – Latitude in decimal degrees. Positive north of equator, negative to south.
            #longitude (float) – Longitude in decimal degrees. Positive east of prime meridian, negative to west.
            df_spa = pvlib.solarposition.get_solarposition(time=datetimeindex_utc, latitude=self.latitude,
            longitude=self.longitude, altitude=self.altitude, pressure=self.pressure, method='nrel_numpy',
            temperature=self.temperature)
            #adding for check purpose
            df["equation_of_time"] = df_spa.loc[:,"equation_of_time"].values
            if "apparentzenith" in outputs and notindf("apparentzenith"):
                df["apparentzenith"] = df_spa.loc[:,"apparent_zenith"].values
            if "zenith" in outputs and notindf("zenith"):
                df["zenith"] = df_spa.loc[:,"zenith"].values
            #if cosin == True: #as reminder
            #    sza = cos(deg2rad(sza))
            if "azimuth" in outputs and notindf("azimuth"):
                df["azimuth"] = df_spa.loc[:,"azimuth"].values
            if "apparentelevation" in outputs and notindf("apparentelevation"):
                df["apparentelevation"] = df_spa.loc[:,"apparent_elevation"].values
            if "elevation" in outputs and notindf("elevation"):
                df["elevation"] = df_spa.loc[:,"elevation"].values
        if "airmassrelative" in outputs and notindf("airmassrelative"):
            df["airmassrelative"] = atm.get_relative_airmass(df["zenith"], model=self.apparent_zenith_model)
        if  "airmassabsolute" in outputs and notindf("airmassabsolute"):
            df["airmassabsolute"] = atm.get_absolute_airmass(df["airmassrelative"], self.pressure)
        parameters = ["angleofincidence", "angleofincidenceprojection"]
        if any(o in parameters for o in outputs) and any(notindf(o) for o in parameters):
        #if "angleofincidence" in outputs or "angleofincidenceprojection" in outputs:
            if "angleofincidence" in outputs and notindf("angleofincidence"):
                df["angleofincidence"] = irr.aoi(self.surface_zenith, self.surface_azimuth, df["zenith"], df["azimuth"])
                #DEV NOTE 2/4/21: removed since could be useful
                #df.loc[df.angleofincidence > 90, "angleofincidence"] = 90
            if "angleofincidenceprojection" and notindf("angleofincidenceprojection"):
                df["angleofincidenceprojection"] = irr.aoi_projection(self.surface_zenith, self.surface_azimuth, df["zenith"],
                                                         df["azimuth"])
        # 12/01/21 possibility of providing already outputs not considered
        if any(i in ["irradiancediffuse", "irradiancebeam", "irradiancetotal"] for i in outputs):
            if clear_sky_model == 'ineichenperez':
                # https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.clearsky.ineichen.html
                # P. Ineichen and R. Perez, “A New airmass independent formulation for the Linke turbidity coefficient”,
                # Solar Energy, vol 73, pp. 151-157, 2002.
                # R. Perez et. al., “A New Operational Model for Satellite-Derived Irradiances: Description and Validation”,
                # Solar Energy, vol 73, pp. 307-317, 2002.
                csky_ineichen = csky.ineichen(df["apparentzenith"], df["airmassabsolute"], df["linketurbidity"],
                                              self.altitude, df["extraradiation"])

                df["irradiancediffuse"] = csky_ineichen["dhi"]
                df["irradiancebeam"] = csky_ineichen["dni"]
                df["irradiancetotal"] = csky_ineichen["ghi"]
            elif clear_sky_model == 'liujordan':
                """ [1] Campbell, G. S., J. M. Norman (1998) An Introduction to
                Environmental Biophysics. 2nd Ed. New York: Springer.
                [2] Liu, B. Y., R. C. Jordan, (1960). "The interrelationship and
                characteristic distribution of direct, diffuse, and total solar
                radiation".  Solar Energy 4:1-19"""
                #zenith not refraction corrected
                csky_liujordan = irr.liujordan(df['zenith'], self.transmittance,
                                 df['airmassabsolute'], df['extraradiation'])
                df["irradiancediffuse"] = csky_liujordan["dhi"]
                df["irradiancebeam"] = csky_liujordan["dni"]
                df["irradiancetotal"] = csky_liujordan["ghi"]
        if "irradiancediffuseskypoa" in outputs:
            if poa_model == 'klucher1979':
                r"""Klucher's 1979 model determines the diffuse irradiance from the sky
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
                df["irradiancediffuseskypoa"] = irr.klucher(self.surface_zenith, self.surface_azimuth,
                df["irradiancediffuse"], df["irradiancetotal"], df["zenith"], df["azimuth"])
            elif poa_model == 'perez':
                df["irradiancediffuseskypoa"] = irr.perez(self.surface_zenith, self.surface_azimuth,
                df["irradiancediffuse"], df["irradiancebeam"], df["extraradiation"], df["zenith"], df["azimuth"],
                df["airmassrelative"], model=self.diffuse_model, return_components=False)
        if "irradiancediffusegroundpoa" in outputs:
            df["irradiancediffusegroundpoa"] = irr.get_ground_diffuse(self.surface_zenith, df["irradiancetotal"],
                                                                      albedo=self.albedo, surface_type=None)
        if "irradiancediffusepoa" in outputs:
            df["irradiancediffusepoa"] = df["irradiancediffuseskypoa"] + df["irradiancediffusegroundpoa"]
        if "irradiancebeampoa" in outputs:
            df["irradiancebeampoa"] = df.apply(lambda x: x["irradiancebeam"] * max(np.cos(np.radians(x["angleofincidence"])), 0), axis=1)
        if "irradiancetotalpoa" in outputs:
            df["irradiancetotalpoa"] = df["irradiancediffusepoa"] + df["irradiancebeampoa"]
        return df

    @decorators.timer
    def getsuninout(self, utc: pd.DatetimeIndex,
                    freq="min"):  # 26/2/20 date with 00:00:00 otherwise timedelta into next day
        # freq: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        # DEV NOTE 6/5/19: could be improved with minimum according to irradiance value
        # DEV NOTE 1/4/21: timezone not used to avoid conversion problems inside the function
        # DEV NOTE 17/4/21: index where max s.merge_meas_with_spince surface not necessarily south oriented.
        # DEV NOTE 17/4/21 bis: still if two maximum idxmax will only return first one
        # DEV NOTE 1/5/21: already using entire series, could be kept for later
        """ estimate when sun is in or out a tilted surface """
        # Created on 4/4/19 by wsfm

        """
        dt = DataFrame({'start': date_utc.values}, index=date_utc.values)  # day start
        dt["end"] = dt.apply(lambda x: x["start"] + dtt.timedelta(hours=23, minutes=59, seconds=59),
                             axis=1)  # day end
        
        utc = concat([DataFrame(date_range(start=dt.loc[index, "start"],
                                                    end=dt.loc[index, "end"], freq=freq), columns=["utc"]) for index in
                         list(dt.index)], ignore_index=True)
        utc_dti = DatetimeIndex(utc.utc, tz=None)
        """
        dates = [x.date() for x in utc]
        #extending values from start of first day and end of last day
        utc = pd.DatetimeIndex(pd.date_range(start=min(dates),
                                             end=max(dates) + timedelta(days=1), freq=freq), tz=None)
        #remove last row (e.g. next year)
        utc = utc[:-1]

        df = pd.DataFrame({'angleofincidence': self.getsolardataframe(utc, outputs= ["angleofincidence"]).loc[:,"angleofincidence"]
                      , 'date': utc.date,  'datetime': utc}, index=utc)
        
        df_valid = df[(df.angleofincidence > 0) & (df.angleofincidence < 90)]

        df_valid_g = df_valid.groupby(by=['date'], as_index=True).agg({'datetime': ['idxmin', 'idxmax'], 'angleofincidence': ['idxmin', 'min']})

        results = pd.DataFrame({'date': df_valid_g.index,
                                'sun_on': df_valid_g.loc[:, ('datetime', 'idxmin')],
                                'sun_off': df_valid_g.loc[:, ('datetime', 'idxmax')],
                                'sun_max': df_valid_g.loc[:, ('angleofincidence', 'idxmin')],
                                'angleofincidence_min': df_valid_g.loc[:, ('angleofincidence', 'min')]})  # include both min & max in a new df_valid and plot the result

        # results = results.pd.merge(df["date"],how="right",on="date")

        return results

    def getsunrisesettransit(self, datetimeindex: pd.DatetimeIndex):
        # TO BEC CHECKED
        """ calculate sunrise, sunset & transit according to the proposed package"""
        # DEV NOTE 3/4/19: dateimeindex and "times must be localized"
        # Returns pandas.DataFrame
        # index is the same as input `times` argument
        # columns are 'sunrise', 'sunset', and 'transit'
        # https://pvlib-python.readthedocs.io/en/latest/_modules/pvlib/solarposition.html
        # version 1 created on 30/10/17 by fm
        # identify datetime components for the calculation
        # Function information (extracted on 30/10/17):
        # Calculate the sunrise, sunset, and sun transit times using the NREL SPA algorithm described in
        # Reda, I., Andreas, A., 2003. Solar position algorithm for solar radiation applications.
        # Technical report: NREL/TP-560- 34302. Golden, USA, http://www.nrel.gov.
        if self.package == 'pvlib':
            #dti Must be localized to the timezone for latitude and longitude.
            #1/5/21 not clear why giving error on py but not on nb
            datetimeindex_tz = datetimeindex.tz_localize(tz=self.timezone, ambiguous='NaT', nonexistent='NaT')
            transit, sunrise, sunset = spnrel.transit_sunrise_sunset(dates= datetimeindex_tz.to_series().apply(
                lambda x: datetime(x.year, x.month, x.day)),
            lat=self.latitude, lon=self.longitude, delta_t=self.delta_t, numthreads=self.numthreads)
            #DEV NOTE 1/5/21 commented since not used (also zenith
            """spa = spnrel.solar_position(datetimeindex_to_utc(datetimeindex_utc), self.latitude, self.longitude, 
            self.altitude, self.pressure, self.temperature, self.delta_t, self.atmos_refract, numthreads=4)"""
            # DEV NOTE 3/4/19: not clear why dateindex necessary again


            df_out = pd.DataFrame({'date': datetimeindex.date,
                                   'sunrise': sunrise, #sunpos.loc[:, 'sunrise'].values,
                                   'sunset': sunset, #sunpos.loc[:, 'sunset'].values,
                                   'suntransit': transit, # sunpos.loc[:, 'transit'].values,
                                   'datetime': datetimeindex,
                                   'datetime_tz': datetimeindex_tz
                                   #'zenith': spa.loc[:, "zenith"].values  # 23/2/20 added for angleofincidence
                                   },
                                  index=datetimeindex_tz.tz_localize(None)) #to get index in local time
            # index = datetime.date)
        # else:
        #    raise ValueError('times must be localized')
        return df_out

"""from irradiance preliminary nb TBC"""
#fast enough no @decorators.timer

def merge_meas_with_sp(ms_utc:pd.DataFrame, sl:SolarLibrary, dropnadt=False):
    #cs_resolution_s = 300
    dti = pd.DatetimeIndex(ms_utc.index)
    cs = sl.getsolardataframe(dti, outputs=["irradiancetotalpoa"])
    #30/5/21 cs not needed for merge
    #cs['dt'] = cs.index
    #dti = pd.date_range(date,date+pd.Timedelta(1,'D'),freq=str(cs_resolution_s)+"s")[:-1]
    # right otherwise shifted points become NaN
    #keeping same dataframe
    mscs = cs.merge(ms_utc, how='right', left_index=True, right_index=True)
    #mscs = cs.merge(ms_utc.gpoa, how='right', left_index=True, right_index=True)
    #24/5/21 useless dt removed anyway due usince ms_utc.gpoa
    if dropnadt and "dt" in mscs.columns.to_list():
        mscs.dropna(subset=["dt"], axis=0, inplace=True)
    return mscs

#fast no decorators needed
def get_sun_rise_set_transit(dti_tz:pd.DatetimeIndex, sl:SolarLibrary) -> pd.DataFrame:
    #AmbiguousTimeError: 2017-10-29 02:00:00
    srst = pvlib.solarposition.sun_rise_set_transit_spa(dti_tz, sl.latitude, sl.longitude,
                                                        delta_t=sl.delta_t, numthreads=sl.numthreads, how='numpy') #                                                        sl.delta_t=67.0, sl.numthreads=4)
    #DEV NOTE ²2/5/21 fast fix to not modify entire version
    #srst.rename(columns={'sunrise':'sunrise', 'sunset':'sunset', 'transit':'sun_transit'}, inplace=True)
    #keep original name to avoid confusion
    dt_columns = ['sunrise', 'sunset', 'transit']
    #DEV NOTE 2/5/21 maybe not necessary in new version
    for c in dt_columns:
        srst[c+'_h'] = dtop.add_hour(srst[c])
    #back to simple date
    srst.index = srst.index.date
    return srst


"""
PREVIOUS STABILITY REQUIREMENTS IN UNCERTAINTY.PY (FORMER CALIBRATION ANALYSIS)

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

END PREVIOUS STABILITY REQUIREMENTS IN UNCERTAINTY.PY (FORMER CALIBRATION ANALYSIS)
"""

@decorators.timer
#def stabilityflagging(ms_utc
def stabilityflagging(ms_utc: pd.DataFrame, sl: SolarLibrary, steps: List[str], periods: List[str], counts: List[int],
        g_mins:List[int]=[None], aoi_maxs:List[int]=[None], pearson_mins:List[float]=[None], g_uncs:List[int]=[None],
        kcs_cv_maxs:List[int]=[None], kcs_uncs:List[int]=[None], pwr_cv_maxs:List[int]=[None],
        cs_columns:List[str]=[None]) -> (pd.DataFrame, Dict[str, pd.DataFrame]):
    """
    :param ms_utc:
    :param sl:
    :param steps:
    :param periods:
    :param counts:
    :param g_mins:
    :param aoi_maxs:
    :param pearson_mins:
    :param g_uncs:
    :param kcs_cv_maxs:
    :param kcs_uncs:
    :param pwr_cv_maxs:
    :param cs_columns:
    :return:
    """

    #parameter to select if taking first or last values when grouping gpoa and dt
    #21/8/21 TBC still not clear why working for last and not for first
    dtgpoa_agg = "last"
    # 21/8/21 changed last to first at:
    # stp_agg_flt["dt" + prd_sff] = prd_grp['dt'].transform('first')
    #30/5/21 better to not mix with calibration
    #30/5/21 agg methods could be introduced to choose between last or median (better than avg) for the two? phases
    #start with longer steps to use less data when shorter steps
    #filled original dataset NaN with 0 to avoid problem with grouping later
    #TBC why fill na necessary
    ms_grps = dict()
    #initialise df which will progressively filtered
    ms_flt = ms_utc.copy()
    #TBC if necessary
    #if pwr: ms_flt.loc[:, "power"] = ms_flt.loc[:, "power"].fillna(0)
    ms_flt.loc[:, "gpoa"] = ms_flt.loc[:, "gpoa"].fillna(0)
    #if element missed, None provided
    #initialising stp0 to check if output prd used as stp
    prd0 = None
    for stp, prd, cnt, g_min, aoi_max, pearson_min, g_unc, kcs_cv_max, kcs_unc, pwr_cv_max in ittl.zip_longest(
        steps, periods, counts, g_mins, aoi_maxs, pearson_mins, g_uncs, kcs_cv_maxs, kcs_uncs, pwr_cv_maxs,
        fillvalue=None):
        transit = False
        if isinstance(cnt, tuple):
            (cnt_m, cnt_a) = cnt
            cnt = cnt_m + cnt_a
            transit = True
        print(f"{len(ms_flt)} rows. Extracting {stp} data before {prd} grouping of min {cnt} valid values")
        pwr = False
        if "power" in ms_flt.columns.to_list() and pwr_cv_max is not None: pwr = True
        #a df copy for each type of analysis
        #DEV NOT 23/5/21 TBD system to reuse grouping from other analysis
        stp_sff = "_s_" + stp
        #grouping starting from previous filtered dataset to reduce number of data
        #previously timed but very fast no need
        def groupby_timed(df, freq, key):
            return df.groupby([pd.Grouper(freq=freq, key=key)])
        if stp == prd0:
            #skipping validation also since datetime shift of average comparing to spa on last value
            stp_agg_flt = prd_agg_flt
            #replace g_stable_p with d_stable_s TBC if necessary
            (o, n) = [(c, c.replace("g_stable_p", "g_valid_s")) for c in stp_agg.columns.to_list() if "g_stable_p" in c][0]
            stp_agg_flt.rename(columns={o:n}, inplace=True)
            ms_flt["dt" + stp_sff] = stp_agg['dt']
        else:
            if stp == "":
                stp_agg = ms_flt
                ms_flt["dt" + stp_sff] = stp_agg['dt']
            elif stp != "":
                stp_grp = groupby_timed(ms_flt, stp, 'dt')
                #in not grouped dataset reporting reference for merging later
                #old index kept to be used for merging with original df
                ms_flt["dt"+stp_sff] = stp_grp['dt'].transform(dtgpoa_agg)
                # using last value for groupby instead of median
                agg_d: Dict = {'gpoa': dtgpoa_agg, 'dt': dtgpoa_agg}
                if pwr: agg_d["power"] = dtgpoa_agg
                stp_agg = stp_grp.agg(agg_d)
            # dropping NaN before continuing
            # clean nan before grouping# https://github.com/pandas-dev/pandas/issues/9697
            stp_agg.dropna(subset=["dt"], axis=0, inplace=True)
            #index required for meas
            stp_agg.index = stp_agg.dt.rename('index')
            #31/5/21 dt removed not clear why
            stp_agg = merge_meas_with_sp(ms_utc=stp_agg, sl=sl, dropnadt=False)
            #rename to simplify
            stp_agg.rename(columns={"irradiancetotalpoa": "gpoa_cs"}, inplace=True)
            # aoi kept as check
            ms_columns = ["dt", "gpoa", "gpoa_cs", "gpoa_rf", "angleofincidence"]
            if cs_columns is not None: ms_columns += [c for c in cs_columns]
            # DEV NOTE 21/8/21 only most useful columns kept to limit the number of aggregation
            stp_agg.drop(columns=[c for c in stp_agg.columns.to_list() if c not in ms_columns], inplace=True)
            # kcs nan if no gpoa_cs, check why better than 0
            stp_agg["kcs"] = stp_agg.apply(lambda x: x["gpoa"] / x["gpoa_cs"] if x["gpoa_cs"] > 0 else np.NaN, axis=1)
            # filters create different last datetime comparing to original df
            def flag_agg(agg:pd.DataFrame, columns:List[str], parameters:List, column:str):
                #using False instead of 0 to not confuse it with empty values
                #keeping entire dataset flagging to provide complete step/period phase overview later
                #starting from True and iteratively change into False for each condition
                agg.loc[:,column] = True
                for c, p in zip(columns, parameters):
                    #retrieving index for which condition fulfilled
                    if p is not None:
                        # parameters like Pearson defined in agg only if needed thus exclusions cannot be defined before
                        if 'g_min' in c: e = agg.gpoa<p
                        elif 'aoi_max' in c: e = agg.angleofincidence>p
                        elif 'kcs_unc' in c: e = abs(1- agg.kcs)>p
                        elif 'g_unc' in c: e = abs(agg.gpoa-agg.gpoa_cs)> p
                        #TBC for differnet day
                        elif 'cnt_stb' in c:
                            if transit:
                                e = ((agg.mrn_len < cnt_m) | (agg.aft_len < cnt_a) | (agg.srs_len < p))
                            else:
                                e = agg.srs_len < p
                        elif 'prs' in c: e = agg.pearson < p
                        elif 'kcs_cv' in c: e = (abs(agg.kcs_cv) > p) | (agg.kcs_cv == 0)
                        #no need to specify rsl for single check since used only for the specific analysis
                        agg.loc[:, c] = True
                        try:
                            agg.loc[agg.loc[e].index, [column, c]] = False, False
                        except UnboundLocalError:
                            raise UnboundLocalError
                    #true not necessary?
                #copy to avoid warning
                agg_flt = agg.copy()
                agg_flt = agg_flt.loc[agg_flt.loc[:,column],:]
                return agg, agg_flt

            #filter of min g applied to mantain same step if possible
            stp_agg, stp_agg_flt = flag_agg(stp_agg, columns=[c+'_vld' for c in ['g_min','aoi_max','kcs_unc','g_unc']],
            parameters=[g_min, aoi_max, kcs_unc, g_unc], column='g_valid'+ stp_sff)
            """
            stp_agg, stp_agg_flt = flag_agg(stp_agg, columns=[c + '_vld' for c in ['g_min', 'aoi_max', 'kcs_unc', 'g_unc']],
                                            parameters=[g_min, aoi_max, kcs_unc, g_unc],
                                            exclusions=[stp_agg.gpoa < g_min, stp_agg.angleofincidence > aoi_max,
                                                        abs(1 - stp_agg.kcs) > kcs_unc,
                                                        abs(stp_agg.gpoa - stp_agg.gpoa_cs) > g_unc],
                                            column='g_valid' + stp_sff)"""

            #if aoi_max is not None: stp_agg = stp_agg.loc[(stp_agg.angleofincidence < aoi_max)]
            #if kcs_unc is not None: stp_agg = stp_agg.loc[(abs(1- stp_agg.kcs) < kcs_unc)]
            #if g_unc is not None: stp_agg = stp_agg.loc[abs(stp_agg.gpoa - stp_agg.gpoa_cs) <= g_unc]
            #DEV NOTE similar procedure of later to be unified to reduce mistake

            def flag_flt_dz(ms_flt:pd.DataFrame, ms_utc:pd.DataFrame, agg_flt_srs:pd.Series, key_column:str, vld_column:str):
                # flagging of analysed values in filtered dataset to use it also in the native df
                ms_flt = ms_flt.loc[ms_flt.loc[:, key_column].isin(agg_flt_srs), :]
                ms_utc.loc[:, vld_column] = False
                if len(ms_flt) > 0:
                    ms_utc.loc[ms_flt.index, vld_column] = True
                return ms_flt, ms_utc

            ms_flt, ms_utc = flag_flt_dz(ms_flt, ms_utc, stp_agg_flt["dt"], "dt"+stp_sff, "g_valid"+stp_sff)
        if len(stp_agg_flt) == 0:
            print(f"No clear sky data after {stp} step filter")
            #exit iteration
            break
        elif len(stp_agg_flt) > 0:
            if transit:
                # DEV NOTE 24/5/21 logically to be launched when grouping since one per day but performance TBC
                # raise ValueError('Input dates must be at 00:00 UTC')
                dti = dtop.reset_tzi_convert_tze(stp_agg_flt.index.date, sl.timezone, 'utc')
                trn = pvlib.solarposition.sun_rise_set_transit_spa(dti, sl.latitude, sl.longitude, how='numpy',
                                                                   delta_t=sl.delta_t, numthreads=sl.numthreads).loc[:,
                      'transit'].values
                if len(stp_agg_flt) == len(trn):
                    stp_agg_flt["transit"] = dtop.reset_tzi_convert_tze(trn, 'utc', sl.timezone)
                else:
                    raise ValueError
            #grouping filtered data per period
            prd_grp = groupby_timed(stp_agg_flt, prd, 'dt')
            prd_sff = "_p_" + prd
            # in step aggregation apply same grouping for merging later
            # DEV NOTE 21/8/21 changed from last to first
            stp_agg_flt["dt" + prd_sff] = prd_grp['dt'].transform(dtgpoa_agg)

            # defining parameters aggregation. no timer since called multiple times
            def parameters(x, parameters_index):
                #DEV NOTE 25/5/21 passing dictionary to define how performing aggregation or count, e.g. median
                #DEV NOTE 25/5/21 iteration possible also for calibration
                d = {}
                if transit:
                    #DEV NOTE 24/5/21 logically to be launched when grouping since one per day but performance TBC
                    d["aft_len"] = x.loc[x['dt'] >= x['transit'],'gpoa'].count()
                    d["mrn_len"] = x.loc[x['dt'] <= x['transit'],'gpoa'].count()
                #DEV NOTE 24/5/21 average since in standard and correlated to std
                # dt not altered by calibration for merging with stp_agg
                d["dt"] = x["dt"].max()
                #calibration part to be transferred into calibration
                """if clb:
                    xavg0 = x["kcs"].mean()
                    #repeating only once p5 ISO 9847
                    x = x.loc[abs(x['kcs']-xavg0)>kcs_cv_max,:]"""
                d["kcs"] = x["kcs"].mean()
                d['srs_len'] = x['gpoa'].count()
                d["kcs_std"] = x["kcs"].std()
                d["angleofincidence"] = x["angleofincidence"].mean()
                d["gpoa"] = x["gpoa"].mean()
                d["gpoa_cs"] = x["gpoa_cs"].mean()
                if pearson_min is not None:  # pearson only if enough low resolution (cs minute -> prs_d_cs ==0) #https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
                    d['prs_n'] = (x['gpoa'].count() * (x['gpoa'] * x['gpoa_cs']).sum() - x[
                        'gpoa'].sum() * x['gpoa_cs'].sum())
                    d['prs_d'] = (x['gpoa'].count() * (x['gpoa'] * x['gpoa']).sum()) - (
                        x['gpoa'].sum()) ** 2
                    d['prs_d_cs'] = (x['gpoa_cs'].count() * (
                            x['gpoa_cs'] * x['gpoa_cs']).sum()) - (x['gpoa_cs'].sum()) ** 2
                    parameters_index = parameters_index.extend(['prs_n', 'prs_d', 'prs_d_cs'])
                if pwr:
                    d["power"] = x["power"].mean()
                    d["power_std"] = x["power"].std()
                    parameters_index = parameters_index.extend(['power', 'power_std'])

                return pd.Series(d, index=parameters_index)
            # 6/1/20 could apply be replaced with faster transform?
            @decorators.timer
            def get_parameters_timed(prd_grp, parameters, parameters_index):
                return prd_grp.apply(parameters, parameters_index)

            parameters_index = ['kcs', 'kcs_std', 'srs_len', 'angleofincidence', 'gpoa', 'gpoa_cs', 'dt']

            prd_agg = get_parameters_timed(prd_grp, parameters, parameters_index)
            # clean nan for easier readness
            prd_agg.dropna(how='any', subset=["dt"], inplace=True)
            prd_agg.loc[:, "kcs_cv"] = prd_agg.apply(lambda x: (x["kcs_std"] / x["kcs"]) if x["kcs"] != 0 else np.nan,
                                                   axis=1)
            if pearson_min is not None:
                prd_agg.loc[:, "pearson"] = prd_agg.apply(lambda x: (x["prs_n"]/((x["prs_d"]**0.5)/(x["prs_d_cs"] ** 0.5))) 
                if (x["prs_d"] > 0 and x["prs_d_cs"] > 0) else np.nan, axis=1)
            prd_agg.index = prd_agg.dt

            def gnn(labels, vars):
                #get not nones
                l: List[str] = []
                for k, v in zip(labels, vars):
                    if v is not None:
                        l += [k]
                return ','.join([k for k in l])

            print(f"Validation done with {gnn(['g_min', 'aoi_max', 'kcs_unc', 'g_unc'],[g_min, aoi_max, kcs_unc, g_unc])}"+
            f". Filtering for {gnn(['values count','pearson','kcs std','power std'],[cnt, pearson_min, kcs_cv_max, pwr_cv_max])}")

            #indipendent group for power before filtering for g
            if pwr:
                prd_agg_pwr = prd_agg.copy()
                prd_agg_pwr = prd_agg_pwr[abs(prd_agg_pwr.power_cv) < pwr_cv_max]
                ms_flt, ms_utc = flag_flt_dz(ms_flt, ms_utc, prd_agg_pwr.index, "p_stable"+prd_sff)
            #filtering for g
            prd_agg ,prd_agg_flt = flag_agg(prd_agg, columns=['cnt_stb', 'prs_stb', 'kcs_cv_stb'],
                               parameters=[cnt, pearson_min, kcs_cv_max],
                               column='g_stable'+prd_sff)
            """
            prd_agg, prd_agg_flt = flag_agg(prd_agg, columns=['cnt_stb', 'prs_stb', 'kcs_cv_stb'],
                                            parameters=[cnt, pearson_min, kcs_cv_max],
                                            exclusions=[prd_agg.srs_len < cnt, prd_agg.pearson < pearson_min,
                                                        (abs(prd_agg.kcs_cv) > kcs_cv_max) | (prd_agg.kcs_cv == 0)],
                                            column='g_stable' + prd_sff)"""
            # first merging filtered step with periods analysis having dt prd_sff label
            #using index and filtering columns to avoid conflict of dt column and index having same name
            stp_agg_flt = stp_agg_flt.merge(
                right=prd_agg.loc[:, [c for c in prd_agg.columns.to_list() if c not in stp_agg_flt.columns.to_list()]],
                left_on=["dt" + prd_sff], right_index=True, how='left', sort=True)
            #then merging with complete stp_agg for control
            stp_agg = stp_agg.merge(right=stp_agg_flt.loc[:, [c for c in stp_agg_flt.columns.to_list() if
                                                              c not in stp_agg.columns.to_list()]],
                                    left_index=True,
                                    right_index=True,
                                    how='left',
                                    sort=True)
            # recording analysis for this step
            ms_grps["s" + stp + "_p" + prd + "_c" + str(cnt)] = stp_agg
            # now filtering for stability
            # flagging
            # TBC if stp == prd0:
            ms_flt_tmp, ms_utc = flag_flt_dz(ms_flt, ms_utc, stp_agg_flt.loc[stp_agg_flt.loc[:,'g_stable'+prd_sff],"dt"],
                                       "dt"+stp_sff, "g_stable"+prd_sff)
            if stp == "":
                ms_utc = ms_utc.merge(right=stp_agg.loc[:, "gpoa_cs"],
                                      left_index=True,
                                      right_index=True,
                                      how='left',
                                      sort=True)

            if len(ms_flt_tmp) == 0:
                print(f"No clear sky data after {prd} period filter")
                break
            elif len(ms_flt_tmp) > 0:
                ms_flt = ms_flt_tmp
            # recording steps if to be chained after
            prd0 = prd
    if len(ms_flt_tmp) > 0:
        print(f"Returning {len(ms_flt)} rows in filtered dataset.")

    #returing last valid ms_flt
    return ms_utc, ms_flt, ms_grps

"""FUNCTIONS ORIGINALLY IN FIRST CLEAR SKY IDENTIFICATION, TO BE TESTED"""

def get_sunpath_tz(dti:pd.DatetimeIndex, sl:SolarLibrary, freq:str="min", suninoutonly=True) -> pd.DataFrame:
    #DEV NOTE 9/6/21 originally in cs identification, to be replaced with utc time or new function
    #AmbiguousTimeError: 2017-10-29 02:00:00
    #replace with nan since solar path only for visualisation
    dti_tz = dti.tz_localize(tz=sl.timezone, ambiguous='NaT')
    dti_utc = dti_tz.tz_convert('utc')
    #DEV NOTE 17/4/21 removed since double convert extend index !!
    #dti_none = dti_utc.tz_convert(None)
    if suninoutonly:
        cs = sl.getsuninout(utc= dti_utc, freq= freq)
        dt_columns = ['sun_on', 'sun_off', 'sun_max']
    elif suninoutonly == False:
        cs = sl.getsolardataframe(dti_utc, outputs=["irradiancetotalpoa"])
        cs['dt'] = cs.index
        dt_columns = ['dt']
    for c in dt_columns:
        dti_sl_utc = pd.DatetimeIndex(cs.loc[:,c].values, tz='utc', ambiguous='NaT')
        dti_sl_tz = dti_sl_utc.tz_convert(tz=sl.timezone)
        #reconvert into original df
        cs.index = dti_sl_tz.tz_localize(None)
    return cs

@decorators.timer
def get_clear_sky_candidates(values_datetime_indexed_tz:pd.Series, df_dly_dc:pd.DataFrame, resolution_s=300,
                             candidatesp=0.05) -> pd.DataFrame:
    #important df_dly_dc hours coherent with values_datetime_index time zone
    #keeping only days where values inside sunpath
    df_dly_dc = df_dly_dc.dropna(subset=["first_hour"])
    df_dly_dc = df_dly_dc.loc[(df_dly_dc.first_hour<df_dly_dc.sun_on_h)&(df_dly_dc.last_hour>df_dly_dc.sun_off_h)]
    #dropping not necessary columns
    ctodrop = ["null","after_last_hours","before_first_hours","first_hour","last_hour","before_first_hours_cml"]
    df_dly_dc.drop(columns=[c for c in df_dly_dc.columns.to_list() if c in ctodrop], inplace=True)
    df_dly_mm = dtop.get_min_max_by_date(values_datetime_indexed=values_datetime_indexed_tz, resolution_s=resolution_s)
    df_dly_mm.rename(columns={'hour_value_2d_max_morning':'hour_first_ray', 'hour_value_2d_max_afternoon': 'hour_last_ray'},
                inplace=True)
    #dropping not necessary columns
    df_dly_mm.drop(columns=["len","null","hour_min","hour_max"], inplace=True)
    #merging disconnection (sun path info) with function  analysis
    df_dly_cs = df_dly_mm.merge(df_dly_dc, how='inner', left_index=True, right_index=True)
    # calculating delay of parameters from theoretical sun path
    df_dly_cs["delay_first_ray"] = df_dly_cs["hour_first_ray"] - df_dly_cs["sun_on_h"]
    df_dly_cs["delay_max_ray"] = df_dly_cs["hour_value_max"] - df_dly_cs["sun_max_h"]
    df_dly_cs["delay_last_ray"] = df_dly_cs["hour_last_ray"] - df_dly_cs["sun_off_h"]
    df_dly_cs["delays_product"] =  df_dly_cs.apply(lambda x: x["delay_first_ray"]*x["delay_max_ray"]*x["delay_last_ray"], axis=1)
    #analyse distribution of delays_product
    dn = Cdf.from_seq(df_dly_cs.delays_product, normalize=True)
    #extract minimum positive and maximum negative delay
    dppmin = min(dn[dn.index>0].index)
    dpnmax = max(dn[dn.index<0].index)
    pmin = dn[dppmin] if abs(dppmin)<abs(dpnmax) else dn[dpnmax]
    df_dly_cs_dsc = df_dly_cs.describe(percentiles=[pmin-candidatesp/2, pmin+candidatesp/2])
    pmini = str(round((pmin-candidatesp/2)*100,1))+'%'
    pmaxi = str(round((pmin+candidatesp/2)*100,1))+'%'
    # selecting candidates identifying dates of clear sky candidates
    delays_product_min = df_dly_cs_dsc.loc[pmini,"delays_product"]
    delays_product_max = df_dly_cs_dsc.loc[pmaxi,"delays_product"]
    cs_candidates = df_dly_cs.loc[(df_dly_cs.delays_product>delays_product_min ) & (df_dly_cs.delays_product<delays_product_max),:]
    return cs_candidates

#WINDOWS TESTING
LATITUDE = 52.7616
LONGITUDE = -1.2406 #may have to be changed
ALTITUDE = 79
SURFACE_ZENITH = 34
SURFACE_AZIMUTH = 180

#LOCAL_FOLDER = "wsfm"
#LOCAL_FOLDER = "nothing"
#pvlib vs dataframe
_WINDOWS_TESTING = "comparison_CREST_SNL"
_WINDOWS_TESTING = "dataframe"
_WINDOWS_TESTING = "stability_csv"
_WINDOWS_TESTING = "calibration"
_WINDOWS_TESTING = ""

if _WINDOWS_TESTING == 'calibration':
    FILEPATHS, FOLDERPATHS, CHANNELS, FILTERS = {}, {}, {}, {}
    COLUMNS: Dict[str, Dict] = {}
    FILTER_TYPES = ["series_count", "series_minutes", "readings_count", "beam_min", "diffuse_min", "diffuse_max",
                    "dfraction_max", "aoi_max", "wind_max"]

    EURAC_RESOLUTION_SECONDS_2017 = 60
    FILTERS["EURAC"] = dict(zip(FILTER_TYPES, [999, 20, 20, 700, 10, 150, 0.15, 70, 2]))
    DATETIME_FORMAT_EURAC = "%d/%m/%Y %H:%M"
    # (46° 27’ 28’’ N, 11° 19’ 43’’ E, 247 meters above sea level
    SL_EURAC = SolarLibrary(latitude=46.464010, longitude=11.330354, altitude=247,
                                 surface_zenith=0, surface_azimuth=180, timezone="Europe/Rome")


    FOLDERPATHS["outdoor EURAC 2017"] = (  # path to eurac folder
        "C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/170530-170614_eurac_calibration/")
    FILEPATHS["outdoor kz11 e24 2017"] = join(FOLDERPATHS["outdoor EURAC 2017"],
                                              "calibration_calculations_Ch4_fm_181122.csv")
    COLUMNS_CALIBRATION = ["vr_ji", "vf_ji", "dt", "aoi", "t_ambient", "diffuse", "gpoa", "wind_speed", "azimuth"]
    COLUMNS_OUTDOOR_2017 = ["PYR_REF_Ch3 raw", "CH raw", "date", "Zenith angle (ETR) SolPos", "T_ambient scaled",
                            "CMP11_diffuse scaled", "CMP11_global_horiz scaled", "Gill_wind_speed scaled",
                            "Azimuth angle SolPos"]

    resolution = EURAC_RESOLUTION_SECONDS_2017

    clb = "outdoor kz11 e24 2017"
    CHANNELS["outdoor kz11 e24 2017"] = "PYR_TEST_Ch4 raw"
    file_columns = [CHANNELS[clb] if x == "CH raw" else x for x in COLUMNS_OUTDOOR_2017]
    COLUMNS[clb] = dict(zip(file_columns, COLUMNS_CALIBRATION))
    filepath = FILEPATHS[clb]
    filters = FILTERS["EURAC"]

    usecols, columns = [k for k in COLUMNS[clb].keys()], COLUMNS[clb]
    # 280520 modify name depending on file
    df = pd.read_csv(filepath, header=0, skiprows=0, usecols=usecols)
    df.rename(columns=columns, inplace=True)

    df["diffuse_fraction"] = df["diffuse"] / df["gpoa"]
    df["beam"] = df["gpoa"] - df["diffuse"]  #
    df.loc[:, "dt"] = df.loc[:, "dt"].apply(lambda x: pd.to_datetime(x, format=DATETIME_FORMAT_EURAC))
    dti = pd.DatetimeIndex(df.loc[:, "dt"], tz=SL_EURAC.timezone).tz_convert(tz='utc')

    df.loc[:, "dt"] = dti
    df.index = dti

    df_flt = df[(df["beam"] >= filters["beam_min"]) &
                (df["diffuse"] > filters["diffuse_min"]) & (df["diffuse"] < filters["diffuse_max"]) &
                (df["diffuse_fraction"] < filters["dfraction_max"]) &
                (df["aoi"] <= filters["aoi_max"])].copy()

    itr = 2
    # DEV NOTE 24/5/21 be careful keep labels different (or change structure) to avoid overwriting
    steps = ["", "20min"]
    # 20 instead of 21 to allow spot values per minute
    periods = ["20min", "m"]
    # minimum 15 series
    counts = [20, 15]
    # light requirement on kcs error
    # more than kcs better uncertainty since more balanced during the day and less affected by shift in theory
    # kcs_range=[1-KCS_DEV_LMT,1+KCS_DEV_LMT] #in original algorithm
    kcs_uncs = [None] * itr
    # Pearson for initial one since no other strong req but maybe could applied also to later analysis
    pearson_mins = [None] * itr
    # CV analysis only on series not for initial filtering
    kcs_cv_maxs = [None, KCS_CV_MAX]
    # initially trying with no requirements on g_uncs. pwr_cv_maxs not needed
    # no valid data for G_UNC 30 or 60, trying with KCS although more stricter for lower elevation
    # g_uncs = [None]*3
    # pwr_cv_maxs

    # separately defining the ones not needing comments
    (ms_tz, sl, g_mins, aoi_maxs) = (df_flt, SL_EURAC, [G_MIN] * itr, [AOI_MAX] * itr)

    ms, ms_flt, ms_grps = stabilityflagging(ms_utc=df_flt, sl=sl, steps=steps, periods=periods, counts=counts,
                                                 kcs_uncs=kcs_uncs, pearson_mins=pearson_mins,
                                                 kcs_cv_maxs=kcs_cv_maxs, g_mins=g_mins, aoi_maxs=aoi_maxs)

if _WINDOWS_TESTING == 'stability_csv':
    PV_FOLDER_C = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/Skytron_Data_TEST/"
    PV_FILE_C = r"devicedata_0921.csv"
    gpoa_com = pd.read_csv(filepath_or_buffer=PV_FOLDER_C + PV_FILE_C, delimiter=";", skip_blank_lines=True, header=11,
                           nrows=597792)
    dfc_g, duplicates_c = dtop.clean_df(df=gpoa_com, dt_format='%Y-%m-%d %H:%M:%S')
    gpoa_com_info = pd.read_csv(filepath_or_buffer=PV_FOLDER_C + 'pv_com_info.csv', delimiter=",", header=0,
                                index_col=0)
    # Module position is retrieved through [digimap](#https://digimap.edina.ac.uk/)
    LATITUDE = 52.7616
    # negative longitude based on solar convention
    LONGITUDE = - (1.2406)
    ALTITUDE = 79
    PRESSURE = 101325.
    TEMPERATURE = 12
    DELTA_T = 67.0
    ATMOS_REFRACT = 0.5667
    TRANSMITTANCE = 0.5
    G_MIN = 20
    SURFACE_ZENITH = 34
    SURFACE_AZIMUTH = 180
    NUMTHREADS = 4
    # tbc where used
    delta_t = DELTA_T
    observer_longitude = LONGITUDE
    observer_latitude = LATITUDE
    observer_elevation = ALTITUDE
    local_pressure = PRESSURE
    local_temp = TEMPERATURE
    atmos_refract = ATMOS_REFRACT
    numthreads = NUMTHREADS
    surface_zenith_com = gpoa_com_info.loc["surface_tilt", "value"]  # %%
    altitude_com = gpoa_com_info.loc["altitude", "value"]  #
    latitude_com = gpoa_com_info.loc["latitude", "value"]  #
    longitude_com = gpoa_com_info.loc["longitude", "value"]  #
    sl_g = SolarLibrary(latitude=latitude_com, longitude=longitude_com, altitude=altitude_com,
                        temperature=TEMPERATURE, delta_t=DELTA_T, pressure=PRESSURE,
                        surface_zenith=surface_zenith_com, surface_azimuth=SURFACE_AZIMUTH,
                        atmos_refract=ATMOS_REFRACT, transmittance=TRANSMITTANCE,
                        timezone="Europe/London")
    print(sl_g.clear_sky_model)
    print(sl_g.poa_model)
    #transforming dt since used for grouping after
    #no clear data after h step if (wrong) tz used, using utc
    dfc_g.dt = dtop.reset_tzi_convert_tze(dfc_g.dt, tzs='utc', tze=sl_g.timezone)
    #index also because used in merge
    dfc_g.index = dfc_g.dt.rename('index')


    itr = 2
    #DEV NOTE 24/5/21 be careful keep labels different (or change structure) to avoid overwriting
    steps = ["h", "", "20min"]
    # 20 instead of 21 to allow spot values per minute
    periods = ["d", "20min", "d"]
    # minimum 15 series
    counts = [(3,3), 20, 15]
    # light requirement on kcs error
    # more than kcs better uncertainty since more balanced during the day and less affected by shift in theory
    kcs_uncs = [KCS_UNC] * itr
    # Pearson for initial one since no other strong req but maybe could applied also to later analysis
    pearson_mins = [PEARSON_MIN]
    # CV analysis only on series not for initial filtering
    kcs_cv_maxs = [None] + [KCS_CV_MAX] *2
    # initially trying with no requirements on g_uncs. pwr_cv_maxs not needed
    #no valid data for G_UNC 30 or 60, trying with KCS although more stricter for lower elevation
    # g_uncs = [None]*3
    # pwr_cv_maxs

    # separately defining the ones not needing comments
    (ms_tz, sl, g_mins, aoi_maxs) = (dfc_g, sl_g, [G_MIN] * itr, [AOI_MAX] * itr)

    ms, ms_flt, ms_grps  = stabilityflagging(ms_tz=ms_tz, sl=sl, steps=steps, periods=periods, counts=counts,
                                                       kcs_uncs=kcs_uncs, pearson_mins=pearson_mins,
                                                       kcs_cv_maxs=kcs_cv_maxs, g_mins=g_mins, aoi_maxs=aoi_maxs)

    
    ms.to_csv(join(CSVS_FOLDER, "Glamorgan_ms" + ".csv"))
    ms_flt.to_csv(join(CSVS_FOLDER, "Glamorgan_cs" + ".csv"))
    for s, p, c in zip(steps, periods, counts):
        if isinstance(c, tuple):
            (m, a) = c
            c = m+a
        key = "s"+s+"_p"+p+"_c"+str(c)
        print("generating " + key + " file")
        mscs_tmp = ms_grps[key]
        mscs_tmp.to_csv(join(CSVS_FOLDER, "Glamorgan_" + key + ".csv"))

    ms.to_csv(join(CSVS_FOLDER, "Glamorgan_" + ".csv"))

    


#testing solardataframe
if _WINDOWS_TESTING == 'dataframe':
#if LOCAL_FOLDER in getcwd():
    LATITUDE = 52.7616
    LONGITUDE = -1.2406 #may have to be changed
    ALTITUDE = 79
    SURFACE_ZENITH = 34
    SURFACE_AZIMUTH = 180

    sl = SolarLibrary(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE,
                      surface_zenith=SURFACE_ZENITH, surface_azimuth=SURFACE_AZIMUTH,
                      temperature=TEMPERATURE, delta_t=DELTA_T, pressure=PRESSURE,
                      atmos_refract=ATMOS_REFRACT, transmittance=TRANSMITTANCE)


    df_gpoa, errors = dtop.csv_read_datetime_format(filepath=join(ISC_IRRADIANCE_FILES_FOLDER,'chpoa_y20.csv'),
                                         datetime_column="date_trunc")

    if len(errors) == 0:
        df_gpoa.drop(columns='tmstamp', inplace=True)
        times = sl.getsuninout(utc= df_gpoa.date_trunc, freq= "H")
        print(times)
        #testing difference between clear sky models
        sl.clear_sky_model = 'liujordan'
        # klucher1979
        print(sl.clear_sky_model, sl.poa_model)
        df_solar = sl.getsolardataframe(pd.DatetimeIndex(df_gpoa.date_trunc), outputs=["irradiancetotalpoa"])
        # df_solar = sl.getsolardataframe(DatetimeIndex(times), outputs=["irradiancetotalpoa"])
        cs_lj_k = df_solar.loc[:, "irradiancetotalpoa"]
        # %%
        sl.poa_model = 'klucher1979'
        print(sl.clear_sky_model, sl.poa_model)
        df_solar = sl.getsolardataframe(pd.DatetimeIndex(df_gpoa.date_trunc), outputs=["irradiancetotalpoa"], solardataframe=df_solar)
        cs_lj_p = df_solar.loc[:, "irradiancetotalpoa"]
        #sl.clear_sky_model = "ineichenperez"
        sl.clear_sky_model = 'ineichenperez'
        print(sl.clear_sky_model, sl.poa_model)
        df_solar = sl.getsolardataframe(pd.DatetimeIndex(df_gpoa.date_trunc), outputs=["irradiancetotalpoa"], solardataframe=df_solar)
                                        #solardataframe=df_solar)
        cs_ip_p = df_solar.loc[:, "irradiancetotalpoa"]
        sl.poa_model = 'perez'
        print(sl.clear_sky_model, sl.poa_model)
        df_solar = sl.getsolardataframe(pd.DatetimeIndex(df_gpoa.date_trunc), outputs=["irradiancetotalpoa"], solardataframe=df_solar)
                                        #solardataframe=df_solar)
        cs_ip_k = df_solar.loc[:, "irradiancetotalpoa"]
        # extracting date
        # index before adding new columns
        df_gpoa.index = pd.DatetimeIndex(df_gpoa.date_trunc)
        df_gpoa.loc[:, 'lj_k'] = cs_lj_k
        df_gpoa.loc[:, 'lj_p'] = cs_lj_p
        df_gpoa.loc[:, 'ip_p'] = cs_ip_p
        df_gpoa.loc[:, 'ip_k'] = cs_ip_k

        print(df_gpoa.corr(method='pearson'))

    else:
        print("Import failed")


if _WINDOWS_TESTING == 'comparison_CREST_SNL':
#if LOCAL_FOLDER in getcwd():
    LATITUDE = 52.7616
    #DEV NOTE 2/4/21: if opposite orientation inverted aoi per eot close to zero
    LONGITUDE = -1.2406
    ALTITUDE = 79
    SURFACE_ZENITH = 34
    SURFACE_AZIMUTH = 180

    FILEPATH = join(ASSETS_FOLDER,'crest_nrel_spa.csv') #filepath one day
    FILEPATH = join(dirname(getcwd())+r"/experimental/"+'crest_spa_2020.csv')

    crest_spa, errors = dtop.csv_read_datetime_format(filepath=FILEPATH,
                                         datetime_column="SPA_Datetime")

    sl = SolarLibrary(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE,
                      surface_zenith=SURFACE_ZENITH, surface_azimuth=SURFACE_AZIMUTH,
                      temperature=TEMPERATURE, delta_t=DELTA_T, pressure=PRESSURE,
                      atmos_refract=ATMOS_REFRACT, transmittance=TRANSMITTANCE)

    outputs = ["apparentzenith","zenith","azimuth","equation_of_time", "elevation", "apparentelevation",
               "angleofincidence","irradiancetotalpoa"]

    spa_outputs = sl.getsolardataframe(datetimeindex_utc=pd.DatetimeIndex(crest_spa["spa_datetime"]), outputs=outputs)

    crest_spa.spa_datetime = pd.to_datetime(crest_spa.spa_datetime)

    comparison = pd.merge(crest_spa, spa_outputs, left_on="spa_datetime", right_on="datetime", suffixes=("_crest", "_nrel"))

    comparison["date"] = pd.DatetimeIndex(comparison.spa_datetime).date

    c_aoi = comparison[[c for c in comparison.columns.to_list() if (('aoi' in c or 'angleofincidence' in c or c == 'date' or
                                                                     c == 'equation_of_time' or c == 'datetime')
                                                                    and '45' not in c)]]

    c_aoi_g = c_aoi[(c_aoi.angleofincidence < 90) & (c_aoi.angleofincidence > 0)].groupby("date").agg(['first','last'])

    #gpoa_cs_dly = sl.getsuninout(utc= crest_spa.spa_datetime, freq= "H")

    #comparison.to_csv('spa_crest_vs_nrel.csv')
