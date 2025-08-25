# Pycharm notes
# Bookmarks: navigate Shift + F11
# shift F11 to navigate bookmarks
_WINDOWS_TESTING = "calibration 2017 flt none"
_WINDOWS_TESTING = "gum"
_WINDOWS_TESTING = "gum limit aoi"
_WINDOWS_TESTING = "monte carlo limit"
_WINDOWS_TESTING = "monte carlo negative"
_WINDOWS_TESTING = "monte carlo negative calibration single"
_WINDOWS_TESTING = "monte carlo negative calibration limit "
_WINDOWS_TESTING = ""

_TEST_UNC = False
# 6/11/21 checking if negative into positive
_TEST_UNC_MC = False

import clear_sky as cs
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from os.path import join
# for timer
import datetime as dt
# for testing
# import decorators as ds
import copy

UNCERTAINTY_METHOD="isoa_astm"

#STANDARD UNCERTAINTY FOR KZ UNCERTAINTY_METHOD
#Estimated standard uncertainties for KZ, see "21sheet" of "Pyranometer_Uncertainties_180410_v2"
#DEV NOTE 5/11/18: sources for different su to be reported

#max number of decimals to consider in MC
MC_NORMAL_DECIMALS = 3
#percentile to be considered in MC, for p90 is 1.645

COVERAGE_FACTOR = 1.645

#higher percentile
PERCENTILE_MAX = 95.45
PERCENTILE_MIN = 100-95.45

# uncertainty source for pyranometer iso9060, astmG213, iec61724-1
# G213 − 17 consider the ISO acceptance values as already expanded uncertainty (which makes sense)
# if data already provided as standard uncertainty (energy yield simulation) no need for divisor (=1)
# total offset split into parts in case different distribution for offset A to be applied
ZERO_OFFSET_TOTAL={'uncertainty':'zero_offset_total','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_a':10,'acceptance_b':21,'acceptance_c':41}
# offset A considered as negative in 2015 Uncertainty evaluation of measurements with pyranometers and pyrheliometers Konings, Jörgen Habte, Aron
# hereby considered positive otherwise more complex to assess other sources offset
# 29/8/21 added unit which affect calculation
ZERO_OFFSET_A={'uncertainty':'zero_offset_a','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_a':7,'acceptance_b':15,'acceptance_c':30}
ZERO_OFFSET_B={'uncertainty':'zero_offset_b','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_a':2,'acceptance_b':4,'acceptance_c':8}
# other sources estimated as difference
ZERO_OFFSET_OTHER={'uncertainty':'zero_offset_other','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_a':1,'acceptance_b':2,'acceptance_c':3}
NON_STABILITY={'uncertainty':'non_stability','id':'isoc1','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':0.8,'acceptance_b':1.5,'acceptance_c':3}
NON_LINEARITY={'uncertainty':'non_linearity','id':'isoc2','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':0.5,'acceptance_b':1,'acceptance_c':3}
DIRECTIONAL_RESPONSE={'uncertainty':'directional_response','id':'isoc3','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_a':10,'acceptance_b':20,'acceptance_c':30}
SPECTRAL_ERROR={'uncertainty':'spectral_error','id':'isoc4','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':0.5,'acceptance_b':1,'acceptance_c':5}
TEMPERATURE_RESPONSE={'uncertainty':'temperature_response','id':'isoc5','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':1,'acceptance_b':2,'acceptance_c':4}
TILT_RESPONSE={'uncertainty':'tilt_response','id':'isoc6','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':0.5,'acceptance_b':2,'acceptance_c':5}
SIGNAL_PROCESSING={'uncertainty':'signal_processing','id':'isoc7','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_a':2,'acceptance_b':5,'acceptance_c':10}
#21/5/20 signal processing should be f(voltage) not f(irradiance)
#30/9/23 divisor of 2 used to obtain the standard uncertainty, normal from calibration certificate
CALIBRATION={'uncertainty':'calibration','id':'manufacturer','parameter':'sensitivity','distribution':'normal','shape':'symmetric','divisor': 2,'unit':'%','acceptance_a': 1.4,'acceptance_b': 1.4,'acceptance_c': 1.4}
#CALIBRATION={'uncertainty':'calibration','id':'astm1','parameter':'sensitivity','distribution':'normal','shape':'symmetric','divisor':2,'unit':'%','acceptance_a':5.62,'acceptance_b':5.62,'acceptance_c':5.62}
MAINTENANCE={'uncertainty':'maintenance','id':'astm8','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':0.3,'acceptance_b':0.3,'acceptance_c':0.3}
# 28/9/23 alignment error in degree transformed in radiant after in get_uncertainty_gum
# 30/9/23 Class A & B updated for final standard, class C in TC now remove in standard
ALIGNMENT_ZENITH={'uncertainty':'alignment_zenith','id':'iec6b','parameter':'zenith','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'degree','acceptance_a':0.5,'acceptance_b':1,'acceptance_c':1} #'acceptance_c':2}
ALIGNMENT_AZIMUTH={'uncertainty':'alignment_azimuth','id':'iec6a','parameter':'azimuth','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'degree','acceptance_a':1,'acceptance_b':2,'acceptance_c':2} #'acceptance_c':4}

UNCERTAINTIES=[ZERO_OFFSET_TOTAL, ZERO_OFFSET_A, ZERO_OFFSET_B, ZERO_OFFSET_OTHER,
               NON_STABILITY,NON_LINEARITY,DIRECTIONAL_RESPONSE,
               SPECTRAL_ERROR,TEMPERATURE_RESPONSE,TILT_RESPONSE,SIGNAL_PROCESSING,
               CALIBRATION,MAINTENANCE,ALIGNMENT_ZENITH,ALIGNMENT_AZIMUTH]

UNCERTAINTIES_PYRANOMETERS=pd.DataFrame(columns=ZERO_OFFSET_TOTAL.keys())

# creation of the complete irradiance uncertainty dataframe
for value in list(UNCERTAINTIES):
    # 12/8/23 Transposition needed
    UNCERTAINTIES_PYRANOMETERS=pd.concat([UNCERTAINTIES_PYRANOMETERS,pd.Series(value).to_frame().T],ignore_index=True)
UNCERTAINTIES_PYRANOMETERS.index=UNCERTAINTIES_PYRANOMETERS.loc[:,"uncertainty"]

# CMP21 is a class A
# Non linearity & non stability from Instruction Manual - CM21 Precision Pyranometer
# Calibration from certificate
UNCERTAINTIES_CMP21={"non_stability":0.5,"non_linearity":0.2,"calibration":1.4}


#13/8/23 adding pyrheliometer ISO 9060:2018

ZERO_OFFSET_A={'uncertainty':'zero_offset_a','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_aa':0.1,'acceptance_a':1,'acceptance_b':3,'acceptance_c':6}
ZERO_OFFSET_TOTAL={'uncertainty':'zero_offset_total','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_aa':0.2,'acceptance_a':2,'acceptance_b':4,'acceptance_c':7}
NON_STABILITY={'uncertainty':'non_stability','id':'isoc1','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_aa':0.01,'acceptance_a':0.5,'acceptance_b':1,'acceptance_c':2}
NON_LINEARITY={'uncertainty':'non_linearity','id':'isoc2','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_aa':0.01,'acceptance_a':0.2,'acceptance_b':0.5,'acceptance_c':2}
SPECTRAL_ERROR={'uncertainty':'spectral_error','id':'isoc4','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_aa':0.01,'acceptance_a':0.2,'acceptance_b':1,'acceptance_c':2}
TEMPERATURE_RESPONSE={'uncertainty':'temperature_response','id':'isoc5','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_aa':0.01,'acceptance_a':0.5,'acceptance_b':1,'acceptance_c':5}
TILT_RESPONSE={'uncertainty':'tilt_response','id':'isoc6','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_aa':0.01,'acceptance_a':0.2,'acceptance_b':0.5,'acceptance_c':2}
SIGNAL_PROCESSING={'uncertainty':'signal_processing','id':'isoc7','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'Wm-2','acceptance_aa':0.1,'acceptance_a':1,'acceptance_b':5,'acceptance_c':10}
CALIBRATION={'uncertainty':'calibration','id':'manufacturer','parameter':'sensitivity','distribution':'normal','shape':'symmetric','divisor': 2,'unit':'%','acceptance_a': 1.1,'acceptance_b': 1.1,'acceptance_c': 1.1}
#21/5/20 signal processing should be f(voltage) not f(irradiance)
# Equivalent of ASTM G213 for pyranometer not searched/found
# CALIBRATION={'uncertainty':'calibration','id':'astm1','parameter':'sensitivity','distribution':'normal','shape':'symmetric','divisor':2,'unit':'%','acceptance_a':5.62,'acceptance_b':5.62,'acceptance_c':5.62}
# MAINTENANCE={'uncertainty':'maintenance','id':'astm8','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':0.3,'acceptance_b':0.3,'acceptance_c':0.3}
# Equivalent not found for pyrheliometers 
# ALIGNMENT_ZENITH={'uncertainty':'alignment_zenith','id':'iec6b','parameter':'zenith','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':1,'acceptance_b':1.5,'acceptance_c':2}
#ALIGNMENT_AZIMUTH={'uncertainty':'alignment_azimuth','id':'iec6a','parameter':'azimuth','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'unit':'%','acceptance_a':2,'acceptance_b':3,'acceptance_c':4}


#including zero_offset a even if total only used
UNCERTAINTIES=[ZERO_OFFSET_A, ZERO_OFFSET_TOTAL, 
               NON_STABILITY,NON_LINEARITY,
               SPECTRAL_ERROR,TEMPERATURE_RESPONSE,TILT_RESPONSE,SIGNAL_PROCESSING,
               CALIBRATION]

UNCERTAINTIES_PYRHELIOMETERS=pd.DataFrame(columns=ZERO_OFFSET_TOTAL.keys())

# creation of the complete irradiance uncertainty dataframe
for value in list(UNCERTAINTIES):
    # 12/8/23 Transposition needed
    UNCERTAINTIES_PYRHELIOMETERS=pd.concat([UNCERTAINTIES_PYRHELIOMETERS,pd.Series(value).to_frame().T],ignore_index=True)
UNCERTAINTIES_PYRHELIOMETERS.index=UNCERTAINTIES_PYRHELIOMETERS.loc[:,"uncertainty"]

# Instruction Manual - CM21 Precision Pyranometer
UNCERTAINTIES_CHP1={"calibration":1.1}

"""
"""
#Point values for the calibration of KZ CMP21 090318
TEMPERATURE_DEVIATION_KZ_CMP21 = pd.Series(
[-0.29,0.04,0.11,0.01,0.00,0.06,0.03,0.54],
index=  [-20,-10,0,10,20,30,40,50])
# 12/9/21 DEV NOTE relative cosine error % but absolute would be more in line with new iso ?
# 12/9/21 index was previously inverted
# cable pointing north 180/-180
# west is 90, east is -90 in line with (e.g. North = 0, South=180 East = 90, West = 270)
# 14/8/23 standards limits outside characteristation boundaries ?
DIRECTIONAL_DEVIATION_KZ_CMP21 = pd.DataFrame(
{"-180": [1.23725,0.49,0.18375,0.21],
 "-90": [3.20675,1.27,0.47625,0.23],
 "0": [2.02,0.8,0.3,0.07],
 "90": [-0.47975,-0.19,-0.07125,-0.05],
 "180": [1.23725,0.49,0.18375,0.21]},
index = ["80","70","60","40"])
# 8/23 conversion
DIRECTIONAL_DEVIATION_KZ_CMP21 = pd.DataFrame(
{"0": [1.23725,0.49,0.18375,0.21],
 "90": [3.20675,1.27,0.47625,0.23],
 "180": [2.02,0.8,0.3,0.07],
 "270": [-0.47975,-0.19,-0.07125,-0.05],
 "360": [1.23725,0.49,0.18375,0.21]},
index = ["80","70","60","40"])




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

CALIBRATION_METHOD = "EURAC 9847 1a"
CALIBRATION_READINGS = 21
ITERATIONS_MAXIMUM = 100
#absolute deviation from calibration factor
DEVIATION_MAX = 0.02 #10/1/20 modified from 2

class Irradiance_Uncertainty:
    # DEV NOTE 19/9/21 sensitivity could be omitted if measurement uncertainty not estimated or only percentages
    # defining uncertainty framework
    def __init__(self, sensitivity: float, coverage_factor=COVERAGE_FACTOR, method=UNCERTAINTY_METHOD,
                 radiometer_type="pyranometer"):  # =500):
        """
        :param sensitivity float: 
        :param coverage_factor:
        :param method: rather pyranometer class
        """
        # 22/8/23 added to be called 
        self.directional_deviation = DIRECTIONAL_DEVIATION_KZ

        # sensitivity in V/Wm-2
        if radiometer_type == "pyranometer": self.uncertainties_df = UNCERTAINTIES_PYRANOMETERS
        elif radiometer_type == "pyrheliometer": self.uncertainties_df = UNCERTAINTIES_PYRHELIOMETERS
        else:
            raise Exception("radiometer_type must be pyranometer or pyrheliometer")

        if "isoa" in method:
            self.uncertainties_df.loc[:, "acceptance"] = self.uncertainties_df.loc[:, "acceptance_a"]
        elif "isob" in method:
            self.uncertainties_df.loc[:, "acceptance"] = self.uncertainties_df.loc[:, "acceptance_b"]
        elif "isoc" in method:
            self.uncertainties_df.loc[:, "acceptance"] = self.uncertainties_df.loc[:, "acceptance_c"]
        # if KZ 11 or 21, uncertainty defined from datasheet.
        # Note 12/10/22: somewhere KZ employee declared not so much difference ?
        if ("cmp21" in method) or ("cmp11" in method):
            for k, v in UNCERTAINTIES_CMP21.items(): self.uncertainties_df.loc[k, "acceptance"] = v
        elif "chp1" in method:
            for k, v in UNCERTAINTIES_CHP1.items(): self.uncertainties_df.loc[k, "acceptance"] = v


        #defining standard uncertainty from acceptance
        self.method = method
        # DEV NOTE 16/11/18: to be renamed as test, irradiance new 
        self.sensitivity = sensitivity
        self.coverage_factor = coverage_factor
        #13/5/20 to be checked
        """elif method == "mrt2":
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
            self.su_directional = SU_DIRECTIONAL_KZ"""

    def interpolate_directional_error(self, sys_az:pd.Series, 
                                          sys_aoi:pd.Series,
                                          step=10,
                                          method="linear",
                                          inplace=False,
                                          decimals=3) -> pd.DataFrame:
        # compatible with version 2 of directional deviation
        # 24/9/23 need complete range for proper backfill
        """def get_s10_range(s:pd.Series, step=step):
            smin = int(round(min(s)/step,0)*step)
            smax = int(round(max(s)/step,0)*step) + step
            return range(smin, smax, step)
        
        dd = pd.DataFrame(np.nan,
        columns= get_s10_range(s=sys_az, step=step),
        index = get_s10_range(s=sys_aoi, step=step))"""

        dd = pd.DataFrame(np.nan,
        columns= range(0, 360, step),
        index = range(0, 90, step))

        dd0 = self.directional_deviation
        # 23/8/23 necessary since old system with string instead of integer
        dd0.rename(
        columns=dict(zip(dd0.columns, [int(c) for c in dd0.columns])), 
        index=dict(zip(dd0.index, [int(i) for i in dd0.index])),
        inplace=True
        )
        dd.update(dd0)
        # first aoi 
        dd= dd.interpolate(method=method, axis=0)
        dd= dd.interpolate(method=method, axis=1)
        dd= dd.round(decimals)
        # backfill since characterisation not available for low aoi
        dd.bfill(inplace=True)
        if inplace: self.directional_deviation = dd
        #return dd for checking but not necessary
        return dd
    
    def add_directional_error(self, df:pd.DataFrame,
                                  sys_aoi_clm: str, 
                                  sys_az_clm:str,
                                  step=10,
                                  method="linear") -> pd.DataFrame:
        dd = self.interpolate_directional_error(
                                          sys_az=df[sys_az_clm], 
                                          sys_aoi=df[sys_aoi_clm],
                                          step=step,
                                          method=method,
                                          inplace=False,
                                          decimals=3)
        dd["sys_aoi_step"] = dd.index
        # unpivot
        ddu = pd.melt(dd, id_vars="sys_aoi_step", value_vars=dd.columns, 
            var_name="sys_az_step", value_name="directional_error")
        df["sys_aoi_step"] = df[sys_aoi_clm].apply(lambda x: int(round(x/step,0)*step))
        df["sys_az_step"] = df[sys_az_clm].apply(lambda x: int(round(x/step,0)*step))
        # 26/9/23 datetime as column to reintroduce as index after
        df["datetime"] = df.index
        df_ddu = df.merge(ddu, 
        left_on=["sys_aoi_step", "sys_az_step"], 
        right_on=["sys_aoi_step", "sys_az_step"]
        )
        df_ddu.set_index("datetime", inplace=True, drop=True)
        df_ddu.sort_index(inplace=True) 
        return df_ddu
        

    def get_deviation_on_temperature(self, temperature=None):
        # DEV NOTE 31/10/18: only Series and not series cases covered here
        # mrt approach using characterisation information as precise deviation
        temp_dev_tab = TEMPERATURE_DEVIATION_KZ
        x = list(temp_dev_tab.index.values)
        f = list(temp_dev_tab.values)
        temperature_reference = TEMPERATURE_REFERENCE_KZ

        # uncertainty due to temperature, estimated interpolation from KZ data
        def temp_deviation(temperature):
            # if (temperature < min(x))or(temperature>max(x)):
            #    raise ValueError("temperature out of the function bounds")
            #    return None
            # elif (temperature >= min(x)) or (temperature<=max(x)):
            if temperature <= min(x):
                temp_deviation = min(f)
            elif temperature >= max(x):
                temp_deviation = max(f)
            elif (temperature > min(x)) & (temperature < max(x)):
                temp_deviation = np.interp(temperature, x, f)
            return temp_deviation

        # regardless of model used, return single value or series depending on istance             
        if isinstance(temperature, pd.Series) == False:
            if temperature == None:
                temperature = temperature_reference
            deviation_on_temperature = temp_deviation(temperature)
        if (isinstance(temperature, pd.Series) == True) and (np.all(temperature) is not None):
            # deviation_on_temperature = temperature.apply(lambda x: temp_deviation(x))
            deviation_on_temperature = temperature.transform(lambda x: temp_deviation(x))
            # DEV NOTE 5/11/18: old working version
            # deviation = pd.Series(index=temperature.index)
            # _t_tmp = temperature[temperature <= -20]
            # deviation.loc[_t_tmp.index] = np.full(len(_t_tmp),-0.29)
            # _t_tmp = temperature[temperature >= 50]
            # deviation.loc[_t_tmp.index] = np.full(len(_t_tmp),0.54)
            # _t_tmp = temperature[(temperature > -20) & (temperature < 50)]
            # deviation.loc[_t_tmp.index]=temperature.loc[_t_tmp.index].apply(lambda t: np.interp(t, x, f))
        return deviation_on_temperature

    def get_deviation_on_direction(self, sys_az=None, sys_aoi=None):
        # directional response in 
        # DEV NOTE 5/11/18: check if interpolation could be done with
        # x,y = np.meshgrid(az,aoi,sparse=True)
        # data = f(*np.meshgrid(az,aoi,indexing='ij',sparse=True))
        # from scipy.interpolate import RegularGridInterpolator
        # my_interpolating_function = RegularGridInterpolator((az,aoi,dev),data)

        # uncertainty due to angle of incidence and azimuth of the radiometer, estimated interpolation from KZ data
        dir_dev_tab = self.directional_deviation
        az = list(dir_dev_tab.columns.values.astype(int))
        aoi = list(dir_dev_tab.index.values.astype(int))
        azimuth_reference = AZIMUTH_REFERENCE_KZ
        angle_of_incidence_reference = ANGLE_OF_INCIDENCE_REFERENCE_KZ

        # KZ dependency for directional response
        # mrt approach assuming manufacturer characterisation gave precise deviation with negligible uncertainty
        
        def directional_deviation(sys_az, sys_aoi):
            if (sys_az is None or sys_az is np.nan) or (sys_aoi is None or sys_aoi is np.nan):
                raise ValueError(f"Not valid az {sys_az} and/or aoi {sys_aoi}")  
            #initialise for check
            # 12/8/23 already taking care of aoi outside min and max
            aoi_start , aoi_end = None, None
            if sys_aoi <= min(aoi):
                aoi_start = min(aoi)
                aoi_end = min(aoi)
            elif sys_aoi >= max(aoi):
                aoi_start = max(aoi)
                aoi_end = max(aoi)
            elif (sys_aoi > min(aoi)) and (sys_aoi < max(aoi)):
                aoi_start = max([a for a in aoi if a <= sys_aoi])
                aoi_end = min([a for a in aoi if a >= sys_aoi])
            """else:
            print(f"{sys_aoi} not in  {aoi}")"""
            # if not outside range, iterate to find interpolation interval
            """for index, value in enumerate(aoi):
                if value == max(aoi):
                    break
                elif value <= sys_aoi:
                    # '=' condition if exact value
                    aoi_start = value
                    aoi_end = aoi[index + 1]"""
            # defining related azimuth range
            if sys_az <= min(az):
                az_start = min(az)
                az_end = min(az)
            elif sys_az >= max(az):
                az_start = max(az)
                az_end = max(az)
            elif (sys_az > min(az)) and (sys_az < max(az)):
                az_start = max([a for a in az if a <= sys_az])
                az_end = min([a for a in az if a >= sys_az])
            """elif (sys_az > min(az)) and (sys_az < max(az)):
                # if not outside range, iterate to find interpolation interval
                for index, value in enumerate(az):
                    if value == max(az):
                        break
                    elif value <= sys_az:
                        # '=' condition if exact value
                        az_start = value
                        az_end = az[index + 1]"""
            # combined interpolation
            az0_aoi = [aoi_start, aoi_end]
            #except UnboundLocalError:

            # retrieving values from kz results
            try:
                az0_dev = [dir_dev_tab.loc[str(aoi_start), str(az_start)],
                           dir_dev_tab.loc[str(aoi_end), str(az_start)]]
            except KeyError:
                print(r"Error in definition of az0_dev")
                print(f"system az {sys_az}, aoi {sys_aoi}")
                print(str(aoi_start), str(az_start), str(aoi_end), str(az_start))
            
            az0_val = np.interp(sys_aoi, az0_aoi, az0_dev)
            az1_aoi = [aoi_start, aoi_end]
            az1_dev = [dir_dev_tab.loc[str(aoi_start), str(az_end)],
                       dir_dev_tab.loc[str(aoi_end), str(az_end)]]
            az1_val = np.interp(sys_aoi, az1_aoi, az1_dev)
            az_range = [az_start, az_end]
            dev_range = [az0_val, az1_val]
            return np.interp(sys_az, az_range, dev_range)

        if (isinstance(sys_az, pd.Series) == False) & (isinstance(sys_aoi, pd.Series) == False):
            # DEV NOTE 29/11/18: temp sol due to '<' not supported between instances of 'NoneType' and 'int'
            if sys_aoi is None:
                deviation = None
            elif sys_aoi is not None:
                if sys_aoi <= 90 and sys_az <= 360:
                    # print(f"sys_aoi {sys_aoi}<= 90 and sys_az {sys_az} <= 360")
                    deviation = directional_deviation(sys_az, sys_aoi)
                elif sys_aoi > 90:
                    raise ValueError("angle of incidence  {sys_aoi} higher than 90 degree")        
                    deviation = None
        
        elif (isinstance(sys_az, pd.Series) == True) or (isinstance(sys_aoi, pd.Series) == True):
            if (isinstance(sys_aoi, pd.Series) == False):
                sys_aoi = pd.Series(np.full(shape=len(sys_az), fill_value=angle_of_incidence_reference))
            if (isinstance(sys_az, pd.Series) == False):
                sys_az = pd.Series(np.full(shape=len(sys_aoi), fill_value=azimuth_reference))
            if (np.any(sys_aoi) is None) or (np.any(sys_az) is None) or (np.any(sys_aoi) is np.nan) or (np.any(sys_az) is np.nan):
                deviation = None
            elif ((np.all(sys_aoi) is not None) and (np.all(sys_aoi) is not np.nan)) or ((np.all(sys_az) is not None) and (np.all(sys_az) is not np.nan)):
                # 12/8/23 using 360 since directional deviation converted for pvlib convention
                if np.all(sys_aoi) <= 90 and abs(np.all(sys_az)) <= 360:
                    # print(f"np.all(sys_aoi) {sys_az} <= 90 and abs(np.all(sys_az)) {sys_aoi} <= 360")
                    df_inputs = pd.DataFrame({"sys_az": sys_az, "sys_aoi": sys_aoi})
                    # returning deviation as series to keep same index                   
                    deviation = df_inputs.apply(lambda x: directional_deviation(x["sys_az"], x["sys_aoi"]), axis=1)
                elif np.any(sys_aoi) > 90 and abs(np.all(sys_az)) > 360:
                    deviation = None
        
        return deviation
    
    def get_uncertainty(self, irradiance: Union[pd.Series, float], temperature=None, azimuth=None,
                        angle_of_incidence=None, coverage_factor=COVERAGE_FACTOR,
                        uncs_mc_cffs:dict=None, test=_TEST_UNC, asymmetric_pair=False,
                        directional_error=False):
        # all the elemens in the azimuth and AOI series need to be <> None to calculate deviation
        """
        :param irradiance:
        :param temperature:
        :param azimuth:
        :param angle_of_incidence:
        :param coverage_factor:
        :param uncs_mc_cffs:
        :param test:
        :param asymmetric_pair: only for GUM not needed for MC
        :param directional: 
        :return:
        """
        #16/10/21 adding float
        #26/9/21 DEV NOTES mixing both mc and not, coudl be splitted ?
        #26/9/21 asymmetric_pair to be tested
        #19/9/21 TBE keeping self as storage for uncertainty parameters
        #29/8/21 returning deviation for mc instead of combined uncertainty should be separate
        #31/8/21 alignement part in get_uncertainty_gum to be integrated
        #defining starting parameters value different than changing mc
        p0 = {}
        # if no parameters provided at all use stored self.uncertainties_df
        if uncs_mc_cffs is None:
            uncertainties_lst = self.uncertainties_df.uncertainty.to_list()
        else:
            uncertainties_lst = [u for u in uncs_mc_cffs.keys()]
        parameters_lst = [u for u in np.unique(self.uncertainties_df.loc[uncertainties_lst, "parameter"])]
        # checking if mc parameters provided
        # it is not so useful to combine mc with gum but could be eventually coded just to check effect of asymmetric
        # removing temperature and directional response if provided
        if np.all(temperature) is not None:
            # 16/10/21 TBC if transformation into series required
            p0["sensitivity"] = self.sensitivity * (100 + self.get_deviation_on_temperature(temperature)) / 100
            uncertainties_lst.remove("temperature_response")
        else: p0["sensitivity"] = self.sensitivity
        #29/8/21 possibility of calculating only when valid ?



        if directional_error and ( 
        (((np.all(azimuth) is not None) and (np.all(azimuth) is not np.nan)) or
           ((np.all(angle_of_incidence) is not None) or (np.all(angle_of_incidence) is not np.nan)))):
            # check if deviation not None (e.g. aoi <90 )
            # print(f"deviation_directional = self.get_deviation_on_direction AZ {azimuth} AOI{angle_of_incidence}")
            deviation_directional = self.get_deviation_on_direction(azimuth, angle_of_incidence)
            if deviation_directional is not None:
                irradiance = irradiance * (100 + deviation_directional) / 100
                # directional response different than tilt
                uncertainties_lst.remove("directional_response")
        #else:
        #    print("Skip directional error since not valid azimuth and/or aoi")
        
        # initialise p as p0 for mc too
        # p0 defined regardless if in self.uncertainties_df.parameter.to_list() since may be used for coefficients later
        p0["irradiance"] = irradiance
        # 17/10/21 test if not None
        if p0["sensitivity"] is not None:  p0["voltage"] = p0["sensitivity"] * p0["irradiance"]
        if uncs_mc_cffs is not None and np.all([u for u in uncs_mc_cffs.values()]) is not None:
            mc_ck = True
            #deviation functions
            f = {}
            irradiance_mc = irradiance
            # 26/9/21 TBC if proper approach
            # 24/10/21
            f["irradiance"] = lambda irr, mc_dev: irr + mc_dev
            f["voltage"] = lambda irr, mc_dev: irr + mc_dev
            # reduction (negative mc) of sensitivity -> irradiance could be higher
            f["sensitivity"] = lambda irr, mc_dev: irr * (p0["sensitivity"]) / (p0["sensitivity"]+mc_dev)
            """f["irradiance"] = lambda irr, mc_dev: irr * mc_dev
            f["voltage"] = lambda irr, mc_dev: irr * mc_dev
            f["sensitivity"] = lambda irr, mc_dev: irr / mc_dev"""

        else:
            mc_ck = False
            # DEV 26/9/21 to be adapted for difference types of uncertainties
            # only if not mc:
            # calculate sensitivity coefficient for gum
            # c also needed to check which parameter to be considered for mc
            c = {}
            # 29/8/21 in ISO voltage already provided as irradiance [Wm-2] acceptance
            # coefficients only for GUM
            if isinstance(irradiance, pd.Series):
                if "voltage" in parameters_lst: c["voltage"] = p0["irradiance"].apply(lambda x: 1 / p0["sensitivity"])
                if "sensitivity" in parameters_lst: c["sensitivity"] = p0["voltage"].apply(lambda x: - x / (p0["sensitivity"] ** 2))
                if "irradiance" in parameters_lst: c["irradiance"] = p0["irradiance"].apply(lambda x: 1)
            else:
                if "voltage" in parameters_lst: c["voltage"] = 1 / p0["sensitivity"]
                if "sensitivity" in parameters_lst: c["sensitivity"] = -(p0["voltage"]) / (p0["sensitivity"] ** 2)
                if "irradiance" in parameters_lst: c["irradiance"] = 1

        # defining parameters
        comb_stnd_uncr = 0
        c_s_u_positive_only = 0
        c_s_u_negative_only = 0
        for u in uncertainties_lst:
            parameter, acceptance, unit, divisor, shape = \
                self.uncertainties_df.loc[u,["parameter","acceptance","unit","divisor","shape"]]
            #calculate only if parameter dependency is defined
            try:
                if unit == "%": acceptance = p0[parameter] * acceptance / 100
            except KeyError:
                print(f"Missing {parameter} requested for {u}")
                raise KeyError
            # if using mc 
            if mc_ck and parameter in [k for k in f.keys()]:
                # transforming in % if not already
                #if unit != "%": acceptance = acceptance / p0[parameter] * 100
                #24/10/21 regardless of distribution still sum (if negative -> negative already)
                mc_dev = uncs_mc_cffs[u] * acceptance
                """if shape == "symmetric":
                    #mc_dev = (1 + uncs_mc_cffs[u] * acceptance / 100)
                    mc_dev = uncs_mc_cffs[u] * acceptance
                elif shape == "negative":
                    #mc_dev = (1 - uncs_mc_cffs[u] * acceptance / 100)
                
                elif shape == "positive":
                    #mc_dev = (1 + uncs_mc_cffs[u] * acceptance / 100)"""

                if test:
                    self.uncertainties_df.loc[u, "mc"] = uncs_mc_cffs[u]
                    self.uncertainties_df.loc[u, "irr_dev"] = mc_dev
                irradiance_mc = f[parameter](irradiance_mc, mc_dev)
                """if parameter == "irradiance" or parameter == "voltage":
                    irradiance_mc = irradiance_mc * mc_dev
                elif parameter == "sensitivity":
                    irradiance_mc = irradiance_mc / mc_dev"""
            elif mc_ck == False and parameter in [k for k in c.keys()]:
                # for GUM percentage transformed into absolute due to different coefficient
                if unit == "%": acceptance = p0[parameter] * acceptance / 100
                acceptance = acceptance / divisor
                # DEALING WITH ASYMMETRICAL DISTRIBUTION
                # "Uncertainty evaluation of measurements with pyranometers and pyrheliometers"
                # Konings, Jörgen   Habte, Aron
                # but other approaches could be possible
                # 19/9/21 not considering that symmetric should be only in one direction
                if shape != "symmetric":
                    acceptance = acceptance / 2
                if asymmetric_pair:
                    if acceptance > 0: c_s_u_positive_only += c[parameter] ** 2 * (acceptance ** 2)
                    elif acceptance < 0: c_s_u_negative_only += c[parameter] ** 2 * (acceptance ** 2)
                elif asymmetric_pair == False:
                    comb_stnd_uncr += c[parameter] ** 2 * (acceptance ** 2)
                #if test: print(f"{}")
                # calculate combined standard uncertainty
        if mc_ck:
            # 24/10/21
            uncr = irradiance_mc - irradiance
        elif mc_ck == False:
            if asymmetric_pair:
                uncr = ((comb_stnd_uncr+c_s_u_negative_only)** 0.5 * coverage_factor,
                        (comb_stnd_uncr+c_s_u_positive_only)** 0.5 * coverage_factor)
            elif asymmetric_pair == False:
                # keep different steps for better understanding of parameters involved
                comb_stnd_uncr = comb_stnd_uncr ** 0.5
                # calculate combined expanded uncertainty
                exp_uncr = comb_stnd_uncr * coverage_factor
                uncr = exp_uncr
        return uncr


    def get_uncertainty_gum(self, irradiance_total, diffuse_fraction=0, zenith=None, azimuth=None, surface_zenith=None,
                            surface_azimuth=None, temperature=None):
        # not none temperature for certificate uncertainty
        # 100% total as conservative estimation of beam (minimum of 80%) for irradiance magnitude uncertainty
        # DEV NOTE 28/8/21 see also uncertainty/uncertainty xlsx and ppt as various examples
        # DEV NOTE 29/8/21 TO BE REVIEWED !! Acceptance units varies, % and Wm-2 (to be adapted)

        #coefficient irradiance total
        cit = 1
        uit2 = 0
        #coefficient voltage
        cv = 1 / self.sensitivity
        uv2 = 0
        #coefficient sensitivity
        voltage = self.sensitivity * irradiance_total
        cs = -(voltage) / (self.sensitivity ** 2)
        us2 = 0
        for row in list(self.uncertainties_df.index):
            # In ASTM G213 − 17 the acceptance values are interpreted as extended uncertainty
            # distribution are rectangular or normal, divided by divisor to get comparable standard uncertainty
            u = self.uncertainties_df.loc[row, "acceptance"] / self.uncertainties_df.loc[row, "divisor"]
            p = self.uncertainties_df.loc[row, "parameter"]
            # 27/9/23 adding (not) percentage, not necessary for zenith/azimuth
            if p == "irradiance":
                uit2 += u ** 2
            elif p == "voltage":
                uv2 += u ** 2
            elif p == "sensitivity":
                us2 += u ** 2
       
        #28/9/23: conversion needed only for uncertaintities linked to sensitivity parameter, only one using percentage
        us2 = us2 * (self.sensitivity / 100) ** 2  # 13/5/20 adapt to percentage values
        comb_stnd_uncr2 = (uv2 * cv ** 2 + us2 * cs ** 2 + uit2 * cit ** 2)
        # 12/8/23 uncertainty for alignment ?

        # 10/9/23 both zenith and azimuth for alignment, no sense calculating only one
        # "Accuracy of ground surface broadband shortwave radiation monitoring"
        if (zenith is not None) & (azimuth is not None) & (surface_zenith is not None) & (surface_azimuth is not None) & (
            ("alignment_zenith" in self.uncertainties_df.index) & ("alignment_azimuth" in self.uncertainties_df.index)):
            # transformation into radiant
            z = zenith / 180 * np.pi
            a = azimuth / 180 * np.pi
            z0 = surface_zenith / 180 * np.pi
            a0 = surface_azimuth / 180 * np.pi          
            uz = self.uncertainties_df.loc["alignment_zenith", "acceptance"] / self.uncertainties_df.loc[
                "alignment_zenith", "divisor"] * np.pi / 180
            ua = self.uncertainties_df.loc["alignment_azimuth", "acceptance"] / self.uncertainties_df.loc[
                "alignment_azimuth", "divisor"] * np.pi / 180
            # definition previous 10/9/23, not clear why different, put inside thesis
            # 30/9/23 (1 - diffuse_fraction) not needed if irradiance_total specified as beam
            
            ca0 = irradiance_total * (
                        np.cos(a0) * np.sin(a) * np.sin(z) * np.sin(z0) - np.cos(a) * np.sin(a0) * np.sin(z) * np.sin(
                    z0))
            cz0 = irradiance_total * (
                        np.cos(a) * np.cos(a0) * np.cos(z0) * np.sin(z) - np.cos(z) * np.sin(z0) + np.sin(a) * np.sin(
                    a0) * np.cos(z0) * np.sin(z))
            # 10/9/23 definition according to "Accuracy of ground surface broadband shortwave radiation monitoring"
            # assuming same convention used for zenith and azimuth both for solar position and surface
            ca0 = irradiance_total * (
                - np.sin(z0) * np.cos(a0) * np.sin(z) * np.sin(a)
                + np.sin(z0) * np.sin(a0) * np.sin(z) * np.cos(a))
            
            cz0 = irradiance_total * (
                + np.sin(z0) * np.cos(a0) * np.cos(z) * np.cos(a)
                + np.sin(z0) * np.sin(a0) * np.cos(z) * np.sin(a)
                - np.sin(z) * np.cos(z0))
            
            comb_stnd_uncr2 += ca0 ** 2 * ua ** 2 + cz0 ** 2 * uz ** 2

        exp_uncr = (comb_stnd_uncr2 ** 0.5) * self.coverage_factor
        return exp_uncr

    def get_uncertainty(self, irradiance: Union[pd.Series, float], temperature=None, azimuth=None,
                        angle_of_incidence=None, coverage_factor=COVERAGE_FACTOR,
                        uncs_mc_cffs:dict=None, test=_TEST_UNC, asymmetric_pair=False,
                        directional_error=False):
        # all the elemens in the azimuth and AOI series need to be <> None to calculate deviation
        """
        :param irradiance:
        :param temperature:
        :param azimuth:
        :param angle_of_incidence:
        :param coverage_factor:
        :param uncs_mc_cffs:
        :param test:
        :param asymmetric_pair: only for GUM not needed for MC
        :param directional: 
        :return:
        """
        #16/10/21 adding float
        #26/9/21 DEV NOTES mixing both mc and not, coudl be splitted ?
        #26/9/21 asymmetric_pair to be tested
        #19/9/21 TBE keeping self as storage for uncertainty parameters
        #29/8/21 returning deviation for mc instead of combined uncertainty should be separate
        #31/8/21 alignement part in get_uncertainty_gum to be integrated
        #defining starting parameters value different than changing mc
        p0 = {}
        # if no parameters provided at all use stored self.uncertainties_df
        if uncs_mc_cffs is None:
            uncertainties_lst = self.uncertainties_df.uncertainty.to_list()
        else:
            uncertainties_lst = [u for u in uncs_mc_cffs.keys()]
        parameters_lst = [u for u in np.unique(self.uncertainties_df.loc[uncertainties_lst, "parameter"])]
        # checking if mc parameters provided
        # it is not so useful to combine mc with gum but could be eventually coded just to check effect of asymmetric
        # removing temperature and directional response if provided
        if np.all(temperature) is not None:
            # 16/10/21 TBC if transformation into series required
            p0["sensitivity"] = self.sensitivity * (100 + self.get_deviation_on_temperature(temperature)) / 100
            uncertainties_lst.remove("temperature_response")
        else: p0["sensitivity"] = self.sensitivity
        #29/8/21 possibility of calculating only when valid ?



        if directional_error and ( 
        (((np.all(azimuth) is not None) and (np.all(azimuth) is not np.nan)) or
           ((np.all(angle_of_incidence) is not None) or (np.all(angle_of_incidence) is not np.nan)))):
            # check if deviation not None (e.g. aoi <90 )
            # print(f"deviation_directional = self.get_deviation_on_direction AZ {azimuth} AOI{angle_of_incidence}")
            deviation_directional = self.get_deviation_on_direction(azimuth, angle_of_incidence)
            if deviation_directional is not None:
                irradiance = irradiance * (100 + deviation_directional) / 100
                # directional response different than tilt
                uncertainties_lst.remove("directional_response")
        #else:
        #    print("Skip directional error since not valid azimuth and/or aoi")
        
        # initialise p as p0 for mc too
        # p0 defined regardless if in self.uncertainties_df.parameter.to_list() since may be used for coefficients later
        p0["irradiance"] = irradiance
        # 17/10/21 test if not None
        if p0["sensitivity"] is not None:  p0["voltage"] = p0["sensitivity"] * p0["irradiance"]
        if uncs_mc_cffs is not None and np.all([u for u in uncs_mc_cffs.values()]) is not None:
            mc_ck = True
            #deviation functions
            f = {}
            irradiance_mc = irradiance
            # 26/9/21 TBC if proper approach
            # 24/10/21
            f["irradiance"] = lambda irr, mc_dev: irr + mc_dev
            f["voltage"] = lambda irr, mc_dev: irr + mc_dev
            # reduction (negative mc) of sensitivity -> irradiance could be higher
            f["sensitivity"] = lambda irr, mc_dev: irr * (p0["sensitivity"]) / (p0["sensitivity"]+mc_dev)
            """f["irradiance"] = lambda irr, mc_dev: irr * mc_dev
            f["voltage"] = lambda irr, mc_dev: irr * mc_dev
            f["sensitivity"] = lambda irr, mc_dev: irr / mc_dev"""

        else:
            mc_ck = False
            # DEV 26/9/21 to be adapted for difference types of uncertainties
            # only if not mc:
            # calculate sensitivity coefficient for gum
            # c also needed to check which parameter to be considered for mc
            c = {}
            # 29/8/21 in ISO voltage already provided as irradiance [Wm-2] acceptance
            # coefficients only for GUM
            if isinstance(irradiance, pd.Series):
                if "voltage" in parameters_lst: c["voltage"] = p0["irradiance"].apply(lambda x: 1 / p0["sensitivity"])
                if "sensitivity" in parameters_lst: c["sensitivity"] = p0["voltage"].apply(lambda x: - x / (p0["sensitivity"] ** 2))
                if "irradiance" in parameters_lst: c["irradiance"] = p0["irradiance"].apply(lambda x: 1)
            else:
                if "voltage" in parameters_lst: c["voltage"] = 1 / p0["sensitivity"]
                if "sensitivity" in parameters_lst: c["sensitivity"] = -(p0["voltage"]) / (p0["sensitivity"] ** 2)
                if "irradiance" in parameters_lst: c["irradiance"] = 1

        # defining parameters
        comb_stnd_uncr = 0
        c_s_u_positive_only = 0
        c_s_u_negative_only = 0
        for u in uncertainties_lst:
            parameter, acceptance, unit, divisor, shape = \
                self.uncertainties_df.loc[u,["parameter","acceptance","unit","divisor","shape"]]
            #calculate only if parameter dependency is defined
            try:
                if unit == "%": acceptance = p0[parameter] * acceptance / 100
            except KeyError:
                print(f"Missing {parameter} requested for {u}")
                raise KeyError
            # if using mc 
            if mc_ck and parameter in [k for k in f.keys()]:
                # transforming in % if not already
                #if unit != "%": acceptance = acceptance / p0[parameter] * 100
                #24/10/21 regardless of distribution still sum (if negative -> negative already)
                mc_dev = uncs_mc_cffs[u] * acceptance
                """if shape == "symmetric":
                    #mc_dev = (1 + uncs_mc_cffs[u] * acceptance / 100)
                    mc_dev = uncs_mc_cffs[u] * acceptance
                elif shape == "negative":
                    #mc_dev = (1 - uncs_mc_cffs[u] * acceptance / 100)
                
                elif shape == "positive":
                    #mc_dev = (1 + uncs_mc_cffs[u] * acceptance / 100)"""

                if test:
                    self.uncertainties_df.loc[u, "mc"] = uncs_mc_cffs[u]
                    self.uncertainties_df.loc[u, "irr_dev"] = mc_dev
                irradiance_mc = f[parameter](irradiance_mc, mc_dev)
                """if parameter == "irradiance" or parameter == "voltage":
                    irradiance_mc = irradiance_mc * mc_dev
                elif parameter == "sensitivity":
                    irradiance_mc = irradiance_mc / mc_dev"""
            elif mc_ck == False and parameter in [k for k in c.keys()]:
                # for GUM percentage transformed into absolute due to different coefficient
                if unit == "%": acceptance = p0[parameter] * acceptance / 100
                acceptance = acceptance / divisor
                # DEALING WITH ASYMMETRICAL DISTRIBUTION
                # "Uncertainty evaluation of measurements with pyranometers and pyrheliometers"
                # Konings, Jörgen   Habte, Aron
                # but other approaches could be possible
                # 19/9/21 not considering that symmetric should be only in one direction
                if shape != "symmetric":
                    acceptance = acceptance / 2
                if asymmetric_pair:
                    if acceptance > 0: c_s_u_positive_only += c[parameter] ** 2 * (acceptance ** 2)
                    elif acceptance < 0: c_s_u_negative_only += c[parameter] ** 2 * (acceptance ** 2)
                elif asymmetric_pair == False:
                    comb_stnd_uncr += c[parameter] ** 2 * (acceptance ** 2)
                #if test: print(f"{}")
                # calculate combined standard uncertainty
        if mc_ck:
            # 24/10/21
            uncr = irradiance_mc - irradiance
        elif mc_ck == False:
            if asymmetric_pair:
                uncr = ((comb_stnd_uncr+c_s_u_negative_only)** 0.5 * coverage_factor,
                        (comb_stnd_uncr+c_s_u_positive_only)** 0.5 * coverage_factor)
            elif asymmetric_pair == False:
                # keep different steps for better understanding of parameters involved
                comb_stnd_uncr = comb_stnd_uncr ** 0.5
                # calculate combined expanded uncertainty
                exp_uncr = comb_stnd_uncr * coverage_factor
                uncr = exp_uncr
        return uncr


    def get_uncertainty_OLD(self,
                        irradiance,
                        temperature=None,
                        azimuth=None,
                        angle_of_incidence=None,
                        coverage_factor=COVERAGE_FACTOR,
                        mc_coeff=None):

        # DEV NOTE 29/8/21 TO BE REVIEWED !! Acceptance units varies, % and Wm-2 (to be adapted)

        #retrieving all su even but temperature and directional response (which may be used or not)
        su_datalogger = self.uncertainties_df.loc["signal_processing", "su"]
        su_calibration = self.uncertainties_df.loc["calibration", "su"]
        su_nonstability = self.uncertainties_df.loc["non_stability", "su"]
        su_nonlinearity = self.uncertainties_df.loc["non_linearity", "su"]
        su_maintenance = self.uncertainties_df.loc["maintenance", "su"]
        su_offset_total = self.uncertainties_df.loc["zero_offset_total", "su"]
        if np.all(temperature) is not None:
            # DEV NOTE 15/11/18: positive chosen over negative
            sensitivity = self.sensitivity * (100 + self.get_deviation_on_temperature(temperature)) / 100
            su_temperature = 0
        elif np.any(temperature) is None:
            sensitivity = self.sensitivity
            #su_temperature = self.su_temperature
            su_temperature = self.uncertainties_df.loc["temperature_response","su"]
        if (np.all(azimuth) is not None) or (np.all(angle_of_incidence) is not None):
            # check if deviation not None (e.g. aoi <90 )
            # print(azimuth,angle_of_incidence)
            deviation_directional = self.get_deviation_on_direction(azimuth, angle_of_incidence)
            if deviation_directional is not None:
                irradiance = irradiance * (100 + deviation_directional) / 100
                su_directional = 0
            if deviation_directional is None:
                su_directional = None
        elif (np.any(azimuth) is None) | (np.any(angle_of_incidence) is None):
            #su_directional = self.su_directional
            su_directional = self.uncertainties_df.loc["directional_response", "su"]
            # DEV NOTE 9/11/18: MC part currently compatible only with specific iteration
        # estimation of expanded uncertainty with irradiance measurements based on voltage and sensitivity only (negligible infrared contribution)
        # first check if su_temperature and su_directional have been properly calculated

        if (su_directional is not None) and (su_temperature is not None):
            voltage = sensitivity * irradiance

            if np.any(mc_coeff) is None:

                # calculate uncertainties related to each parameter
                uv2 = (su_datalogger) ** 2
                us2 = ((su_calibration) ** 2 + (su_nonstability) ** 2 + (su_nonlinearity) ** 2 + (
                    su_temperature) ** 2 + (su_maintenance) ** 2)
                # 29/8/21 removed distinction between offset a and b since from 2018 is a, b and other sources
                # that change also the number of MC values needed
                ue2 = ((su_offset_total) ** 2 + (su_directional) ** 2)
                # calculate sensitivity coefficient
                cv = 1 / self.sensitivity
                cs = -(voltage) / (self.sensitivity ** 2)
                ce = 1
                # calculate combined standard uncertainty
                comb_stnd_uncr = (cv ** 2 * uv2 + cs ** 2 * us2 + ce ** 2 * ue2) ** 0.5
                # calculate combined expanded uncertainty 
                exp_uncr = comb_stnd_uncr * coverage_factor
                uncr = exp_uncr
            elif np.all(mc_coeff) is not None:
                # transformation function from su to
                def su_to_val_rect(su: float, random_num: float, shape="symmetric"):
                    # random number fom 0 to 1
                    if shape == "symmetric":
                        su_to_val_rect = su * (3 ** 0.5) * (-1 + 2 * random_num)
                    elif shape == "positive":
                        su_to_val_rect = su * (3 ** 0.5) * (random_num) * 2
                    elif shape == "negative":
                        su_to_val_rect = -su * (3 ** 0.5) * (random_num) * 2
                    return su_to_val_rect

                def su_to_val_norm(su: float, random_num: float, asymmetric="no"):
                    # random number fom 0 to 1
                    su_to_val_norm = su * (-1 + 2 * random_num)
                    return su_to_val_norm
                    # deviation with normal distribution
                dev_calibration = su_to_val_norm(su_calibration, mc_coeff[0], asymmetric="no")
                # deviation with rectangular distribution asymmetric     
                # deviation with rectangular distribution asymmetric
                # DEV NOTE 29/8/21 in the code shape should be dynamic not fix
                dev_datalogger = su_to_val_rect(su_datalogger, mc_coeff[1])
                dev_nonlinearity = su_to_val_rect(su_nonlinearity, mc_coeff[3])
                dev_temperature = su_to_val_rect(su_temperature, mc_coeff[4])
                dev_maintenance = su_to_val_rect(su_maintenance, mc_coeff[5])
                dev_offset_total = su_to_val_rect(su_offset_total, mc_coeff[6])
                #dev_offset_b = su_to_val_rect(self.su_offset_b, mc_coeff[7], asymmetric="no")
                dev_directional = su_to_val_rect(su_directional, mc_coeff[7])
                # deviation with rectangular distribution negative only  
                dev_nonstability = su_to_val_rect(su_nonstability, mc_coeff[2],shape="negative")
                # DEV NOTE 29/8/21 offset is negative due to paper:
                # 2015 Uncertainty evaluation of measurements with pyranometers and pyrheliometers Konings, Jörgen Habte, Aron
                # however symmetric could be considered as well
                #dev_offset_a = su_to_val_rect(self.su_offset_a, mc_coeff[6], asymmetric="negative")
                # test symmetrical rectangular distribution 
                # dev_nonstability=su_to_val_rect(self.su_nonstability,mc_coeff[2],asymmetric="no")/2     
                # dev_offset_a=su_to_val_rect(self.su_offset_a,mc_coeff[6],asymmetric="no")/2
                # uncertainties per parameter
                v2 = (voltage + dev_datalogger) / voltage
                s2 = ((sensitivity + dev_calibration) *
                      (sensitivity + dev_nonstability) *
                      (sensitivity + dev_nonlinearity) *
                      (sensitivity + dev_temperature) *
                      (sensitivity + dev_maintenance) / sensitivity ** 5)
                e2 = ((irradiance + dev_offset_total) *
                      (irradiance + dev_directional) / irradiance ** 3)
                irr_adj = v2 * s2 * e2 * irradiance
                uncr = irr_adj - irradiance
        elif (su_directional is None) or (su_temperature is None):
            uncr = None
        return uncr



    def get_uncertainty_mc_v2(self,
                           randomstate: np.random.RandomState,
                           irradiance: pd.Series, #previously was float
                           simulations: int,
                           angle_of_incidence: pd.Series = None,
                           temperature: pd.Series = None,
                           azimuth: pd.Series =None,
                           percentile:list =PERCENTILE_MAX,
                           coverage_factor=COVERAGE_FACTOR,
                           uncertainties_labels:list=None,
                           test=_TEST_UNC_MC):
        """
        Version started but not continued
        """
        nn = []
        if np.any(angle_of_incidence) is None or np.any(angle_of_incidence) is np.nan:
            nn.append("angle_of_incidence")
        if np.any(temperature) is None or np.any(temperature) is np.nan:
            nn.append("temperature")  
        if np.any(azimuth) is None or np.any(azimuth) is np.nan:
            nn.append("azimuth")  
        # if len(nn) > 0: print(f"Not valid (None or nan) series: {', '.join(nn)}")
        
        uncertainties_df = self.uncertainties_df
        if uncertainties_labels is not None: uncertainties_df = uncertainties_df[uncertainties_df.index.isin(uncertainties_labels)]
        else: uncertainties_df = self.uncertainties_df
        uncs_mc_cffs = {}

        start_time = dt.datetime.now()
        for i in uncertainties_df.index.to_list():
            distribution, shape = uncertainties_df.loc[i,["distribution","shape"]]
            if distribution == "rectangular":
                if shape == "symmetric": arr_mc = randomstate.uniform(low=-1, high=1, size=(len(irradiance), simulations, 1))
                # 19/10/21 trying quick fix
                # else: arr_mc = randomstate.uniform(low=0, high=1, size=(len(irradiance), simulations, 1))
                elif shape == "positive": arr_mc = randomstate.uniform(low=0, high=1, size=(len(irradiance), simulations, 1))
                #to be checked why still positive coefficient
                elif shape == "negative": arr_mc = randomstate.uniform(low=-1, high=0, size=(len(irradiance), simulations, 1))
                # 19/10/21 not working properly in def uncertainty
            elif distribution == "normal": arr_mc = randomstate.normal(loc=0, scale=1, size=(len(irradiance), simulations, 1))
            arr_mc = arr_mc.round(decimals=MC_NORMAL_DECIMALS)
            uncs_mc_cffs[i] = arr_mc
            # 19/10/21 possible calibration compensating whatever distribution for negative ?
            if test:
                acceptance, unit = uncertainties_df.loc[i, ["acceptance", "unit"]]
                print(f"{i}, {distribution}, {shape}, {acceptance}, {unit}: {min(arr_mc[0])} -> {max(arr_mc[0])}")

            #DEV NOTE 29/8/21 df may be too much memory intensive, could be simplified into dictionary
            #str to avoid 50 becoming 50.0
            if len(percentile) > 1: exp_results = pd.DataFrame(columns=[str(p) for p in percentile])
            #elif isinstance(percentile, np.ndarray) == False:
            else: exp_results = []
            # 13/8/23 v2 keep separate array for mpp processing





    def get_uncertainty_mc(self,
                           randomstate: np.random.RandomState,
                           irradiance: pd.Series, #previously was float
                           simulations: int,
                           angle_of_incidence: pd.Series = None,
                           temperature: pd.Series = None,
                           azimuth: pd.Series =None,
                           percentile:list =PERCENTILE_MAX,
                           coverage_factor=COVERAGE_FACTOR,
                           uncertainties_labels:list=None,
                           directional_error:bool = False,
                           test=_TEST_UNC_MC):
        #return series if single percentile otherwise dataframe
        """
        
        """
        nn = []
        if np.any(angle_of_incidence) is None or np.any(angle_of_incidence) is np.nan:
            nn.append("angle_of_incidence")
        if np.any(temperature) is None or np.any(temperature) is np.nan:
            nn.append("temperature")  
        if np.any(azimuth) is None or np.any(azimuth) is np.nan:
            nn.append("azimuth")  
        # if len(nn) > 0: print(f"Not valid (None or nan) series: {', '.join(nn)}")

        uncertainties_df = self.uncertainties_df
        if uncertainties_labels is not None: uncertainties_df = uncertainties_df[uncertainties_df.index.isin(uncertainties_labels)]
        else: uncertainties_df = self.uncertainties_df
        uncs_mc_cffs = {}

        #start_time = dt.datetime.now()
        for i in uncertainties_df.index.to_list():
            distribution, shape = uncertainties_df.loc[i,["distribution","shape"]]
            if distribution == "rectangular":
                if shape == "symmetric": arr_mc = randomstate.uniform(low=-1, high=1, size=(len(irradiance), simulations, 1))
                # 19/10/21 trying quick fix
                # else: arr_mc = randomstate.uniform(low=0, high=1, size=(len(irradiance), simulations, 1))
                elif shape == "positive": arr_mc = randomstate.uniform(low=0, high=1, size=(len(irradiance), simulations, 1))
                #to be checked why still positive coefficient
                elif shape == "negative": arr_mc = randomstate.uniform(low=-1, high=0, size=(len(irradiance), simulations, 1))
                # 19/10/21 not working properly in def uncertainty
            elif distribution == "normal": arr_mc = randomstate.normal(loc=0, scale=1, size=(len(irradiance), simulations, 1))
            arr_mc = arr_mc.round(decimals=MC_NORMAL_DECIMALS)
            uncs_mc_cffs[i] = arr_mc
            # 19/10/21 possible calibration compensating whatever distribution for negative ?
            if test:
                acceptance, unit = uncertainties_df.loc[i, ["acceptance", "unit"]]
                print(f"{i}, {distribution}, {shape}, {acceptance}, {unit}: {min(arr_mc[0])} -> {max(arr_mc[0])}")

        #run_time = dt.datetime.now() - start_time #end_time - start_time    # 3
        #print(f"Finished Randomstate in {str(run_time)} secs")

        # DEV NOTE 29/8/21 tentative description
        # get directly percentile values not all simulations (better saving memory in graphs)
        # random state to reproduce same,
        # number of uncertainty sources with an assumed gaussian distribution
        """us_nrm_n = 1
        # number of uncertainty sources with an assumed rectangular distribution
        us_rct_n = 8
        # definining sensitivity and voltage as dependent or indipendent from temperature and angle
        # DEV NOTE 8/11/18: to be improved different su for t,a,aoi depending if characterisation or not.
        # generation of entire dataframe with normal distribution
        # loc is mean, scale is the standard deviation
        arr_nrm = randomstate.normal(loc=0, scale=1, size=(len(irradiance), simulations, us_nrm_n))
        # 20/8/21 previously 0 to 1 which is asymmetrical 
        arr_rct = randomstate.uniform(low=-1, high=1, size=(len(irradiance), simulations, us_rct_n))
        # round to 3 decimals
        # DEV NOTE 9/11/18: to be checked sum
        arr_mc = np.concatenate((arr_nrm, arr_rct), axis=2)
        arr_mc = arr_mc.round(decimals=MC_NORMAL_DECIMALS)"""
        # if percentile is not np.ndarray:
        # DEV NOTE 29/8/21 updated len not type to be checked (could be a list or series and not array)
        #if isinstance(percentile, np.ndarray) == True:
        #DEV NOTE 29/8/21 df may be too much memory intensive, could be simplified into dictionary
        #str to avoid 50 becoming 50.0
        if None not in percentile:        
            if len(percentile) > 1: 
                exp_results = pd.DataFrame(columns=[str(p) for p in percentile])
                print(exp_results)
            #elif isinstance(percentile, np.ndarray) == False:
            else: exp_results = []
        elif None in percentile:
            exp_results = {}

        # 12/08/23 For iteration slow, replace with dataframe and lambda ?
        for i in range(0, len(irradiance)):
            # print(irradiance.iloc[i])

            #reinitialise irradiance unc for each irradiance value
            unc_irr = []

            # if len(percentile)==1:
            #        unc_irr = []
            # elif len(percentile) > 1 :
            #    unc_irr = pd.DataFrame(columns=list(percentile))

            # cnt = 0
            for j in range(0, simulations):
                # if cnt == 0: start_time = dt.datetime.now()
                """2
                a = 0                     
                for k in range(0,unc_n-1):
                    if np.isnan(arr_mc[i,j,k]) == True:
                        print([i,j,k])
                    if np.isnan(arr_mc[i,j,k]) == False:
                        a = a + arr_mc[i,j,k]
                    print(a)                    
                """
                # if all valid temperatures provided analyse responsitivity on temperature

                # calculate uncertainty
                unc_sim = self.get_uncertainty(
                    irradiance=irradiance.iloc[i],
                    temperature=temperature.iloc[i] if "temperature" not in nn else [None],
                    azimuth=azimuth.iloc[i] if "azimuth" not in nn else [None],
                    angle_of_incidence=angle_of_incidence[i] if "angle_of_incidence" not in nn else [None],
                    #coverage factor not used in mc
                    coverage_factor=None,
                    uncs_mc_cffs=dict(zip(uncs_mc_cffs.keys(),[u[i,j,:][0] for u in uncs_mc_cffs.values()])),
                    directional_error=directional_error
                    )
                    #mc_coeff=arr_mc[i, j, :])
                
                # testing null by printing it
                # 13/8/23 to be adapted if None uncertainty ?
                if unc_sim is None:
                    # if np.isnan(unc_sim) == True:
                    print([i, j])
                    print(dict(zip(uncs_mc_cffs.keys(),[u[i,j,:][0] for u in uncs_mc_cffs.values()])))
                    #print([unc_sim, irradiance.iloc[i], arr_mc[i, j, :]])


                unc_irr.append(unc_sim)
                
            if None not in percentile:
                if len(percentile) > 1:
                    # providing dataframe if multiple percentile
                    dict_tmp = {}
                    for index, value in enumerate(percentile):
                        dict_tmp[str(value)] = np.percentile(a=unc_irr, q=value)  
                    #exp_results = exp_results.append(dict_tmp, ignore_index=True)
                    exp_results = pd.concat([exp_results, pd.Series(dict_tmp).to_frame().T], ignore_index=True)
                    # print(f"exp_results {exp_results} /n dict_tmp {dict_tmp}")                
                    # example
                    # UNCERTAINTIES_PYRANOMETERS=pd.concat([UNCERTAINTIES_PYRANOMETERS,pd.Series(value).to_frame().T],ignore_index=True)
                #if isinstance(percentile, np.ndarray) == False:
                else:
                    # providing series if single percentile
                    # exp_results.append((np.percentile(a=unc_irr, q=percentile)))
                    exp_results = exp_results.concat((np.percentile(a=unc_irr, q=percentile)))
                    exp_results = pd.Series(exp_results)
            # elif percentile is np.ndarray:
        
            if None in percentile:
                exp_results[str(i)] = unc_irr

        if None in percentile:
            exp_results = pd.DataFrame(exp_results).T
            exp_results.index = irradiance.index

        return exp_results


def get_calibration_factor(# improved version calculating median of other variables
                   df: pd.DataFrame,
                   reference_column: str,
                   field_column: str,
                   calibration_factor:float,
                   method:str,
                   deviation_max=DEVIATION_MAX,
                   readings_min=CALIBRATION_READINGS,  # 21
                   iterations_max=ITERATIONS_MAXIMUM,
                   log=False,
                   median=False,  # if median=True
                   ):
    """

    :param df:
    :param reference_column:
    :param field_column:
    :param calibration_factor:
    :param method:
    :param deviation_max: EURAC used a 4%
    :param readings_min:
    :param iterations_max:
    :param log:
    :param median:
    :return:
    """
    # DEV NOTE 21/8/21 EURAC method seems to not work properly (too few series) avoid
    # NOTES: in ISO 9060:2018 directional response analysed for 40°, 60°, 70°, 80°
    # In ASTM G213 apart of overall uncertainty also zenith step +- 0.3 degree uncertainty assessed as well
    #2/6/21 spotted differences with previous version: using df instead of srs
    df = df.rename(columns={reference_column: "vr_ji", field_column: "vf_ji"}) #, datetime_column: "datetime"})
    #datetime not needed
    #df["datetime"] = pd.DatetimeIndex(df["datetime"])
    df = df[(df["vr_ji"] > 0) & (df["vf_ji"] > 0)]  # filter only for positive values
    #define point calibration factor
    df = df.assign(f_ji=calibration_factor * df["vr_ji"] / df["vf_ji"])  # calculate factor for each point
    df_tmp = df.copy()  # establish temporary df
    valid = True  # initialise break condition
    # DEV NOTE 21/8/21 EURAC method seems to not work properly (too few series) avoid
    if ("EURAC" in method):  # check if distance min-max is not higher than 2%
        # 6/1/20: is EURAC method or other?  #DEV NOTE 19/11/18: traditional irradiance checks should be added
        #for EURAC double deviation max
        if (abs(df_tmp.loc[:, "vr_ji"].max() - df_tmp.loc[:, "vr_ji"].min()) / df_tmp.loc[:, "vr_ji"].mean() > deviation_max*2):
            valid = False
            if log: print("Error: too high percentage deviation from min to max !!")
    if valid == True and ("9847" in method) and ("1a" in method):  # valid == True and # ref ISO 9847 1a & others.
        #iterations not necessary if series starts already with minimum number of values
        for i in range(0, iterations_max - 1):  # artificial limit to ammissible iterations
            if len(df_tmp) < readings_min:  # close if not enough values
                if log == True:
                    print("aborted due to insufficient number of values: " + str(len(df_tmp)))
                    print(df_tmp)
                valid = False
                break
            #calculate reference f_j_tmp as median or not
            if median == False:
                f_j_tmp = calibration_factor * df_tmp["vr_ji"].sum() / df_tmp[
                    "vf_ji"].sum()  # tmp calibration factor
            elif median == True:
                f_j_tmp = calibration_factor * df_tmp["vr_ji"].median() / df_tmp["vf_ji"].median()
            #calculate deviation of single point calibration factor from series calibration factor
            df_tmp = df_tmp.assign(
                f_dev=df["f_ji"].map(lambda x: abs((x - f_j_tmp) / f_j_tmp)))  # 10/1/20 removed factor 100
            # ISO filter not use by eurac
            if "EURAC" in method:
                break
            elif "EURAC" not in method:
                #if highest deviation still acceptable provide results and break
                if df_tmp["f_dev"].max() <= deviation_max:  # stop iterations and provide results if value found
                    if log == True:
                        print("f_j found: " + str(f_j_tmp));
                        print("std of f_j: " + str(df_tmp["f_dev"].std()))
                    valid = True
                    break
                #otherwise remove values and continue
                elif df_tmp["f_dev"].max() > deviation_max:
                    # new filter f_j
                    df_tmp = df_tmp[(df_tmp["f_dev"] <= deviation_max)]
                    valid = False
                    # print("too high max deviation of: "+str(df_tmp["f_dev"].max()))
    if valid == False:
        # print("not valid series")
        return 1, None, None, None, None, None  # 16/1/20 updated to 1 for easy counting of errors
    elif valid == True:
        # DEV NOTE 8/1/19: df_tmp["f_dev"].std() not useful, better std of the series
        # return valid,f_j_tmp,df_tmp["f_dev"].std(),len(df_tmp),df_tmp
        df_tmp["r_ji"] = 1 / df_tmp["f_ji"]

        return 0, f_j_tmp, df_tmp["f_ji"].std() / f_j_tmp, df_tmp["r_ji"].std() / f_j_tmp, len(
            df_tmp), df_tmp  # 090620 relative variance
        # valid Boolean, f_j_tmp scalar,

def get_calibration_factors(df:pd.DataFrame,
                   srs_id_start_end,
                   #parameters for series calibration
                   reference_column: str,
                   field_column: str,
                   calibration_factor:float,
                   method:str=CALIBRATION_METHOD,
                   deviation_max=DEVIATION_MAX,
                   readings_min=CALIBRATION_READINGS,  # 20 instead of 21
                   iterations_max=ITERATIONS_MAXIMUM,
                   log=False,
                   median=False) -> pd.DataFrame:
    #calibration series requires id, start, end
    calibration_series = pd.DataFrame()
    calibration_measurements = pd.DataFrame(columns=["id", "vr_ji", "vf_ji", "dt", "f_ji", "f_dev"])
    for i in list(srs_id_start_end.index.values):
        df_s_tmp = df[(df["dt"] >=  srs_id_start_end.loc[i, "start"]) & (
                df["dt"] <  srs_id_start_end.loc[i, "end"])]  # filter per series
        #TEST 22/8/21 to verify indeed calibration values calculated for high diffuse fraction
        #df_s_tmp =  df_s_tmp[(df_s_tmp["dt"] >=  pd.to_datetime('06062017', format="%d%m%Y", utc="utc")) &
        #              (df_s_tmp["dt"] < pd.to_datetime('07062017', format="%d%m%Y", utc="utc"))]

        if df_s_tmp.empty == False:  # skip if empty
            # vr_j=df_s_tmp["vr_ji"] #not required with get_factor_df2
            # vf_j=df_s_tmp["vf_ji"]
            # dt=df_s_tmp["dt"]
            # print("series from "+str( calibration_series.loc[i,"start"])+" to "+str( calibration_series.loc[i,"end"]))
            # get dataframe and main parameters for calculation
            valid, f_j, f_j_std, r_j_std, len_j, results_j_df = get_calibration_factor(
                df_s_tmp,
                reference_column=reference_column, #"vr_j",
                field_column=field_column, #"vf_j",
                calibration_factor=calibration_factor,
                method=method,
                deviation_max=deviation_max,
                readings_min=readings_min, #filters["readings_count"],  # for eurac
                iterations_max= iterations_max,
                log=log,
                median=median
            )
            # create array with series id
            if results_j_df is not None:
                results_j_df.loc[:, "id"] = np.full(len(results_j_df),  srs_id_start_end.loc[i, "id"])
                # concatenate series df with previous one
                calibration_measurements = pd.concat([calibration_measurements, results_j_df], sort=False)
                # define main results for series
                results_j = pd.DataFrame()
                for column in [c for c in list(df.columns) if (c != "vr_ji") & (c != "vf_ji")]:
                    r = results_j_df.loc[:, column].dropna()
                    if column == "dt": r = r.astype('datetime64[ns]')
                    if len(r) > 0:
                        if column == "dt":
                            # https://stackoverflow.com/questions/43889611/median-of-panda-dt64-column
                            results_j.loc[0, "dt_quantile"] = r.quantile(.5)

                        # taking median only for float64 not for other like dates
                        elif results_j_df[column].dtype == 'float64':
                            results_j.loc[0, str(column + "_median")] = results_j_df.loc[:, column].median()
                            if (column != "azimuth") and (column != "aoi"):
                                results_j.loc[0, str(column + "_std")] = results_j_df.loc[:, column].std()
                results_j.loc[0, "id"] =  srs_id_start_end.loc[i, "id"]
                results_j.loc[0, "start"] =  srs_id_start_end.loc[i, "start"]
                results_j.loc[0, "end"] =  srs_id_start_end.loc[i, "end"]
                results_j.loc[0, "f_j"] = f_j
                results_j.loc[0, "f_j_std"] = f_j_std
                results_j.loc[0, "r_j_std"] = r_j_std
                results_j.loc[0, "len_j"] = len_j
                # append results
                if calibration_series.empty == True:
                    calibration_series = results_j
                elif calibration_series.empty == False:
                    calibration_series = calibration_series.append(results_j)
        elif df_s_tmp.empty:
            if log: print("series "+str( srs_id_start_end.loc[i, "id"])+" from "+str(srs_id_start_end.loc[i, "start"])+" to "+str(srs_id_start_end.loc[i, "end"])+" is empty in the dataframe")
    return calibration_series

def get_calibration_factor_OLD(reference_series: pd.Series,
                  field_series: pd.Series,
                  time_series: pd.Series,
                  calibration_factor: float,
                  method:str,
                  deviation_max=DEVIATION_MAX,
                  readings_min=CALIBRATION_READINGS,  # 21
                  iterations_max=ITERATIONS_MAXIMUM,
                  log=False,
                  median=False
                  ):
    df = pd.DataFrame(
        {"vr_ji": reference_series,
         "vf_ji": field_series,
         "datetime": pd.DatetimeIndex(time_series)
         })
    df = df[(df["vr_ji"] > 0) & (df["vf_ji"] > 0)]  # filter only for positive values
    df = df.assign(f_ji=calibration_factor * df["vr_ji"] / df["vf_ji"])  # calculate factor for each point
    df_tmp = df  # establish temporary df
    valid = True  # initialise break condition
    if ("eurac" in method):  # check if distance min-max is not higher than 2%
        # 6/1/20: is EURAC method or other?
        # DEV NOTE 19/11/18: traditional irradiance checks should be added
        if ((reference_series.max() - reference_series.min()) / reference_series.mean() > deviation_max):
            valid == False
            #print("Error: too high percentage deviation from min to max !!") #too many not useful
    if valid == True and ("9847" in method) and ("1a" in method):
        # ref ISO 9847 1a & others.
        for i in range(0, iterations_max - 1):  # artificial limit to ammissible iterations
            if len(df_tmp) < readings_min:  # close if not enough values
                if log == True:  # log always true
                    print("aborted due to insufficient number of values: " + str(len(df_tmp)))
                    print(df_tmp)
                valid = False
                break
            f_j_tmp = calibration_factor * df_tmp["vr_ji"].sum() / df_tmp["vf_ji"].sum()
            # 17/1/20 test/compare with median? (@Tom)
            # recalculate calibration if conditions ok
            # calculate relative standard deviation from estimated f_j for each point
            # DEV NOTE 8/1/19: deviation from f_j_tmp not the contrary
            # df_tmp=df_tmp.assign(f_dev=df["f_ji"].map(lambda x: abs((x-f_j_tmp)/f_j_tmp)*100))   #BEFORE 10/1/20
            df_tmp = df_tmp.assign(
                f_dev=df["f_ji"].map(lambda x: abs((x - f_j_tmp) / f_j_tmp)))  # 10/1/20 removed factor 100
            # stop iterations and provide results if value found
            if df_tmp["f_dev"].max() <= deviation_max:
                if log == True:
                    print("f_j found: " + str(f_j_tmp))
                    print("std of f_j: " + str(df_tmp["f_dev"].std()))
                    valid = True
                break
            elif df_tmp["f_dev"].max() > deviation_max:
                # new filter f_j
                df_tmp = df_tmp[(df_tmp["f_dev"] <= deviation_max)]
                valid = False
                # print("too high max deviation of: "+str(df_tmp["f_dev"].max()))
    if valid == False:
        # print("not valid series")
        return 1, None, None, None, None  # 16/1/20 updated to 1 for easy counting of errors
    elif valid == True:
        # DEV NOTE 8/1/19: df_tmp["f_dev"].std() not useful, better std of the series
        # return valid,f_j_tmp,df_tmp["f_dev"].std(),len(df_tmp),df_tmp
        return 0, f_j_tmp, df_tmp["f_ji"].std() / f_j_tmp, len(df_tmp), df_tmp
        # valid Boolean, f_j_tmp scalar,





# basic parameters not requiring much time
if _WINDOWS_TESTING != "":
    _CALIBRATIONS_ALL = ["outdoor kz11 e24 2017", "outdoor kz11 e20 2017", "outdoor kz11 e21 2017", "outdoor hfl 2017",
                         "outdoor 2018", "outdoor 2018", "outdoor 2018", "outdoor 2018", "outdoor 2018"]
    _FILTERS_ALL = ["EURAC", "ISO", "clear sky"]

    pd.set_option('display.max_columns', None)
    # to solve copy warning
    pd.set_option('mode.chained_assignment', 'raise')

    # global filled during execution
    FILEPATHS = {}
    FOLDERPATHS = {}
    COLUMNS: Dict[str, Dict] = {}
    CHANNELS = {}

    """STABILITY PARAMETERS (see also pyranometers_datetimes_analysis.ipynb)"""
    # [ISO 9847]
    AOI_MAX = 70
    # 0.2 (20%): uncertainty 95% of HOURLY values for moderate quality (class C) radiometers [WMO 2017)
    KCS_UNC = 0.2
    # 3% (hourly class A) on 1000 also directional response ONLY on class C
    G_UNC = 30
    # adapted from 2% variation from average in [ISO 9847]
    KCS_CV_MAX = 0.02
    PWR_CV_MAX = 0.1
    # strong correlation used for daily creiteria
    PEARSON_MIN = 0.8
    # daylight hours IEC 61724-1-2017
    G_MIN = 20

    """CALIBRATION PARAMETERS"""
    TEST_ROWS_NUMBER = None
    # gpoa and dt replace global and datetime
    COLUMNS_CALIBRATION = ["vr_ji", "vf_ji", "dt", "aoi", "t_ambient", "diffuse", "gpoa", "wind_speed", "azimuth"]
    ITERATIONS_MAX = 100  # to deal with errors
    DEVIATION_MAX = 0.02
    COVERAGE_FACTOR = 1.96

    FILTERS = {}
    FILTER_TYPES = ["series_count", "series_minutes", "readings_count", "beam_min", "diffuse_min", "diffuse_max",
                    "dfraction_max", "aoi_max", "wind_max"]
    # strict requirements while EURAC assigning 0-1 points per requirements based on 9 previous & 10 next measurements (stability) & best series manually selected
    FILTERS["ISO"] = dict(zip(FILTER_TYPES, [999, 20, 20, 20, 0, 999, 0.2, 70, 2]))
    FILTERS["clear sky"] = dict(zip(FILTER_TYPES, [999, 20, 20, 20, 20, 999, 0.2, 70, 2]))



if "calibration 2017" in _WINDOWS_TESTING:
    import clear_sky as csky

    # starting with one pyranometer as example
    clb = "outdoor kz11 e24 2017"
    #DEV NOTE 21/8/21 EURAC method not working TBC
    # method = "EURAC 9847 1a"
    method = "9847 1a"

    """EURAC OUTDOOR CALIBRATION PARAMETERS"""
    # ABD (46° 27’ 28’’ N, 11° 19’ 43’’ E, 247 meters above sea level
    # 46.464010, 11.330354, 247
    # SL 46.458, 11.329, 0
    import clear_sky as csky

    FILTERS["EURAC"] = dict(zip(FILTER_TYPES, [999, 20, 20, 700, 10, 150, 0.15, 70, 2]))

    EURAC_SL = csky.SolarLibrary(latitude=46.458, longitude=11.329, altitude=247,
                                 surface_zenith=0, surface_azimuth=180, timezone="Europe/Rome")

    # DEV NOTE 30/5/21 could be improved as dictionary
    EURAC_REFERENCE_SENSITIVITY = 8.35 / 10 ** 6
    EURAC_RESOLUTION_SECONDS_2017 = 60

    """EURAC OUTDOOR CALIBRATION DATA"""
    DATETIME_FORMAT_EURAC = "%d/%m/%Y %H:%M"
    FOLDERPATHS["outdoor EURAC 2017"] = (  # path to eurac folder
        "C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/170530-170614_eurac_calibration/")
    FILEPATHS["outdoor chp 2017"] = join(FOLDERPATHS["outdoor EURAC 2017"], "calibration_calculations_fm_181119.csv")
    FILEPATHS["outdoor kz11 e24 2017"] = join(FOLDERPATHS["outdoor EURAC 2017"],
                                              "calibration_calculations_Ch4_fm_181122.csv")
    FILEPATHS["outdoor kz11 e20 2017"] = join(FOLDERPATHS["outdoor EURAC 2017"],
                                              "calibration_calculations_Ch5_fm_181122.csv")  #
    FILEPATHS["outdoor kz11 e21 2017"] = join(FOLDERPATHS["outdoor EURAC 2017"],
                                              "calibration_calculations_Ch6_fm_181122.csv")
    FILEPATHS["outdoor hfl 2017"] = join(FOLDERPATHS["outdoor EURAC 2017"],
                                         "calibration_calculations_Ch7_fm_181122.csv")

    COLUMNS_OUTDOOR_2017 = ["PYR_REF_Ch3 raw", "CH raw", "date", "Zenith angle (ETR) SolPos", "T_ambient scaled",
                            "CMP11_diffuse scaled", "CMP11_global_horiz scaled", "Gill_wind_speed scaled",
                            "Azimuth angle SolPos"]

    CHANNELS["outdoor chp 2017"] = "CHP1_direct scaled"
    CHANNELS["outdoor kz11 e24 2017"] = "PYR_TEST_Ch4 raw"
    CHANNELS["outdoor kz11 e20 2017"] = "PYR_TEST_Ch5 raw"
    CHANNELS["outdoor kz11 e21 2017"] = "PYR_TEST_Ch6 raw"
    CHANNELS["outdoor hfl 2017"] = "PYR_TEST_Ch7 raw"

    file_columns = [CHANNELS[clb] if x == "CH raw" else x for x in COLUMNS_OUTDOOR_2017]
    COLUMNS[clb] = dict(zip(file_columns, COLUMNS_CALIBRATION))

    # EURAC SELECTED SERIES
    s1 = [1, '31/05/2017 13:11:00', '31/05/2017 13:31:00']  # end not counted
    s8 = [8, '03/06/2017 10:54:00', '03/06/2017 11:14:00']
    s9 = [9, '03/06/2017 11:14:00', '03/06/2017 11:34:00']
    s10 = [10, '03/06/2017 11:34:00', '03/06/2017 11:54:00']
    s11 = [11, '03/06/2017 11:54:00', '03/06/2017 12:14:00']
    s27 = [27, '07/06/2017 11:05:00', '07/06/2017 11:25:00']
    s28 = [28, '07/06/2017 11:25:00', '07/06/2017 11:45:00']
    s29 = [29, '07/06/2017 11:45:00', '07/06/2017 12:05:00']
    s30 = [30, '07/06/2017 12:05:00', '07/06/2017 12:25:00']
    s32 = [32, '07/06/2017 12:45:00', '07/06/2017 13:05:00']
    s33 = [33, '07/06/2017 13:05:00', '07/06/2017 13:25:00']
    s34 = [34, '07/06/2017 13:25:00', '07/06/2017 13:45:00']
    s35 = [35, '07/06/2017 13:45:00', '07/06/2017 14:05:00']
    s36 = [36, '07/06/2017 14:05:00', '07/06/2017 14:25:00']
    s37 = [37, '07/06/2017 14:25:00', '07/06/2017 14:45:00']
    s60 = [60, '10/06/2017 13:16:00', '10/06/2017 13:36:00']
    s62 = [62, '10/06/2017 13:56:00', '10/06/2017 14:16:00']
    s63 = [63, '10/06/2017 14:16:00', '10/06/2017 14:36:00']
    s64 = [64, '10/06/2017 14:36:00', '10/06/2017 14:56:00']
    s80 = [80, '11/06/2017 10:57:00', '11/06/2017 11:17:00']
    s82 = [82, '11/06/2017 11:37:00', '11/06/2017 11:57:00']
    s83 = [83, '11/06/2017 11:57:00', '11/06/2017 12:17:00']
    s84 = [84, '11/06/2017 12:17:00', '11/06/2017 12:37:00']
    s85 = [85, '11/06/2017 12:37:00', '11/06/2017 12:57:00']
    s86 = [86, '11/06/2017 12:57:00', '11/06/2017 13:17:00']
    s87 = [87, '11/06/2017 13:17:00', '11/06/2017 13:37:00']
    s88 = [88, '11/06/2017 13:37:00', '11/06/2017 13:57:00']
    s89 = [89, '11/06/2017 13:57:00', '11/06/2017 14:17:00']
    s103 = [103, '12/06/2017 11:11:00', '12/06/2017 11:31:00']
    s104 = [104, '12/06/2017 11:31:00', '12/06/2017 11:51:00']
    s109 = [109, '13/06/2017 11:12:00', '13/06/2017 11:32:00']
    s110 = [110, '13/06/2017 11:32:00', '13/06/2017 11:52:00']

    s_eurac_20m = [s1, s8, s9, s10, s11, s27, s28, s29, s30, s32, s33, s34, s35, s36, s37, s60, s62, s63, s64, s80, s82,
                   s83, s84, s85, s86, s87, s88, s89, s103, s104, s109, s110]
    s_chp = s_eurac_20m
    s_ch4 = s_eurac_20m
    s_ch5 = s_eurac_20m
    s_ch6 = s_eurac_20m
    s_ch7 = s_eurac_20m


    """EXECUTION"""

    rqr = FILTERS["EURAC"]
    reference_sensitivity = EURAC_REFERENCE_SENSITIVITY
    reference_calibration_factor = 1 / reference_sensitivity
    sl = EURAC_SL

    resolution = EURAC_RESOLUTION_SECONDS_2017
    filepath = FILEPATHS[clb]


    usecols, columns = [k for k in COLUMNS[clb].keys()], COLUMNS[clb]
    # 280520 modify name depending on file
    df = pd.read_csv(filepath, header=0, skiprows=0, usecols=usecols, nrows=TEST_ROWS_NUMBER)
    df.rename(columns=columns, inplace=True)

    # calculating support values
    df["diffuse_fraction"] = df["diffuse"] / df["gpoa"]
    df["beam"] = df["gpoa"] - df["diffuse"]  #
    df.loc[:, "dt"] = df.loc[:, "dt"].apply(lambda x: pd.to_datetime(x, format=DATETIME_FORMAT_EURAC))
    df.loc[:, "dt"] = pd.DatetimeIndex(df.loc[:, "dt"])
    # importing filters and setup
    filters = FILTERS["EURAC"]

    # defining calibration parameters

    # calibration = mdq.SolarData.Calibration(
    # location=setup.solar_library, method=c,calibration_factor=setup.reference_calibration_factor)
    # df=setup.dataframe
    # start results table

    if "manual" in _WINDOWS_TESTING:
        srs_id_start_end = pd.DataFrame([item for item in s_ch7], columns=["id", "start", "end"])
        for c in ["start", "end"]:   srs_id_start_end.loc[:, c] = srs_id_start_end.loc[:, c].apply(
            lambda x: pd.to_datetime(x, format=DATETIME_FORMAT_EURAC + ":00"))

    elif "flt" in _WINDOWS_TESTING:
        #since testing using df instead of df_flt copy
        import clear_sky as csky
        import datetime

        sl = csky.SolarLibrary(latitude=46.458, longitude=11.329, altitude=247,
                                     surface_zenith=0, surface_azimuth=180, timezone="Europe/Rome")



        # utc conversion before clear sky analysis
        dti = pd.DatetimeIndex(df.loc[:, "dt"], tz=EURAC_SL.timezone).tz_convert(tz='utc')
        df.loc[:, "dt"] = dti
        df.index = dti

        if "flt irr" in _WINDOWS_TESTING:

            df = df[(df["beam"] >= filters["beam_min"]) &
                        (df["diffuse"] > filters["diffuse_min"]) & (df["diffuse"] < filters["diffuse_max"]) &
                        (df["diffuse_fraction"] < filters["dfraction_max"]) &
                        (df["aoi"] <= filters["aoi_max"])].copy()

        #elif "flt none" in _WINDOWS_TESTING:
        #    df.dropna(subset=["gpoa"], inplace=True)


        itr = 1  # 2
        #itr = 2
        # DEV NOTE 24/5/21 be careful keep labels different (or change structure) to avoid overwriting
        steps = [""]  # , "20min"]
        #steps = ["","20min"]  # , "20min"]
        # 20 instead of 21 to allow spot values per minute
        periods = ["20min"]  # , "m"]
        #periods = ["20min","60min"]  # , "m"]
        # minimum 15 series
        counts = [20]  # , 15]
        #counts = [20,3]  # , 15]
        # light requirement on kcs error
        # more than kcs better uncertainty since more balanced during the day and less affected by shift in theory
        # kcs_range=[1-KCS_DEV_LMT,1+KCS_DEV_LMT] #in original algorithm
        kcs_uncs = [None]  * itr
        #kcs_uncs = [KCS_UNC]  * itr
        #kcs_uncs = [0.15] * itr
        # Pearson for initial one since no other strong req but maybe could applied also to later analysis
        pearson_mins = [None] * itr
        # CV analysis only on series not for initial filtering
        kcs_cv_maxs = [None]  # , None] # KCS_CV_MAX]
        #kcs_cv_maxs = [KCS_CV_MAX] * itr
        # initially trying with no requirements on g_uncs. pwr_cv_maxs not needed
        # no valid data for G_UNC 30 or 60, trying with KCS although more stricter for lower elevation
        # g_uncs = [None]*3
        # pwr_cv_maxs


        #tried different filter but not able to remove some cloud ratio over 0.20 (max 0.35)
        #impact on calibration coefficients to be evaluated
        # separately defining the ones not needing comments
        (ms_tz, sl, g_mins, aoi_maxs) = (df, EURAC_SL, [G_MIN] * itr, [AOI_MAX] * itr)

        ms, ms_flt, ms_grps = csky.stabilityflagging(ms_utc=df, sl=sl, steps=steps, periods=periods, counts=counts,
                                                     kcs_uncs=kcs_uncs, pearson_mins=pearson_mins,
                                                     kcs_cv_maxs=kcs_cv_maxs, g_mins=g_mins, aoi_maxs=aoi_maxs)

        # %%
        # taking df merged with info grouping period 20 min and count 20 (in this case only one type df)
        # t as temporary
        t = ms_grps["s_p20min_c20"].copy()
        # removing nan (TBC why exist for no filter)
        t.dropna(subset=["g_stable_p_20min"], inplace=True)
        # taking only values where g_stable_p_20_min is true (minimum 20 values)
        tg = t[t.g_stable_p_20min].groupby("dt_p_20min").agg({"dt": "last"})
        # recreating list of series
        srs_id_start_end = pd.DataFrame()
        #DEV NOTE 21/8/21 using tg.values otherwise tg.dt.values will remove tz (why?)
        srs_id_start_end["end"] = tg.values
        #adding one minute since end is the last valid measurement
        srs_id_start_end["end"] = srs_id_start_end.end.apply(lambda x: x + datetime.timedelta(minutes=1))
        srs_id_start_end["start"] = srs_id_start_end.end.apply(lambda x: x - datetime.timedelta(minutes=20))
        srs_id_start_end["id"] = srs_id_start_end.index


    calibration_series = get_calibration_factors(df=df,
                                                          srs_id_start_end=srs_id_start_end,
                                                          reference_column="vr_j",
                                                          field_column="vf_j",
                                                          calibration_factor=reference_calibration_factor,
                                                          method=method,
                                                          deviation_max=DEVIATION_MAX,
                                                          readings_min=filters["readings_count"],  # for eurac
                                                          iterations_max=ITERATIONS_MAX,
                                                          log=False,
                                                          median=False)


    print(calibration_series)

if "monte carlo" in _WINDOWS_TESTING or "gum" in _WINDOWS_TESTING:
    import clear_sky as csky

    """ EURAC CALIBRATION DATA AS EXAMPLE"""
    EURAC_SL = csky.SolarLibrary(latitude=46.458, longitude=11.329, altitude=247,
                                 surface_zenith=0, surface_azimuth=180, timezone="Europe/Rome")

    DATETIME_FORMAT_EURAC = "%d/%m/%Y %H:%M"

    COLUMNS_OUTDOOR_2017 = ["PYR_REF_Ch3 raw", "CH raw", "date", "Zenith angle (ETR) SolPos", "T_ambient scaled",
                            "CMP11_diffuse scaled", "CMP11_global_horiz scaled", "Gill_wind_speed scaled",
                            "Azimuth angle SolPos"]

    CHANNELS["outdoor kz11 e24 2017"] = "PYR_TEST_Ch4 raw"

    clb = "outdoor kz11 e24 2017"

    file_columns = [CHANNELS[clb] if x == "CH raw" else x for x in COLUMNS_OUTDOOR_2017]
    COLUMNS[clb] = dict(zip(file_columns, COLUMNS_CALIBRATION))
    usecols, columns = [k for k in COLUMNS[clb].keys()], COLUMNS[clb]

    # starting with pyranometer irradiance data as example
    #adapted for the single file
    FOLDERPATHS["outdoor EURAC 2017"] = (  # path to eurac folder
        "C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/170530-170614_eurac_calibration/")

    filepath = join(FOLDERPATHS["outdoor EURAC 2017"], "calibration_calculations_Ch4_fm_181122.csv")
    df = pd.read_csv(filepath, header=0, skiprows=0, usecols=usecols, nrows=TEST_ROWS_NUMBER)




    df.rename(columns=columns, inplace=True)
    df.loc[:, "dt"] = df.loc[:, "dt"].apply(lambda x: pd.to_datetime(x, format=DATETIME_FORMAT_EURAC))
    df.loc[:, "dt"] = pd.DatetimeIndex(df.loc[:, "dt"])


    sl = csky.SolarLibrary(latitude=46.458, longitude=11.329, altitude=247,
                           surface_zenith=0, surface_azimuth=180, timezone="Europe/Rome")

    # utc conversion before clear sky analysis
    dti = pd.DatetimeIndex(df.loc[:, "dt"], tz=EURAC_SL.timezone).tz_convert(tz='utc')
    df.loc[:, "dt"] = dti
    df.index = dti


    #not needed aoi already in eurac dataset
    #cs = sl.getsolardataframe(dti, outputs=["angleofincidence"])
    #df = cs.merge(df, how='right', left_index=True, right_index=True)
    #df.rename(columns={"angleofincidence": "aoi"}, inplace=True)

    """energy yield analysis part"""
    #test sensitivities not the ones from EURAC
    CREST_CMP11_05_SENSITIVITY = 8.65 / 10 ** 6
    CREST_CMP11_06_SENSITIVITY = 8.8 / 10 ** 6
    #adding mrt for directional response
    sensor_uncertainty = Irradiance_Uncertainty( method="isoa cmp11",
                                                sensitivity=CREST_CMP11_06_SENSITIVITY)


    # filtering datasets
    dfs = {}
    dfs["0"] = df[(df["gpoa"] > 0) & (df.aoi < 90)]
    dfs["35"] = df[(df["gpoa"] > 35) & (df.aoi < 90)]
    dfs["200"] = df[(df["gpoa"] > 200) & (df.aoi < 90)]

    #randomstate used also for mc later
    randomstate = np.random.RandomState(210829)


    if "limit" in _WINDOWS_TESTING:
        for k,v in dfs.items():
            dfs[k] = v.sample(n=10, frac=None, replace=False, weights=None, random_state=randomstate, axis=None)

    if "aoi" in _WINDOWS_TESTING:
        for k,v in dfs.items():
            v["pyr_azimuth"] = v["azimuth"].apply(lambda x: (x+225) % 360)
            dfs[k] = v


    coverage_factor = 2
    percentiles = [4.5, 10, 50, 90, 95.5]
    #transforming with str to avoid 50 becoming 50.0
    #columns definition not needed before
    #irradiation_df = pd.DataFrame(columns=[str(p) for p in percentiles])
    irradiation_df = pd.DataFrame()

    #uncertainties to be considered both for gum and mc
    uncertainties_labels = sensor_uncertainty.uncertainties_df.index.to_list()
    uncertainties_labels.remove("zero_offset_total")
    uncertainties_labels.remove("alignment_zenith")
    uncertainties_labels.remove("alignment_azimuth")
    uncs_no_mc = dict(zip(uncertainties_labels,[None]*len(uncertainties_labels )))

    if "gum" in _WINDOWS_TESTING:
        deviations = {}
        for d in [k for k in dfs.keys()]:
            deviations[d] = sensor_uncertainty.get_uncertainty(irradiance=dfs[d]["gpoa"],
            coverage_factor=coverage_factor, uncs_mc_cffs=uncs_no_mc)
            irradiation_df.loc[d, "gum_cf"+'{:.0f}%'.format(coverage_factor)] = deviations[d].sum() / dfs[d]["gpoa"].sum() * 100


    if "monte carlo" in _WINDOWS_TESTING:
        simulations = 100
        #mc not needed for calibration uncertainty since function and cofficients known
        deviations_mc = {}
        for d in [k for k in dfs.keys()]:
            #coverage factor not needed for mc

            irradiance=dfs[d]["gpoa"]
            #26/9/21 testing single value
            if "single" in _WINDOWS_TESTING:
                irradiance=pd.Series(dfs[d]["gpoa"].values[0])
            asymmetric_pair = False

            if "negative" in _WINDOWS_TESTING:
                c = 0
                for i, r in sensor_uncertainty.uncertainties_df.iterrows():
                    if "negative calibration" in _WINDOWS_TESTING:
                        sensor_uncertainty.uncertainties_df.loc[i, "distribution"] = "rectangular"
                        sensor_uncertainty.uncertainties_df.loc[i, "shape"] = "negative"

                    elif "all" in _WINDOWS_TESTING:
                        sensor_uncertainty.uncertainties_df.loc[i, "shape"] = "negative"
                    else:
                        if c % 2 == 0:
                            sensor_uncertainty.uncertainties_df.loc[i, "shape"] = "negative"
                    c += 1
                # 24/10/21 remove calibration to check if effect due to it
                if "negative calibration" in _WINDOWS_TESTING:
                    #6/11/21 negative for sensitivity parameter (e.g. calibration) result in positive due to i = V / s
                    uncertainties_labels = ["calibration"]
                else:
                    uncertainties_labels.remove("calibration")
            elif "monte carlo normal calibration":
                uncertainties_labels = ["calibration"]

            deviations_mc[d] = sensor_uncertainty.get_uncertainty_mc(randomstate=randomstate,
                                                                  irradiance=irradiance, coverage_factor=coverage_factor,
                                                                  simulations=simulations, percentile=percentiles,
                                                                  uncertainties_labels=uncertainties_labels,
                                                                  angle_of_incidence=None)

            for p in percentiles:
                #using str(p) as previously
                #24/10/21 different output than get_uncertainty (without mc)
                if "single" in _WINDOWS_TESTING:
                    irradiation_df.loc[d, str(p)] = deviations_mc[d].loc[:,str(p)].sum() / irradiance.sum() * 100
                else:
                    #25/10/21 should be right but to be checked
                    irradiation_df.loc[d, str(p)] = deviations_mc[d].loc[:, str(p)].sum() / dfs[d]["gpoa"].sum() * 100



    if "gum" in _WINDOWS_TESTING and "aoi" in _WINDOWS_TESTING:
        deviations = {}

        for d in [k for k in dfs.keys()]:
            deviations[d] = sensor_uncertainty.get_uncertainty(irradiance=dfs[d]["gpoa"], angle_of_incidence=dfs[d]["aoi"],
            azimuth = dfs[d]["pyr_azimuth"],
            coverage_factor=coverage_factor, uncs_mc_cffs=uncs_no_mc)
            irradiation_df.loc[d, "gum_drc_cf"+'{:.0f}%'.format(coverage_factor)] = deviations[d].sum() / dfs[d]["gpoa"].sum() * 100

    """deviation0_mc = sensor_uncertainty.get_uncertainty_mc(randomstate=randomstate,
        irradiance=dfs["0"]["gpoa"], coverage_factor=2, simulations=simulations, percentile=percentiles)
    deviation35_mc = sensor_uncertainty.get_uncertainty_mc(randomstate=randomstate,
        irradiance=dfs["35"]["gpoa"], coverage_factor=2, simulations=simulations, percentile=percentiles)
    deviation200_mc = sensor_uncertainty.get_uncertainty_mc(randomstate=randomstate,
        irradiance=dfs["200"]["gpoa"], coverage_factor=2, simulations=simulations, percentile=percentiles)
    irradiation_df = pd.DataFrame(
        columns=percentiles)\
        #,index=[">50", ">90", ">95.5"])"""

    """for index, value in enumerate(percentile):
        print("PERCENTILE " + str(value))

        #irradiation_df.loc[">50", value] = deviation0_mc.loc[:, value].sum(), deviation0_mc.loc[:, value].sum() / df0[
        #    "gpoa"].sum() * 100
        irradiation_df.loc["0_mc", value] = deviation0_mc.loc[:, value].sum() / df0["gpoa"].sum() * 100       
        if "gum" in _WINDOWS_TESTING:
        irradiation_df.loc["35_mc", value] = deviation35_mc.loc[:, value].sum() / df35["gpoa"].sum() * 100
        irradiation_df.loc["200_mc", value] = deviation200_mc.loc[:, value].sum() / df200["gpoa"].sum() * 100
        irradiation_df"""


    print(irradiation_df)