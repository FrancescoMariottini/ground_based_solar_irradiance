# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:16:20 2018
maybe not implemented in calibration yet 
@author: wsfm
"""
#mport numpy as np #for operators
from scipy.optimize import fsolve # for equation solving
from scipy import optimize
import numpy as np
import math as m #for sin and cos
import solarlibraries_py.pvlibinterface as pvint
import pandas as pd
import datetime as dtt
import meteodataquality_py.meteodataquality as mdq
import os
from pvlib import solarposition as sp
from pvlib import pvsystem as pvsys
import pvlib

import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)



_PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"

_PATH_TO_INPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/191108_crest_modules/"
_FILE_MMS1=r"MMS1_CR1000_Meteo_2018-09-18_TimestampCorrected_fm.csv"
_FILE_MONITOR=r"PPUK_Outdoor_CH_003_IVMain_2018_09_18_fm.csv" #every minute
_FILE_MMS2=r"mms2tom_fm_spot.csv"
_FILE_IV=r"PPUK_Outdoor_CH_003_IVMain_2018_09_18_fm.csv"
_FILE_INDOOR=r"SolarWorld_TC Pmax_fm.csv"



print("TIME PER MAX ANGLE VARIATION")
LATITUDE= 52.758264
LONGITUDE= -1.248474

TILT_DELTA = 1 #max 1 error both sides IEC 61724-1 2017
DATETIMES = ['201910021200','201910021200']

SolLib=pvint.SolarLibrary(latitude=LATITUDE,longitude=LONGITUDE)


test_functions = ["pr"]
test_functions = ["pvsystems"]
test_functions = ["modulesearch"]
test_functions = ["production1"]
#timemaxzenithvariation
test_functions = ["datetime"]
#test_functions = ["stability"]
#test_functions = ["extend"]
test_functions = ["blackbody"]
test_functions = ["tmp"]
test_functions = [""]

test_functions = ["19/12/18","calibration","multiple"]
# before 26/7/25 was 19/12/18, try again 12/12/18, giving results in data12 of Calibrations_191210
test_functions = ["12/12/18","calibration","multiple","sensitivity"]


TOTAL_ZERO_OFFSET={'uncertainty':'total_zero_offset','id':'isob','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':10,'acceptance_b':21,'acceptance_c':41}
NON_STABILITY={'uncertainty':'non_stability','id':'isoc1','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.8,'acceptance_b':1.5,'acceptance_c':3}
NON_LINEARITY={'uncertainty':'non_linearity','id':'isoc2','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.5,'acceptance_b':1,'acceptance_c':3}
DIRECTIONAL_RESPONSE={'uncertainty':'directional_response','id':'isoc3','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':10,'acceptance_b':20,'acceptance_c':30}
SPECTRAL_ERROR={'uncertainty':'spectral_error','id':'isoc4','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.5,'acceptance_b':1,'acceptance_c':5}
TEMPERATURE_RESPONSE={'uncertainty':'temperature_response','id':'isoc5','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':1,'acceptance_b':2,'acceptance_c':4}
TILT_RESPONSE={'uncertainty':'tilt_response','id':'isoc6','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.5,'acceptance_b':2,'acceptance_c':5}
SIGNAL_PROCESSING={'uncertainty':'signal_processing','id':'isoc7','parameter':'irradiance','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':2,'acceptance_b':5,'acceptance_c':10}
CALIBRATION={'uncertainty':'calibration','id':'astm1','parameter':'sensitivity','distribution':'symmetric','shape':'symmetric','divisor':2,'acceptance_a':5.62,'acceptance_b':5.62,'acceptance_c':5.62}
MAINTENANCE={'uncertainty':'maintenance','id':'astm8','parameter':'sensitivity','distribution':'rectangular','shape':'symmetric','divisor':3**0.5,'acceptance_a':0.3,'acceptance_b':0.3,'acceptance_c':0.3}
   




if "calibration" in test_functions and "single" in test_functions and "12/12/18" in test_functions:
    IR = 700 
    FR = 1/8.35*1000    
    def equations2(p):
        i1,i2,f18 = p
        FR = 1000/8.35
        VR_1 = 7.09191281078571
        VR_2 = 7.45800158526191
        V18_1 = 8.462118463
        V18_2 = 8.025609694
        F = (
        (FR*VR_1+f18*V18_2)/2-i1,
        (f18*V18_1+FR*VR_2)/2-i2,
        FR*(VR_1+VR_2)/(V18_2+V18_1)-f18
        )
        return F
    zGuess = [IR,IR,FR]
    i1,i2,f18 =  fsolve(equations2,zGuess)  
    FR = 1000/8.35
    VR_1 = 7.09191281078571
    VR_2 = 7.45800158526191
    V18_1 = 8.462118463
    V18_2 = 8.025609694
    print("irradiances")
    print([
    FR*VR_1,
    f18*V18_2,
    f18*V18_1,
    FR*VR_2,
    f18    
    ])
    print("f18")
    print(f18)

# 27/7/25 changed into elif
# 27/7/25 old version 
elif "calibration" in test_functions and "multiple" in test_functions and "old" in test_functions and "12/12/18" in test_functions:
    print("MULTIPLE CALIBRATION RESULT 12/12/18 (OLD)")
    def equations(p):
        i1,i2,i3,i4,f18,f12,f13 = p
        FR = 1/8.35*1000
        VR_1 = 7.09191281078571
        VR_2 = 7.45800158526191
        VR_3 = 7.568993156
        VR_4 = 7.37489239095238
        V18_1 = 8.462118463
        V18_2 = 8.025609694
        V18_3 = 8.03380221
        V18_4 = 8.02532257
        V12_1 = 8.299686156
        V12_2 = 8.284438947
        V12_3 = 8.30562187
        V12_4 = 8.547420543
        V13_1 = 8.070679434
        V13_2 = 8.052647269
        V13_3 = 7.95820422
        V13_4 = 7.955149851
        #beware different positions/channels order for 12&13
        F=(
        (FR*VR_1+f18*(V18_2+V18_3+V18_4))/4-i1,
        (f18*V18_1+FR*VR_2+f13*(V13_3+V13_4))/4-i2,
        (f13*(V13_1+V13_2)+FR*VR_3+f12*V12_4)/4-i3,
        (f12*(V12_1+V12_2+V12_3)+FR*VR_4)/4-i4,
        FR*(VR_1+VR_2)/(V18_2+V18_1)-f18,    
        FR*(VR_2+VR_3)/(V13_3+V13_2)-f13,
        FR*(VR_3+VR_4)/(V12_4+V12_3)-f12)
        return F
    IR = 800
    FR = 1/8.35*1000
    zGuess = [IR,IR,IR,IR,FR,FR,FR]
    # 26/7/25 not necessary system in this case
    i1,i2,i3,i4,f18,f12,f13 = fsolve(equations,zGuess)   
    # 26/7/25 using calibration in equations but printing sensivities
    
    
    FR = 1/8.35*1000
    VR_1 = 7.09191281078571
    VR_2 = 7.45800158526191
    VR_3 = 7.568993156
    VR_4 = 7.37489239095238
    V18_1 = 8.462118463
    V18_2 = 8.025609694
    V18_3 = 8.03380221
    V18_4 = 8.02532257
    V12_1 = 8.299686156
    V12_2 = 8.284438947
    V12_3 = 8.30562187
    V12_4 = 8.547420543
    V13_1 = 8.070679434
    V13_2 = 8.052647269
    V13_3 = 7.95820422
    V13_4 = 7.955149851
    
    
    
    print("f18")
    s_18_a = 1000/(FR*VR_1/V18_2)
    s_18_b = 1000/(FR*VR_2/V18_1)
    print(1/f18*1000)

    print("f12")
    s_12_a = 1000/(FR*VR_3/V12_4)
    s_12_b = 1000/(FR*VR_4/V12_3)
    print(1/f12*1000)

    print("f13")
    s_13_a = 1000/(FR*VR_2/V13_3)
    s_13_b = 1000/(FR*VR_3/V13_2)
    print(1/f13*1000)
    

# 27/7/25 new version with sensitivity
# 27/7/25 new version 
elif "calibration" in test_functions and "multiple" in test_functions and "12/12/18" in test_functions and "sensitivity" in test_functions:
    print("MULTIPLE CALIBRATION RESULT 12/12/18")
    def equations(p):
        i1,i2,i3,i4,s18,s12,s13 = p
        sR = 8.35/1000
        VR_1 = 7.09191281078571
        VR_2 = 7.45800158526191
        VR_3 = 7.568993156
        VR_4 = 7.37489239095238
        V18_1 = 8.462118463
        V18_2 = 8.025609694
        V18_3 = 8.03380221
        V18_4 = 8.02532257
        V12_1 = 8.299686156
        V12_2 = 8.284438947
        V12_3 = 8.30562187
        V12_4 = 8.547420543
        V13_1 = 8.070679434
        V13_2 = 8.052647269
        V13_3 = 7.95820422
        V13_4 = 7.955149851
        #beware different positions/channels order for 12&13
        # 27/7/25 modified comparing to previous version multiple calibration
        F=(
        
        (VR_1/sR+(V18_2+V18_3+V18_4)/s18)/4-i1,
        
        (V18_1/s18+VR_2/sR+(V13_3+V13_4)/s13)/4-i2,
        
        ((V13_1+V13_2)/s13+VR_3/sR+V12_4/s12)/4-i3,
        
        ((V12_1+V12_2+V12_3)/s12+VR_4/sR)/4-i4,   
        
        sR*(V18_1/i2*i1/VR_1 + V18_2/i1*i2/VR_2 + V18_3/i1*i3/VR_3 + V18_4/i1*i4/VR_4)/4-s18,  
        
        sR*(V13_1/i3*i1/VR_1 + V13_2/i3*i2/VR_2+ V13_3/i2*i3/VR_3 + V13_4/i2*i4/VR_4)/4-s13,
        
        sR*(V12_1/i4*i1/VR_1 + V12_2/i4*i2/VR_2 + V12_3/i4*i3/VR_3 + V12_4/i3*i4/VR_4)/4-s12)
        
        return F
    
    IR = 800
    sR = 8.35/1000
    zGuess = [IR,IR,IR,IR,sR,sR,sR]
    # 26/7/25 not necessary system in this case
    i1,i2,i3,i4,s18,s12,s13 = fsolve(equations,zGuess)   
    # 26/7/25 using calibration in equations but printing sensivities
       
    VR_1 = 7.09191281078571
    VR_2 = 7.45800158526191
    VR_3 = 7.568993156
    VR_4 = 7.37489239095238
    V18_1 = 8.462118463
    V18_2 = 8.025609694
    V18_3 = 8.03380221
    V18_4 = 8.02532257
    V12_1 = 8.299686156
    V12_2 = 8.284438947
    V12_3 = 8.30562187
    V12_4 = 8.547420543
    V13_1 = 8.070679434
    V13_2 = 8.052647269
    V13_3 = 7.95820422
    V13_4 = 7.955149851
    
    
    def get_std_unc(measurements:np.array):
        # Calculate the standard deviation (sample standard deviation)
        std_dev = np.std(measurements, ddof=1)        
        # Calculate uncertainty of repeatability (standard deviation of the mean)
        uncertainty_repeatability = std_dev / np.sqrt(len(measurements))       
        # Calculate the mean
        mean_value = np.mean(measurements)
        # Convert to percentage of the mean
        std_dev_percent = (std_dev / mean_value) * 100
        uncertainty_repeatability_percent = (uncertainty_repeatability / mean_value) * 100        
        return std_dev_percent, uncertainty_repeatability_percent

       
    
    print("s18")
    s_18_1 = sR*V18_1/i2*i1/VR_1
    s_18_2 = sR*V18_2/i1*i2/VR_2
    s_18_3 = sR*V18_3/i1*i3/VR_3
    s_18_4 = sR*V18_4/i1*i4/VR_4
    s, u = get_std_unc(np.array([s_18_1,s_18_2,s_18_3,s_18_4]))
    
    print(f"average {s18}, s {s}, u u {u} of: {s_18_1, s_18_2, s_18_3, s_18_4}")
 
    print("s13")
    s_13_1 = sR*V13_1/i3*i1/VR_1
    s_13_2 = sR*V13_2/i3*i2/VR_2
    s_13_3 = sR*V13_3/i2*i3/VR_3
    s_13_4 = sR*V13_4/i2*i4/VR_4
    
    s, u = get_std_unc(np.array([s_13_1,s_13_2,s_13_3,s_13_4]))
    
    print(f"average {s13}, s {s}, u u {u} of: {s_13_1, s_13_2, s_13_3, s_13_4}")

    print("s12")
    s_12_1 = sR*V12_1/i4*i1/VR_1
    s_12_2 = sR*V12_2/i4*i2/VR_2
    s_12_3 = sR*V12_3/i4*i3/VR_3
    s_12_4 = sR*V12_4/i3*i4/VR_4
    
    s, u = get_std_unc(np.array([s_12_1,s_12_2,s_12_3,s_12_4]))
    
    print(f"average {s12}, s {s}, u u {u} of: {s_12_1, s_12_2, s_12_3, s_12_4}")


    
# 27/7/25 new version with calibration converging
elif "calibration" in test_functions and "multiple" in test_functions and "12/12/18" in test_functions:
    print("MULTIPLE CALIBRATION RESULT 12/12/18")
    def equations(p):
        i1,i2,i3,i4,f18,f12,f13 = p
        FR = 1/8.35*1000
        VR_1 = 7.09191281078571
        VR_2 = 7.45800158526191
        VR_3 = 7.568993156
        VR_4 = 7.37489239095238
        V18_1 = 8.462118463
        V18_2 = 8.025609694
        V18_3 = 8.03380221
        V18_4 = 8.02532257
        V12_1 = 8.299686156
        V12_2 = 8.284438947
        V12_3 = 8.30562187
        V12_4 = 8.547420543
        V13_1 = 8.070679434
        V13_2 = 8.052647269
        V13_3 = 7.95820422
        V13_4 = 7.955149851
        #beware different positions/channels order for 12&13
        # 27/7/25 modified comparing to previous version multiple calibration
        F=(
        (FR*VR_1+f18*(V18_2+V18_3+V18_4))/4-i1,
        (f18*V18_1+FR*VR_2+f13*(V13_3+V13_4))/4-i2,
        (f13*(V13_1+V13_2)+FR*VR_3+f12*V12_4)/4-i3,
        (f12*(V12_1+V12_2+V12_3)+FR*VR_4)/4-i4,         
        FR*(VR_1/i1*i2/V18_1+VR_2/i2*i1/V18_2+VR_3/i3*i1/V18_3+VR_4/i4*i1/V18_4)/4-f18,  
        FR*(VR_1/i1*i3/V13_1+VR_2/i2*i3/V13_2+VR_3/i3*i2/V13_3+VR_4/i4*i2/V13_4)/4-f13,
        FR*(VR_1/i1*i4/V12_1+VR_2/i2*i4/V13_2+VR_3/i3*i4/V12_3+VR_4/i4*i3/V12_4)/4-f12)
        return F
    
    IR = 800
    FR = 1/8.35*1000
    zGuess = [IR,IR,IR,IR,FR,FR,FR]
    # 26/7/25 not necessary system in this case
    i1,i2,i3,i4,f18,f12,f13 = fsolve(equations,zGuess)   
    # 26/7/25 using calibration in equations but printing sensivities
    
    
    FR = 1/8.35*1000
    VR_1 = 7.09191281078571
    VR_2 = 7.45800158526191
    VR_3 = 7.568993156
    VR_4 = 7.37489239095238
    V18_1 = 8.462118463
    V18_2 = 8.025609694
    V18_3 = 8.03380221
    V18_4 = 8.02532257
    V12_1 = 8.299686156
    V12_2 = 8.284438947
    V12_3 = 8.30562187
    V12_4 = 8.547420543
    V13_1 = 8.070679434
    V13_2 = 8.052647269
    V13_3 = 7.95820422
    V13_4 = 7.955149851
    
    
    print("f18")
    s_18_1 = 1000/(FR*VR_1/i1*i2/V18_1)
    s_18_2 = 1000/(FR*VR_2/i2*i1/V18_2)
    s_18_3 = 1000/(FR*VR_3/i3*i1/V18_3)
    s_18_4 = 1000/(FR*VR_4/i4*i1/V18_4)  
    
            
    print(s_18_1, s_18_2, s_18_3, s_18_4)
    
    print(1/f18*1000)

    print("f12")
    # s_12_a = 1000/(FR*VR_3/V12_4)
    # s_12_b = 1000/(FR*VR_4/V12_3)
    print(1/f12*1000)

    print("f13")
    # s_13_a = 1000/(FR*VR_2/V13_3)
    # s_13_b = 1000/(FR*VR_3/V13_2)
    print(1/f13*1000)
    
# 27/7/25 changed into elif
elif "calibration" in test_functions and "multiple" in test_functions and "19/12/18" in test_functions:
    print("MULTIPLE CALIBRATION RESULT 19/12/18")
    def equations(p):
        i1,i2,i3,i4,f18,f12,fe = p
        fr = 1/8.35*1000
        vr_1 = 7.322558635 #24/6/20 before: 7.323333557
        vr_2 = 7.539660277 #24/6/20 before:7.544916627
        vr_3 = 7.436553847 #24/6/20 before:7.435674153
        vr_4 = 7.074796867 #24/6/20 before:7.078927529
        vr_5 = 7.347565609 #24/6/20 before:7.344274886
        v18_1 = 8.041792061 #24/6/20 before:8.045112157
        v18_2 = 8.052163284 #24/6/20 before:8.05688886
        v18_3 = 8.04825209 #24/6/20 before:8.04484898
        v18_4 = 8.459997735 #24/6/20 before:8.461707361
        v18_5 = 8.485678099 #24/6/20 before:8.48313578
        v12_1 = 8.422930612 #24/6/20 before:8.425579474
        v12_2 = 8.425762452 #24/6/20 before:8.430230692
        v12_3 = 8.516402164 #24/6/20 before:8.516830793
        v12_4 = 8.538131489 #24/6/20 before:8.540423501
        v12_5 = 8.53727855 #24/6/20 before:8.535345428
        ve_1 = 7.827129127 #24/6/20 before:7.829976031
        ve_2 = 7.637175617 #24/6/20 before:7.639888066
        ve_3 = 7.640742596 #24/6/20 before:7.641310716
        ve_4 = 7.658052528 #24/6/20 before:7.659035358
        ve_5 = 7.355404016 #24/6/20 before: 7.358960864  
        #beware different positions/channels order for 12&13
        F=(
           
        (f18*(v18_1+v18_2+v18_3)+fr*vr_4+fe*ve_5)/5-i1,
        
        (f12*(v12_1+v12_2)+fr*vr_3+f18*(v18_4+v18_5))/5-i2,
        
        (fe*(ve_1)+fr*vr_2+f12*(v12_3+v12_4+v12_5))/5-i3,
        
        (fr*(vr_1+vr_5)+fe*(ve_2+ve_3+ve_4))/5-i4,
        
        FR*(vr_3+vr_4)/(v18_3+v18_4)-f18,
        
        FR*(vr_2+vr_3)/(v12_3+v12_2)-f12,
        
        FR*((vr_1+vr_2+vr_4+vr_5)/(ve_1+ve_2+ve_4+ve_5))-fe) #9.038564681881398
        
        #FR*((vr_4+vr_5)/(ve_4+ve_5))-fe)     
        #FR*((vr_1+vr_2)/(ve_1+ve_2))-fe) #9.038564681881398
            
        return F
    
    IR = 800
    FR = 1/8.35*1000
    zGuess = [IR,IR,IR,IR,FR,FR,FR]
    i1,i2,i3,i4,f18,f12,fe = fsolve(equations,zGuess)   
    print("f18")
    print(1/f18*1000)
    print("f12")
    print(1/f12*1000)
    print("fe")
    print(1/fe*1000)


# 27/7/25 changed into elif
elif "calibration" in test_functions and "single" in test_functions and "19/12/18" in test_functions:
    IR = 900 
    FR = 1/8.35*1000  
    zGuess = [IR,IR,FR] 

    VR_A=7.322558635
    VR_B=7.539660277
    VT_B=7.827129127
    VT_A=7.637175617
    
    def equations2(p):
        i1,i2,ft = p
        F = (
        (FR*VR_A+ft*VT_B)/2-i1,
        (ft*VT_A+FR*VR_B)/2-i2,
        FR*(VR_A+VR_B)/(VT_B+VT_A)-ft
        )
        return F

    i1,i2,ft =  fsolve(equations2,zGuess)     
    print("irradiances at 43")
    print([
    VR_A*FR,
    VT_B*ft,
    VT_A*ft,
    VR_B*FR   
    ])
    print("sensitivity eurac [micron]") #0.008688268244262018
    print(1/FR,1/ft)

    VR_A=7.074796867
    VR_B=7.347565609
    VT_B=7.658052528
    VT_A=7.355404016
    
    i1,i2,ft =  fsolve(equations2,zGuess)     
    print("irradiances at 14")
    print([
    VR_A*FR,
    VT_B*ft,
    VT_A*ft,
    VR_B*FR   
    ])
    print("sensitivity eurac [micron]")
    print(1/FR,1/ft) #0.008692221011017668




if "tmp" in test_functions:

    PATH_TO_INPUT=r"C:\Users\wsfm\OneDrive - Loughborough University\_Personal_Backup\_Research_Master\_Data Sources\_crest_pyranometer_calibration_indoor"
    
    files_list = [f for f in os.listdir(PATH_TO_INPUT) if f.endswith('.txt')]
    
    print(files_list)










if "blackbody" in test_functions:
    #modified from:
    #http://python4esac.github.io/fitting/example_blackbody.html
        
    from scipy.optimize import curve_fit
    import pylab as plt
    import numpy as np
    
    def blackbody_lam(lam, T):
        #Blackbody as a function of wavelength (um) and temperature (K).
        #returns units of erg/s/cm^2/cm/Steradian
        from scipy.constants import h,k,c,pi
        lam = 1e-6 * lam # convert nanometers to metres
        #return 2*h*c**2 / (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))
        return 8*pi*h*c**2 / (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))
    
    wa = np.linspace(0.1, 2, 100)   # wavelengths in um
    #numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]
    #Return evenly spaced numbers over a specified interval.  
    
    T1 = 5000.
    T2 = 8000.
    y1 = blackbody_lam(wa, T1)
    y2 = blackbody_lam(wa, T2)
    ytot = y1 + y2
    
    np.random.seed(1) # make synthetic data with Gaussian errors
    
    sigma = np.ones(len(wa)) * 1 * np.median(ytot)
    ydata = ytot + np.random.randn(len(wa)) * sigma
    ydata2 = ytot/2 + ydata/2
    ydata_b = ytot + np.random.randn(len(wa)) * sigma/2
    
    # plot the input model and synthetic data
    
    plt.figure()
    plt.plot(wa, y1, ':', lw=2, label='T1=%.0f' % T1) #float value in points
    plt.plot(wa, y2, ':', lw=2, label='T2=%.0f' % T2)
    plt.plot(wa, ytot, ':', lw=2, label='T1 + T2\n(true model)')
    plt.plot(wa, ydata, ls='steps-mid', lw=2, label='Fake data A')
    plt.plot(wa, ydata2, ls='steps-mid', lw=2, label='Fake data A/2')
    plt.plot(wa, ydata_b, ls='steps-mid', lw=2, label='Fake data B')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Intensity (erg/s/cm$^2$/cm/Steradian)')
    

if "extend" in test_functions:
    df=pd.DataFrame({
            "dt":pd.date_range(start='1/1/2018', periods=2),
            "a":["b","c"]})
    
    df_s=pd.DataFrame(columns=df.columns)
      
    
    for column in df.columns:
        if column!="seconds":
            srs_tmp = df.loc[:,column].repeat(5)
            df_s.loc[:,column]=srs_tmp.values            
    srs_tmp=pd.concat([pd.Series(range(0,5)) for index in df.index],axis=0)
    df_s.loc[:,"seconds"]=srs_tmp.values
    print(df_s)
    
    df_s["dt"]=df_s.apply(lambda x:x["dt"]+dtt.timedelta(seconds=x["seconds"]),axis=1)
    
    print(df_s)
    


if "datetime" in test_functions:
    naive_times = pd.DatetimeIndex(start='28/11/2015', end='29/11/2015', freq='1s')

    naive_irradiance = np.random.randint(1000, size=len(naive_times))
    naive_irradiance2 = naive_irradiance + np.random.randint(10, size=len(naive_times))
    df = pd.DataFrame({"datetime":pd.DatetimeIndex(naive_times),
                       "irradiance":naive_irradiance,
                       "irradiance2":naive_irradiance2
                       },index=pd.DatetimeIndex(naive_times))
    #https://stackoverflow.com/questions/14529838/apply-multiple-functions-to-multiple-groupby-columns
    #from scipy.stats import pearsonr
    def f(x):
        d={}
        d['pearson']=((x['irradiance'].count()*(x['irradiance']*x['irradiance2']).sum()-x['irradiance'].sum()*x['irradiance2'].sum())
        /x['irradiance'].count()        
        /((((x['irradiance']*x['irradiance']).sum()-(x['irradiance'].sum())**2/x['irradiance'].count()))
        *(((x['irradiance2']*x['irradiance2']).sum()-(x['irradiance2'].sum())**2/x['irradiance2'].count())))**(1/2))
        
        d['kcs']=x['irradiance'].mean()
        d['kcs_len']=x['irradiance'].count()
        d['kcs_min']=x['irradiance'].min()
        d['var']=x["irradiance"].std()/x["irradiance"].mean()
        
        
        #return pd.Series(d,index=['pearson'])    
        return pd.Series(d, index=['kcs','kcs_len','kcs_min','var'])
    
    
       # df_grp=pd.DataFrame({
       #"kcs":kcs_mean,
       #"kcs_var":kcs_var,
       #"kcs_len":kcs_len, 
       #"aoi":aoi_mean,
       #"irradiance":irradiance_mean,
       #"irradiance_cs":irradiance_cs_mean,
       #"power":agg.loc[:,("power","mean")].values,
       #"power_var":agg.loc[:,("power","var")].values,
       #"pearson": pearson,
       #"datetime":agg.loc[:,("datetime","max")].values})
        
    
    
    
    #d['pearson']=x.apply(lambda x:pearsonr(x["irradiance"],x["irradiance2"]),axis=1)
      
    nt_grp = df.groupby([pd.Grouper(key="datetime",freq="60s",label='right')])
    
    #t = nt_grp.apply(f)
    
    df.loc[:,"dt_test"]= nt_grp["datetime"].transform(max)
    
    
    
    nt_grp = df.groupby(by=[pd.Grouper(key="datetime",freq="3600s",label='right')])
    
    nt_agg = nt_grp.agg({'irradiance':max,'irradiance2':min})
    
    nt_agg_dt = nt_agg.index
    
    df_slc = df.loc[df.loc[:,"datetime"].isin(nt_agg_dt),:]
    
    df_slc.loc[:,"max"]=df_slc.loc[:,"datetime"].transform(lambda x: nt_agg.loc[x,"irradiance"])
    
    
    
    
        
    #print(nt_agg)


    #nt_trn2 = nt_grp.transform({'irradiance':[np.var,np.mean,len]})
    #df2 = df.iloc[15:len(naive_times)-12,:]
    #nt_grp2 = df2.groupby([pd.Grouper(key="datetime",freq="60s",label='right')])
    #nt_agg2 = nt_grp2.agg({"datetime":[np.min],"irradiance":["mean"]})
    #print(naive_times)
    
    
    


if "modulesearch" in test_functions:
    crest_module=[
    "Solarworld_SW245_mono", #replaced with SolarWorld in manufacturer
    "Solarworld_SW245_mono",
    "Trina_TSM-195DC80.08",
    "Trina_TSM-195DC80.08",
    "Sanyo-HIT_HIT-N240SE11",
    "Sanyo-HIT_HIT-N240SE11",
    "Trina_TSM-235PC05",
    "Trina_TSM-180DC01A",
    "Trina_TSM-235PC05",
    "Solarworld_SW245_poly",
    "Solarworld_SW245_poly",
    "Trina_TSM-180DC01A",
    "Kyocera_KD240GH-2PB",
    "Kyocera_KD240GH-2PB",
    "Bosch_M60 EU43117 255Wp",
    "Bosch_M60 EU30117 245Wp",
    "Qcells_Qpro-245",
    "Yingli_YL260C-30b ('bifacial')",
    "Yingli_YL260C-30b (non-bifacial)",
    "Bosch_M60 EU43117 255Wp",
    "Yingli_YL260C-30b ('bifacial')",
    "Yingli_YL260C-30b (non-bifacial)",
    "Qcells_Qpro-245",
    "Bosch_M60 EU30117 245Wp",
    "Sharp_ND-R245A6",
    "Sharp_ND-R245A5"]
    
    crest_module_manufacturer=[
    "Solarworld", 
    "Solarworld",
    "Trina",
    "Trina",
    "Sanyo",
    "Sanyo",
    "Trina",
    "Trina",
    "Trina",
    "Solarworld",
    "Solarworld",
    "Trina",
    "Kyocera",
    "Kyocera",
    "Bosch",
    "Bosch",
    "Qcells",
    "Yingli",
    "Yingli",
    "Bosch",
    "Yingli",
    "Yingli",
    "Qcells",
    "Bosch",
    "Sharp",
    "Sharp"]
    
    
    crest_module_details=[]
    for index in range(0,len(crest_module)-1):
        value=crest_module[index][len(crest_module_manufacturer[index]):]
        crest_module_details.append(value)
    print(len(crest_module))
  
    
    
    
    import nltk    
    sandia_modules = pvsys.retrieve_sam('SandiaMod')    
    distance = []
    for value in crest_module:
        distance.append(str(value)+"_d")
    df = pd.DataFrame(columns=crest_module+distance)     
    df = pd.DataFrame()
    for index in range(0,len(crest_module)-1): 
        df_tmp = pd.DataFrame(columns=[crest_module[index],"dst"])
        manufacturer = str(crest_module_manufacturer[index])
        MANUFACTURER = manufacturer.upper()
        details = str(crest_module_details[index])
        DETAILS = details.upper()
        for sandia in list(sandia_modules.columns): 
            sandia = str(sandia)
            SANDIA = sandia.upper()
            if MANUFACTURER not in SANDIA:d=0
            elif MANUFACTURER in SANDIA: d=1/nltk.edit_distance(DETAILS,SANDIA)
            df_tmp=df_tmp.append({crest_module[index]:sandia,"dst":d},ignore_index=True)
        df_tmp=df_tmp.sort_values('dst',ascending=False)
        df_tmp=df_tmp.reset_index()
        #print(crest_module_manufacturer[index]+" "+str(index)+" "+crest_module_details[index])#test only
        crest_d=crest_module[index]+"_d"
        df.loc[:,crest_module[index]]=df_tmp.loc[:,crest_module[index]]
        df.loc[:,crest_d]=df_tmp.loc[:,"dst"]
    df.to_csv(_PATH_TO_OUTPUT+"df.csv")



if "production1" in test_functions:
    #naive_times = pd.DatetimeIndex(start='18/9/2018', end='19/9/2018', freq='1m')
    #Pyranometer CMP11_W01 measures global irradiance and is not on a ventilation unit.
    #Pyranometer CMP11_W07 measures global irradiance and is on a ventilation unit.
    #Pyranometer CMP11_W08 measures diffuse horizontal irradiance and is on a ventilation unit.    
    latitude, longitude, name, altitude, timezone,surface_tilt,surface_azimuth = (
    52.762463,-1.249695,'Loughborough',79,"GMT+0",35,180)   
    
    Crest = pvint.SolarLibrary(latitude=latitude,longitude=longitude,altitude=altitude,
    timezone=timezone,surface_tilt=surface_tilt,surface_azimuth=surface_azimuth)
                       
   
    sandia_modules = pvsys.retrieve_sam('SandiaMod')
    module = sandia_modules['SolarWorld_Sunmodule_250_Poly__2013_']
    #updated values for "Solarworld_SW245_mono" based on SW245 poly
    module["Isco"]=6.73
    module["Voco"]=33.5
    module["Impo"]=6.28
    module["Vmpo"]=27.1
    
    
    df_mms1 = pd.read_csv(filepath_or_buffer =_PATH_TO_INPUT+_FILE_MMS1,header=0)#,index_col="Row Labels") #,nrows=nrows)    
    df_mms2=pd.read_csv(filepath_or_buffer =_PATH_TO_INPUT+_FILE_MMS2,header=0)
    df_iv = pd.read_csv(_PATH_TO_INPUT+_FILE_IV,header=0)#,index_col="Row Labels") #,nrows=nrows)   

    df_iv=df_iv[(df_iv.second<3)] #remove delayed IV curves
        
    
    df_mms1["G_CMP11_PPUK"]=df_mms1.apply(lambda x:x["G_CMP11_PPUK"]*1000/8.8,axis=1)
    
    
    #df_mms1=df_mms1.iloc[43200:46800,:]
    
    df_days, df_sd_clear,df_s_clear =mdq.SolarData.irradianceflagging(SolarLibrary=Crest,
                          datetimeutc_values=df_mms1.loc[:,"TMSTAMP"].values,
                          irradiance_values=df_mms1.loc[:,"G_CMP11_PPUK"].values,
                          timeresolution=1,
                          series_number=0)
    
    df_mms1=df_mms1[(df_mms1["second"]==0)] #take only 00 seconds values after stabilisation analysis
    
    
    df_mms1.index=df_mms1["TMSTAMP"]
    df_mms2.index=df_mms2["tmstamp"]
    df_iv.index=df_iv["tmstamp_fm"]    
    intrsct = set.intersection(set(df_mms1.index), set(df_iv.index), set(df_mms2.index))
    df_mms1 = df_mms1[df_mms1.loc[:,"TMSTAMP"].isin(intrsct)]    
    df_iv = df_iv[df_iv.loc[:,"tmstamp_fm"].isin(intrsct)]    
    df_mms2 = df_mms2[df_mms2.loc[:,"tmstamp"].isin(intrsct)]    
    df=pd.DataFrame(index=intrsct)
    df["ghi_m"]=df_mms2["pyro_cmp11_w07"]   
    df["dhi_m"]=df_mms2["pyro_cmp11_w08"]
    df["dni_m"]=df_mms2["pyrheliometer_chp1_02_mms2"]
    df["poa_m"]=df_mms1["G_CMP11_PPUK"]
    df["Impp_m"]=df_iv["Impp"]
    df["Vmpp_m"]=df_iv["Vmpp"]
    df["Isc_m"]=df_iv["Isc"]
    df["Voc_m"]=df_iv["Voc"] 
    df["t_module"]=df_iv["ModuleTemperature1"]     


    def p(i,v):
        if (i>0) and (v>0): 
            p=i*v
        elif  (i<=0) or (v<=0):
            p=0
        return p
    
    df["Pmpp_m"]=df.apply(lambda x: p(x['Impp_m'],x['Vmpp_m']),axis=1)
   
    
    naive_times=pd.DatetimeIndex(intrsct)#datetimeindex after intersect 
    #times = .tz_localize(timezone)  
         
    
    
    df["datetime"]=naive_times
    df.index=naive_times
    df=df.sort_values(["datetime"],ascending=[1])  


    #df_compare, df_cs=mdq.SolarData.irradianceflagging(SolarLibrary=Crest,
    #                      datetimeutc_values=df.loc[:,"datetime"].values,
    #                      irradiance_values=df.loc[:,"poa_m"].values,
    #                      timeresolution=1,
    #                      series_number=0)    
    #datetime_cs = df_cs.loc[:,"datetime"].values    
    #df.loc[(df.loc[:,"datetime"].isin(datetime_cs)==False),"cloudy"]=True

    

    
    df.loc[(df.loc[:,"datetime"].isin(df_s_clear.loc[:,"datetime"].values)==True),"clear"]=True
    
    
    
    

    #df=df[(df.ghi_m>20)&(df.ghi_m>df.dhi_m)&(df.ghi_m>df.dni_m)] #requirerement monitoring filtering  

      
    values = Crest.getsolarseries(df.index,["aoi","apparentzenith","azimuth","airmassabsolute","extraradiation"])  
    #solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)        
    #aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
    #solpos['apparent_zenith'], solpos['azimuth']) 
    df["aoi"]=values["aoi"] #values only since time aware   
    #df["dni"]=df.apply(lambda x:(x["ghi"]-x["dhi"])/np.cos(np.radians(x["aoi"])),axis=1)    
    total_irrad = pvlib.irradiance.total_irrad(
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    apparent_zenith=values["apparentzenith"],
    azimuth=values["azimuth"],       
    dni=df.loc[:,'dni_m'].values, 
    ghi=df.loc[:,'ghi_m'].values, 
    dhi=df.loc[:,'dhi_m'].values,
    dni_extra= values['extraradiation'], 
    model='haydavies')#transposition algorithm
    #solpos.loc[:,'apparent_zenith'].values,
    #solpos.loc[:,'azimuth'].values,   
    df["poa_global"]=total_irrad["poa_global"]
    df["poa_direct "]=total_irrad["poa_direct"]
    df["poa_diffuse"]=total_irrad["poa_diffuse"]
    df["poa_sky_diffuse"]=total_irrad["poa_sky_diffuse"]

    #airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])
    #pressure = pvlib.atmosphere.alt2pres(altitude)
    #am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure)
    
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
    total_irrad['poa_direct'], total_irrad['poa_diffuse'],
    values["airmassabsolute"],values["aoi"],
    #am_abs,aoi, 
    module,reference_irradiance=1000)
    df["poa_effective"]=effective_irradiance
    
    dc_ei = pvlib.pvsystem.sapm(effective_irradiance,df['t_module'], module)
    dc_ei.columns = [str(col) + '_ei' for col in dc_ei.columns]
    df=pd.concat([df,dc_ei],axis=1)
    df["dp_ei"]=df.apply(lambda x:x["p_mp_ei"]-x["Pmpp_m"],axis=1)

    dc_poa = pvlib.pvsystem.sapm(df["poa_global"]/1000,df['t_module'], module)
    dc_poa.columns = [str(col) + '_poa' for col in dc_poa.columns]
    df=pd.concat([df,dc_poa],axis=1)
    df["dp_poa"]=df.apply(lambda x:x["p_mp_poa"]-x["Pmpp_m"],axis=1)


    df_indoor=pd.read_csv(filepath_or_buffer =_PATH_TO_INPUT+_FILE_INDOOR,header=0,nrows=16,
                          index_col="Row Labels")
    
    
    
    
    df["irr_ref"]=df.apply(lambda x:round(x["poa_m"]/50,0)*50,axis=1)    
    df["tmp_ref"]=df.apply(lambda x:round(x["t_module"]/5,0)*5,axis=1)
    
    def pmpp_indoor(i,t):
        #if (i==800) & (t==30):
        #    print("test")
        try:
            value=df_indoor.loc[int(t),str(int(i))]
        except:
            value=np.nan
        return value
    
    
    df["pmpp_indoor"]=df.apply(lambda x:pmpp_indoor(x["irr_ref"],x["tmp_ref"]),axis=1)
    
        
    df["dp_indoor"]=df.apply(lambda x:x["p_mp_poa"]-x["pmpp_indoor"],axis=1)
    

    df.to_csv(_PATH_TO_OUTPUT+"df.csv")

    print(_PATH_TO_OUTPUT)


if "pvsystems" in test_functions:
    naive_times = pd.DatetimeIndex(start='2015', end='2016', freq='1h')
    coordinates = [(30, -110, 'Tucson', 700, 'Etc/GMT+7'),
    (35, -105, 'Albuquerque', 1500, 'Etc/GMT+7'),
    (40, -120, 'San Francisco', 10, 'Etc/GMT+8'),
    (50, 10, 'Berlin', 34, 'Etc/GMT-1')]
    sandia_modules = pvsys.retrieve_sam('SandiaMod') #returns the Sandia Module database
    #https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/pvsystem.html#retrieve_sam    
    #Retrieve latest module and inverter info from a local file or the SAM website.
    sapm_inverters = pvsys.retrieve_sam('cecinverter') #returns the CEC Inverter database
    print(type(sandia_modules)) 
    print(type(sapm_inverters)) 
    sandia_modules.to_csv(_PATH_TO_OUTPUT+"modules.csv")
    sapm_inverters.to_csv(_PATH_TO_OUTPUT+"inverters.csv")
    
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    temp_air = 20
    wind_speed = 0
    
    energies = {}
    
    
    system = {'module': module, 'inverter': inverter,'surface_azimuth': 180}
    for latitude, longitude, name, altitude, timezone in coordinates:
        times = naive_times.tz_localize(timezone)
        system['surface_tilt'] = latitude
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        #dni_extra = pvlib.irradiance.get_extra_radiation(times)
        dni_extra = pvlib.irradiance.extraradiation(times)
        #airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
        airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])
        pressure = pvlib.atmosphere.alt2pres(altitude)
        #am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
        am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure)
        tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
        cs = pvlib.clearsky.ineichen(solpos['apparent_zenith'], am_abs, tl,
        dni_extra=dni_extra, altitude=altitude)
        aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
        solpos['apparent_zenith'], solpos['azimuth'])
        #total_irrad = pvlib.irradiance.get_total_irradiance
        total_irrad = pvlib.irradiance.total_irrad(system['surface_tilt'],
        system['surface_azimuth'],
        solpos['apparent_zenith'],
        solpos['azimuth'],
        cs['dni'], cs['ghi'], cs['dhi'],
        dni_extra=dni_extra,
        model='haydavies')#transposition algorithm     
        temps = pvlib.pvsystem.sapm_celltemp(total_irrad['poa_global'],
                                             wind_speed, temp_air)
        #Estimate cell and module temperatures per the Sandia PV Array
        #Performance Model (SAPM, SAND2004-3535), from the incident
        #irradiance, wind speed, ambient temperature, and SAPM module
        #parameters.        
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        total_irrad['poa_direct'], total_irrad['poa_diffuse'],
        am_abs, aoi, module)
        #Calculates the SAPM effective irradiance using the SAPM spectral
        #loss and SAPM angle of incidence loss functions.
        #Parameters: poa_direct, poa_diffuse, airmass_absolute, aoi, module, reference_irradiance
        # module : dict-like
        #A dict, Series, or DataFrame defining the SAPM performance
        #parameters. See the :py:func:`sapm` notes section for more details.
        #reference_irradiance : numeric, default 1000
        #Reference irradiance by which to divide the input irradiance.
        #...        
        #F1 = sapm_spectral_loss(airmass_absolute, module)
        #F2 = sapm_aoi_loss(aoi, module)    
        #E0 = reference_irradiance    
        #Ee = F1 * (poa_direct*F2 + module['FD']*poa_diffuse) / E0
        #FM 11/11/19: models for spectral & aoi loss to be checked
        dc = pvlib.pvsystem.sapm(effective_irradiance, temps['temp_cell'], module)
        #Use the :py:func:`sapm` function, the input parameters,
        #and ``self.module_parameters`` to calculate
        #Voc, Isc, Ix, Ixx, Vmp/Imp.
        #The Sandia PV Array Performance Model (SAPM) generates 5 points on a
        #PV module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
        #SAND2004-3535. Assumes a reference cell temperature of 25 C.
        
        #The modules in the Sandia module database contain these
        #coefficients, but the modules in the CEC module database do not.
        #Both databases can be accessed using :py:func:`retrieve_sam`.

        #[1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
        #Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
        #NM.    
        ac = pvlib.pvsystem.snlinverter(dc['v_mp'], dc['p_mp'], inverter)
        
        annual_energy = ac.sum()
        energies[name] = annual_energy
        
    print(energies)
        
    
    


if "timemaxzenithvariation" in test_functions:
    print("timemaxzenithvariation")
    datetimes = []
    for i in range(1,23):
        datetimes.append(dtt.datetime(2019,11,1,i,00))
        df = mdq.SolarData.timemaxzenithvariation(SolLib,datetimes)




"""
datetimes = pd.to_datetime(DATETIMES, format='%Y%m%d', errors='ignore')
datetimeindex_utc = pd.DatetimeIndex(datetimes,ambiguous='NaT',nonexistent='raise')
#datetimeindex_utc=datetimeindex_tz.tz_convert('UTC')

df_rts=SolLib.getsunrisesettransit(datetimeindex_utc.date)
ss_rise = SolLib.getsolarseries(df_rts.loc[:,"sunrise"],outputs=["datetime"])
ss_transit = SolLib.getsolarseries(df_rts.loc[:,"suntransit"],outputs=["datetime"])
ss_set = SolLib.getsolarseries(df_rts.loc[:,"sunset"],outputs=["datetime"])

datetimeindex_utc=datetimeindex_utc.append(ss_rise.index)
datetimeindex_utc=datetimeindex_utc.append(ss_transit.index)
datetimeindex_utc=datetimeindex_utc.append(ss_set.index)
datetimeindex_utc=datetimeindex_utc.drop_duplicates(keep='first')

df_az = SolLib.getsolarseries(datetimeindex_utc,outputs=["apparentzenith"])
dt_utc = df_az.index.tz_localize('UTC')
df_az.loc[:,"dt_tz"]=dt_utc.tz_convert('Europe/London')


dfout = pd.DataFrame(columns=['dt','dt_tz','z_s','m_e'])


for index in df_az.index:   
    for s in range(0,28800):        
        def solarseriessingle(datetime:dtt.datetime):
            datetimes=pd.DatetimeIndex([datetime,datetime])
            df_tmp=SolLib.getsolarseries(datetimes,"apparentzenith")
            return df_tmp.iloc[0,1]
        apparent_zenith_end=solarseriessingle(index+dtt.timedelta(seconds=s))
        delta=abs(df_az.loc[index,"apparent_zenith"]-apparent_zenith_end)
        if delta>TILT_DELTA:break  
    
    
    dfout = dfout.append({
    'dt':index,
    'dt_tz':str(df_az.loc[index,"dt_tz"]), #append tz bug https://github.com/pandas-dev/pandas/issues/12985
    'z_s':round(df_az.loc[index,"apparent_zenith"],2),
    'm_e':round(s/60,2)},
    ignore_index=True)

print(dfout)

PATH_TO_RESULTS = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/test.csv"

dfout.to_csv(path_or_buf=PATH_TO_RESULTS)




print("AOI VARIATION CHECK")
LATITUDE= -1.248474; LONGITUDE= 52.758264
SolLib=pvint.SolarLibrary(latitude=LATITUDE,longitude=LONGITUDE)
datetimes=['201910021200','201910021330']
datetimes = pd.to_datetime(datetimes, format='%Y%m%d', errors='ignore')
datetimeindex_tz = pd.DatetimeIndex(datetimes,timezone='Etc',ambiguous='NaT',nonexistent='NaT')


labels=["start","apparent_zenith_start","zenith_start","end","apparent_zenith_end","zenith_end"]

ss_start = SolLib.getsolarseries(datetimeindex_tz,outputs=list(["zenith","apparentzenith","azimuth"]))
ss_end = SolLib.getsolarseries(datetimeindex_tz+dtt.timedelta(hours=1),outputs=list(["zenith","apparentzenith","azimuth"]))

ss_end.index = ss_start.index


for index in ss_start.index:
    values=[ss_start.loc[index,"datetime"],
            ss_start.loc[index,"apparent_zenith"],
            ss_start.loc[index,"zenith"],
            ss_start.loc[index,"datetime"]+dtt.timedelta(hours=1),
            ss_end.loc[index,"apparent_zenith"],
            ss_end.loc[index,"zenith"]]    

    print(dict(zip(labels,values)))




print("AZIMUTH SOLVER")
LATITUDE= 52.758264
LONGITUDE= -1.248474

SolLib=pvint.SolarLibrary(latitude=LATITUDE,longitude=LONGITUDE)

date = '20191002'
date2 = '20191102'

datetimes = pd.to_datetime([date,date2], format='%Y%m%d', errors='ignore')

datetimeindex_tz = pd.DatetimeIndex(datetimes,timezone='Etc',ambiguous='NaT',nonexistent='NaT')

df=SolLib.getsunrisesettransit(datetimeindex_tz)

ss_rise = SolLib.getsolarseries(df.loc[:,"sunrise"],outputs=list(["zenith","apparentzenith","azimuth"]))
ss_transit = SolLib.getsolarseries(df.loc[:,"suntransit"],outputs=list(["zenith","apparentzenith","azimuth"]))
ss_set = SolLib.getsolarseries(df.loc[:,"sunset"],outputs=list(["zenith","apparentzenith","azimuth"]))

sun_zenith = ss_transit.loc[:,"apparent_zenith",].values[0]
sun_azimuth = ss_transit.loc[:,"azimuth",].values[0]
plane_zenith = sun_zenith
aoi = 0

def test_equation(p_a):
    s_z = sun_zenith
    s_a = sun_azimuth
    p_z = plane_zenith    
    f=(m.cos(s_z)*m.cos(p_z)+
       m.sin(s_z)*m.sin(s_a)*m.sin(p_z)*m.sin(s_a+p_a)+
       m.sin(s_z)*m.cos(s_a)*m.sin(p_z)*m.cos(s_a+p_a))-m.cos(aoi)
    return f

guess = 0 
p_a = fsolve(test_equation,guess)






labels=["sun_apparent_zenith","sun_azimuth","plane_azimuth"]
values=[sun_zenith,sun_azimuth,p_a[0]]

print(dict(zip(labels,values)))

sun_zenith = ss_set.loc[:,"apparent_zenith",].values[0]
sun_azimuth = ss_set.loc[:,"azimuth",].values[0]
aoi = m.radians(95)

guess = 0 
p_a = fsolve(test_equation,guess)

values=[sun_zenith,sun_azimuth,p_a[0]]

print(dict(zip(labels,values)))






print("OPTIMIZE IRRADIANCE MAP SINGLE CURVATURE RADIUS BACK LAMP")
#same results
d0 = 7420 #distance from extreme of lamp not inside. 
y0 = 8640.06
z0 = 13244.54
i0 = 1217.09

def test_func(X,r):
    dy0,dz0=X    
    f=(i0*
    ((r+d0)/((r+d0)**2+(dy0)**2)**0.5)*
    ((r+d0)/((r+d0)**2+(dz0)**2)**0.5)*
    ((r+d0)**2/(r+d0)**2+dy0**2+dz0**2))
    #((d0)**2/(d0)**2+dy0**2+dz0**2))#same result if front lamp
    return f

dy0=[1359.4,-1640.6,1359.4,1359.4,-1640.6,4359.4,-1640.6,-4640.6,4359.4,4359.4,-4640.6,1359.4,-4640.6,-1640.6,7359.4,4359.4,7359.4,1359.4,7359.4,-1640.6,-4640.6,4359.4,7359.4,-4640.6,1359.4,-1640.6,10359.4,4359.4,10359.4,7359.4,10359.4,-4640.6,7359.4,10359.4,10359.4,10359.4]
dz0=[-244.540000000001,-244.540000000001,-3244.54,2755.46,-3244.54,-244.540000000001,2755.46,-244.540000000001,-3244.54,2755.46,-3244.54,-6244.54,2755.46,-6244.54,-244.540000000001,-6244.54,-3244.54,5755.46,2755.46,5755.46,-6244.54,5755.46,-6244.54,5755.46,-9244.54,-9244.54,-244.540000000001,-9244.54,-3244.54,5755.46,2755.46,-9244.54,-9244.54,-6244.54,5755.46,-9244.54]
i=[1188.14491331577,1168.22409439086,1096.15248531103,1065.54633629322,1053.26618081331,1035.5636190176,1016.91866040229,935.756546616554,930.348638713359,909.934873759746,859.074442327022,808.054528355598,799.282988727092,768.357874631881,738.230173230171,696.445748090744,690.70437669754,672.218900620937,657.705989599227,645.266351580619,643.033596038818,581.023885309696,538.616028428077,528.887593567371,525.060012638568,505.762625455856,479.303022444248,478.027162134647,455.699606716632,445.028775036334,422.54173707962,422.06328946352,376.465781807899,375.030438959598,312.846747279167,273.15009355545]
#irradiance without shadowing, offset should be corrected ??
#initial 0.03 than decreasing due to temperature ?
#calibration should be corrected too ?

r0=10000

params, params_covariance = optimize.curve_fit(test_func,(dy0,dz0),i,r0)

print(params,params_covariance)



print("TEST")
def test_func(X,rz,ry):
    dz0,dy0=X    
    f=dz0*rz+dy0*ry
    return f
dz0=[1,1,2,2]
dy0=[1,2,1,2]
i=[3,5,4,6]

rz0=1
ry0=1

params, params_covariance = optimize.curve_fit(f=test_func,xdata=(dz0,dy0),ydata=i,p0=(rz0,ry0))

#uncertainty of data to be considered see curve_fit

print(params,params_covariance)



print("OPTIMIZE IRRADIANCE MAP DOUBLE CURVATURE RADIUS")
d0 = 7420
i0 = 1217.09

def test_func(X,ry,rz):
    dy0,dz0=X    
    f=i0*((rz+d0)**2/((rz+d0)**2+(dz0)**2)**0.5*    
         (ry+d0)**2/((ry+d0)**2+(dy0)**2)**0.5*
         d0**2/(d0**2+dz0**2+dy0**2))
    return f

dy0=[1359.4,-1640.6,1359.4,1359.4,-1640.6,4359.4,-1640.6,-4640.6,4359.4,4359.4,-4640.6,1359.4,-4640.6,-1640.6,7359.4,4359.4,7359.4,1359.4,7359.4,-1640.6,-4640.6,4359.4,7359.4,-4640.6,1359.4,-1640.6,10359.4,4359.4,10359.4,7359.4,10359.4,-4640.6,7359.4,10359.4,10359.4,10359.4]
dz0=[-244.540000000001,-244.540000000001,-3244.54,2755.46,-3244.54,-244.540000000001,2755.46,-244.540000000001,-3244.54,2755.46,-3244.54,-6244.54,2755.46,-6244.54,-244.540000000001,-6244.54,-3244.54,5755.46,2755.46,5755.46,-6244.54,5755.46,-6244.54,5755.46,-9244.54,-9244.54,-244.540000000001,-9244.54,-3244.54,5755.46,2755.46,-9244.54,-9244.54,-6244.54,5755.46,-9244.54]
i=[1188.14491331577,1168.22409439086,1096.15248531103,1065.54633629322,1053.26618081331,1035.5636190176,1016.91866040229,935.756546616554,930.348638713359,909.934873759746,859.074442327022,808.054528355598,799.282988727092,768.357874631881,738.230173230171,696.445748090744,690.70437669754,672.218900620937,657.705989599227,645.266351580619,643.033596038818,581.023885309696,538.616028428077,528.887593567371,525.060012638568,505.762625455856,479.303022444248,478.027162134647,455.699606716632,445.028775036334,422.54173707962,422.06328946352,376.465781807899,375.030438959598,312.846747279167,273.15009355545]
#irradiance without shadowing, offset should be corrected ??
#initial 0.03 than decreasing due to temperature ?
#calibration should be corrected too ?

rz0=10000
ry0=10000

params, params_covariance = optimize.curve_fit(f=test_func,xdata=(dy0,dz0),ydata=i,p0=(ry0,rz0))

#uncertainty of data to be considered see curve_fit

print(params,params_covariance)

print("OPTIMIZE RESPONSE MAP")
d0 = 7640
r0 = params[0]
ds = 1500 #sensor distance

def test_func(n,dx,dy,i): #reference is first sensor                      
    f=i*((r0+d0)**4/((r0+d0)**2+(dx+n*ds)**2)**0.5/((r0+d0)**2+(dy)**2)**0.5/
      ((r0+d0)**2+(dx+n*ds)**2+dy**2))
    return f

n = [0,1,2,3]
i = [850.71,902.31,916.07,883.22]

dx0=-22
dy0=+30

params, params_covariance = optimize.curve_fit(test_func,n,i,(dx0,dy0,i0))


print(int(params[0]),int(params[1]),int(params[2]),params_covariance)

print("OPTIMIZE RESPONSE MAP assumed irradiance")
d0 = 7640
r0 = params[0]
ds = 1500 #sensor distance
i0 = 1148

def test_func(n,dx,dy): #reference is first sensor                      
    f=i0*((r0+d0)**4/((r0+d0)**2+(dx+n*ds)**2)**0.5/((r0+d0)**2+(dy)**2)**0.5/
      ((r0+d0)**2+(dx+n*ds)**2+dy**2))
    return f

n = [0,1,2,3]
i = [850.71,902.31,916.07,883.22]

dx0=-2200
dy0=+3000

params, params_covariance = optimize.curve_fit(test_func,n,i,(dx0,dy0))


print(int(params[0]),int(params[1]),params_covariance)

"""

#MANAGING EXCEPTION
#class MyError(Exception):
#    def __init__(self, value):
#        self.value = value
#    def __str__(self):
#        return repr(self.value)
#try:
#    raise MyError("a"*2)
#except MyError as e:
#    print('My exception occurred, value:', e.value)


