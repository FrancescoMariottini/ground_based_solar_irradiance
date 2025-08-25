# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:37:40 2018
objective to make meteonorm datasets easier to be used

@author: wsfm
"""

#all radiation are short wave unless specified
#not in the manual if Az is azimuth
#not in the manual if mi is minute
#not in the manual if Gcs is clear sky

meteonorm_codes={
'ST':'true_solar_time',
'y':'year',
'm':'month',
'dm':'day_of_month',
'h':'hour_of_day',
'G_Gref':'rad_reflected_ground',
'G_Gex':'rad_extraterrestrial',
'G_Bh':'rad_direct',
'G_Gh':'rad_global',
'G_Gk':'rad_global_inclined',
'G_Dh':'rad_diffuse',
'G_Dk':'rad_diffuse_inclined',
'Az':'azimuth',
'hs':'altitude',
'RH':'humidity_relative',
'dy':'day_of_year',
'Ta':'temperature',
'mi': 'minute',
'FF': 'wind_speed',
'G_Gcs':'global_clear_sky'
}

#for key,value in meteonorm_dict.items():
#    print(key,value)



