# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:40:52 2019

Overall file creating graphs for PhD02

@author: wsfm
"""

import pvdata.executable.irradiance_comparison as ir
import pvdata.executable.calibrations_analysis as cag

#import matplotlib.pyplot as plt

#path to thesis graphs folder
_PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_PhD_Thesis_Draft_2Y/__Outputs/PhD02/"
#30/4/20 process descriptions and related figures to be reviewed


"""
#_FIGURE=['f07','f09','f10','f11']
#_PROCESS_FIGURES=['f12']
#_PROCESS_FIGURES=['f07','f09','f10','f11','f12','f13']
_PROCESS_NAMES=[
"crest ground 180824_29 error",
"crest ground 180824_29 error tcorrected",
"crest roof 180825 irradiances",
"crest roof 180825 hour error",
"crest roof almost clear days error month",
"eurac abd error 2015 month",
"eurac abd error 2016 month",
"crest roof months all error"]
"""



_PROCESS_NAMES=["crest ground 180824_29 error"]
_TEST=False
_MARKERSIZE=1
_PROCESS=True
_ERR_ABS=True
_ERR_COS=True
_ymin=[-100,-50]
_ymax=[500,50]
#process="crest ground 180824_29 error absolute"
#process="crest roof almost clear days error month cosine"
#process="eurac abd error 2016 month absolute"

#ylim=_YLIM_ABS
#df = ir.run(process,test=_TEST,path_to_output=_PATH_TO_OUTPUT,ylim=ylim) 


_IRRADIANCE_PROCESSES=[
"crest ground 180824_29 error",
"crest ground 180824_29 error tcorrected",
"crest roof 180825 irradiances",
"crest roof 180825 hour error",
"crest roof almost clear days error month",
"eurac abd error 2015 month",
"eurac abd error 2016 month",
"crest roof months all error"]

_CALIBRATION_PROCESSES=["crest indoor time response", #comparison of time response
                        "calibration outdoor eurac excel series finding iso CH4",
                        "dates flagging series finding excel calibration outdoor eurac CH7 cs check"
                        ]

#process = 
#process = "calibration eurac sql"
#process = "crest indoor calibration"

_CALIBRATION_PROCESSES=["calibration outdoor eurac excel series finding CH4 dates flagging",
                        "calibration outdoor eurac excel series finding CH5 dates flagging",
                        "calibration outdoor eurac excel series finding CH6 dates flagging",
                        "calibration outdoor eurac excel series finding CH7 dates flagging",
                        "calibration outdoor eurac excel series finding CH4 iso",
                        "calibration outdoor eurac excel series finding CH5 iso",
                        "calibration outdoor eurac excel series finding CH6 iso",
                        "calibration outdoor eurac excel series finding CH7 iso",
                        "calibration outdoor eurac filter excel series finding CH4",
                        "calibration outdoor eurac filter excel series finding CH5",
                        "calibration outdoor eurac filter excel series finding CH6",
                        "calibration outdoor eurac filter excel series finding CH7"]

_CALIBRATION_PROCESSES=["calibration outdoor eurac excel series finding CH4 dates flagging",
                        "calibration outdoor eurac excel series finding CH4 dates flagging uncertainty",
                        "calibration outdoor eurac excel series finding CH7 dates flagging",
                        "calibration outdoor eurac excel series finding CH7 dates flagging uncertainty"]

_CALIBRATION_PROCESSES=["calibration outdoor eurac excel series finding CH4 dates flagging"]

_IRRADIANCE_PROCESSES=[]

for value in list(_IRRADIANCE_PROCESSES):
    print(value)
    df_results = ir.run(process=value,test=_TEST,path_to_output=_PATH_TO_OUTPUT) 
    
for value in list(_CALIBRATION_PROCESSES):
    print(value)
    df_results = cag.run(process=value,test=_TEST,path_to_output=_PATH_TO_OUTPUT) 
       
    

"""
for value in list(_PROCESS_NAMES):
    process=value
    if _ERR_ABS==True and _ERR_COS==True:
        process=process+" "+"absolute"
        process=process+" "+"cosine"
        df_results = ir.run(process,test=_TEST,path_to_output=_PATH_TO_OUTPUT,ymin=_ymin,ymax=_ymax)
    elif _ERR_ABS==False or _ERR_COS==False:   
        if _ERR_ABS==True: df_results = ir.run(process,test=_TEST,path_to_output=_PATH_TO_OUTPUT,ymin=_ymin[0],ymax=_ymax[1])
        if _ERR_COS==True: df_results = ir.run(process,test=_TEST,path_to_output=_PATH_TO_OUTPUT,ymin=_ymin[0],ymax=_ymax[1])
    plt.close()
"""

