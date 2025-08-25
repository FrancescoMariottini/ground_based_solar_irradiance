# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:17:12 2018
@author: wsfm
HISTORY
23/7/18: Wacoom
10/10/18: renamed from "crest" as "crest_wacom"
DEV NOTES: should be generalised as extraction tool based on info section and relative positions (using vocabularies)

"""

""" IMPORTING MODULES """
#import pandas for dataframe operations
import pandas as pd
#import datetime for format recognition
import datetime as dt
#import os for file searching
import os 

      

"""GLOBAL VARIABLES (default values) """

PATH_TO_INPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/_crest_pyranometer_calibration_indoor/"

PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/"

FILES_LIST = [f for f in os.listdir(PATH_TO_INPUT) if (f.endswith(".txt"))]

CALIBRATION_INFO_SECTIONS = ["WACOM Software - Measurement File",
           "Sample details",
           "Reference Pyranometer Details",
           "Pyranometer Calibration Settings",
           "Fluke 884X Measurement Settings",
           "Wacom Irradiance Calibration Record Data",
           "Wacom Lamp Calibration Record Data",
           "Pyranometer Calibration Final Results",
           "Pyranometer Calibration Sub-Meas Results",
           "Pyranometer Calibration Reference Meas Results",
           "Pyranometer Calibration Sample Meas Results"]

MAIN_INFO_HEADERS = ["filename","sample_id","sample_type","sample_details", #120620 sample further details added
                       "reference_id","reference_type","reference_sensitivity [uV/Wm-2]",
                       "wacom_id","wacom_type","meas_temp","meas_time",
                       "response_av [uV/Wm-2]", "response_std [uV/Wm-2]",
                       "irradiance_av [uV/Wm-2]", "irradiance_std [uV/Wm-2]"]

SUBMEASURES_HEADERS = ["filename","Sub-Meas ID","Sensitivity [uV/Wm-2]","Irradiance [uV/Wm-2]"]



def _all_sections_positions(path_to_input,
                            files_list=FILES_LIST,calibration_info_sections=CALIBRATION_INFO_SECTIONS,path_to_output=None,time_in_filename=False):
    """ Return types of information per section, examples of values and section positions based on assumed types of info """                          
    #DEV NOTES 23/7/18: similar to the one in coenergy
    #DEV NOTES 8/6/18: faster (0:00:00.368514 vs 0:00:03.028550) than _files_info    
    rows_list = []
    info_types = ('filename','info_line','info_value','start_line','sample_line','end_line')
    for fname in files_list:
        file_path = path_to_input+fname
        fhand = open(file_path ) 
        #DEV NOTE: test line:
        #print(fhand)
        #inizialition 
        sample_ck = False
        start_line = None
        sample_line = None
        info_line = None
        info_value = None
        info_value_previous = "" 
        
        for index, line in enumerate(fhand):
            value = line.rstrip()  
            if value != "":      
                if value.startswith(tuple(calibration_info_sections)):
                    if sample_ck == True:
                        #create lines after getting all values
                        #DEV NOTES 26/7/18: +1 to getllast row
                        sections_tmp = {
                        info_types[0]:fname,
                        info_types[1]:info_line, 
                        info_types[2]:info_value,
                        info_types[3]:start_line,
                        info_types[4]:sample_line,
                        info_types[5]:int(index)+1}
                        rows_list.append(sections_tmp)
                    sample_ck = False
                    #assigning for new line 
                    info_line = str(index)
                    info_value = str([i for i in list(calibration_info_sections) if (i in value)][0])
                    #DEV NOTE: not tested info_value = str([i for i in list(calibration_info_sections) if value.startswith(i)])
                    #Dealing with old format (2x "Meas Results")
                    if (info_value == info_value_previous) and (info_value_previous == "Pyranometer Calibration Reference Meas Results"):
                        info_value = "Pyranometer Calibration Sample Meas Results"
                   
                    info_value_previous = info_value 
                elif sample_ck == False:
                    sample_ck = True
                    start_line = str(index)
                    sample_line = str(value)
        #create for last line
        sections_tmp = {
        info_types[0]:fname,
        info_types[1]:info_line, 
        info_types[2]:info_value,
        info_types[3]:start_line,
        info_types[4]:sample_line,
        info_types[5]:int(index)+1}
        rows_list.append(sections_tmp)    
    rowsdf = pd.DataFrame(rows_list,columns=info_types)     
    if path_to_output!=None:
        if time_in_filename==False:
            time_string = ""
        elif time_in_filename==True:
            time_string = "_"+ str(dt.datetime.strftime(dt.datetime.now(),"%y%m%d%H%M"))
        rowsdf.to_csv(path_to_output+"calibration_template"+time_string+".csv")      
    return rowsdf



CALIBRATION_INFO_SECTIONS = ["WACOM Software - Measurement File",
           "Sample details",
           "Reference Pyranometer Details",
           "Pyranometer Calibration Settings",
           "Fluke 884X Measurement Settings",
           "Wacom Irradiance Calibration Record Data",
           "Wacom Lamp Calibration Record Data",
           "Pyranometer Calibration Final Results",
           "Pyranometer Calibration Sub-Meas Results",
           "Pyranometer Calibration Reference Meas Results",
           "Pyranometer Calibration Sample Meas Results"]

def _main_info_to_dataframe(all_sections_positions,files_list,path_to_input,
                            main_info_headers=MAIN_INFO_HEADERS,path_to_output=None,time_in_filename=False):
    """ based on sections positions, return main information through a dataframe """
    #datetime format for meas time 
    dt_format="%y/%m/%d-%H:%M:%S"
    #table initialisation     
    df_information_all = pd.DataFrame(columns=main_info_headers)
    for fname in files_list:   
        file_path = str(path_to_input) + str(fname) 
        file_sections_positions = all_sections_positions[(all_sections_positions.filename==fname)]
        if file_sections_positions.shape[0] == 0:            
            print(file_path+" not found")
        elif (file_sections_positions.shape[0] != 0):   
            #extracting indexes for Sample details:
            id_info = CALIBRATION_INFO_SECTIONS[1]
            sample_id_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)
            sample_type_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)+3 
            sample_details_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)+6#120620 added to look for maintenance
            #extracting index for Reference Pyranometer Details
            id_info =  CALIBRATION_INFO_SECTIONS[2]
            ref_id_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)
            ref_type_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)+2
            ref_sensitivity_factor_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)+4
            #extracting index for "Wacom Lamp Calibration Record Data
            id_info = CALIBRATION_INFO_SECTIONS[6]                    
            start6 = int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)
            wacom_id_index=start6+1    
            wacom_type_index=start6+3            
            meas_temp_index=start6+18   
            meas_time_index=start6+19   
             #extracting index for Pyranometer Calibration Final Results
            id_info = CALIBRATION_INFO_SECTIONS[7]
            #2 values per row
            responses_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)
            irradiances_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)+1  
            #row list
            rows_list = list([sample_id_index,sample_type_index,sample_details_index,
                              ref_id_index,ref_type_index,ref_sensitivity_factor_index,
                             wacom_id_index,wacom_type_index,
                             meas_temp_index, meas_time_index,                             
                             responses_index,irradiances_index])            
            #file opening
            fhand = open(file_path)
            for index, line in enumerate(fhand):
                if index in rows_list:
                    line_elements = line.split()
                    if index == sample_id_index:
                        sample_id = line_elements[2:3][0]                                             
                    elif index == sample_type_index:
                        sample_type = line_elements[2:3][0]    
                    elif index == sample_details_index:
                        sample_details = line_elements[3:]#120620 added 
                    elif index == ref_id_index: 
                        ref_id = line_elements[2:3][0]    
                    elif index == ref_type_index:
                        ref_type = line_elements[2:3][0]    
                    elif index == ref_sensitivity_factor_index:
                        ref_sensitivity_factor = line_elements[4:5][0]   
                    elif index == wacom_id_index:
                        #alternative for empty information
                        if len(line_elements) > 2: 
                            wacon_id = line_elements[2:3][0]   
                        elif len(line_elements) <=2:
                            wacon_id = None
                    elif index == wacom_type_index:
                        if len(line_elements) > 2:  
                            wacon_type = line_elements[2:3][0]    
                        elif len(line_elements) <=2:
                            wacon_type = None
                    elif index == meas_temp_index:
                        if len(line_elements) > 4:
                            meas_temp = line_elements[4:5][0]
                        elif len(line_elements) <=4:
                            meas_temp = None
                    elif index == meas_time_index:
                        if len(line_elements) > 3:
                            meas_time = line_elements[3:4][0]
                            meas_time = dt.datetime.strptime(str(meas_time),dt_format)
                        elif len(line_elements) <=3:
                            meas_time = None
                    elif index == responses_index:
                        response_av = str(line_elements[3:4][0]).replace('[uV/Wm-2]','')
                        response_std = line_elements[8:9][0]  
                    elif index == irradiances_index:
                        irradiance_av = str(line_elements[3:4][0]).replace('[Wm-2]','')
                        irradiance_std = line_elements[8:9][0] 
                        #appending information
                        df_information_all = df_information_all.append(
                        {main_info_headers[0]:fname,
                        main_info_headers[1]:sample_id,
                        main_info_headers[2]:sample_type,
                        main_info_headers[3]:sample_details,
                        main_info_headers[4]:ref_id,
                        main_info_headers[5]:ref_type,
                        main_info_headers[6]:ref_sensitivity_factor,
                        main_info_headers[7]:wacon_id,
                        main_info_headers[8]:wacon_type,
                        main_info_headers[9]:meas_temp,
                        main_info_headers[10]:meas_time,
                        main_info_headers[11]:response_av,
                        main_info_headers[12]:response_std,
                        main_info_headers[13]:irradiance_av,
                        main_info_headers[14]:irradiance_std
                        },ignore_index=True)  
                        
    if  path_to_output!=None: 
        if time_in_filename==False:
            time_string = ""
        elif time_in_filename==True:
            time_string = "_"+ str(dt.datetime.strftime(dt.datetime.now(),"%y%m%d%H%M"))
        df_information_all.to_csv(path_to_output+"calibration_information"+time_string+".csv")
    return df_information_all



def _submeasures_to_dataframe(all_sections_positions,path_to_input,files_list,
                              submeasures_headers=SUBMEASURES_HEADERS,path_to_output=None,time_in_filename=False):
    id_info = "Pyranometer Calibration Sub-Meas Results"
    df_measurement_all = pd.DataFrame(columns=submeasures_headers)
    for fname in files_list:
        file_path = str(path_to_input) + str(fname)      
        file_sections_positions = all_sections_positions[(all_sections_positions.filename==fname)]
        #next iteration if file not found 
        if file_sections_positions.shape[0] == 0:
            print(file_path+" not found")
        elif (file_sections_positions.shape[0] != 0):  
            #DEV NOTE 24/7/18: not clear why reference pandas.read_csv <> open file path
            
            #print(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)  
            #print("/n")
            #print(file_sections_positions[(file_sections_positions.info_value==id_info)]["end_line"].values) 
            
            
            start_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)  
            end_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["end_line"].values) 
            
            
            #DEV NOTE 24/7/18: file opening not necessary for all if rows already known 
            fhand = open(file_path)
            for index, line in enumerate(fhand):
                line_elements = line.split()
                #impose lengtht 3 to the list of elements as filter 
                if len(line_elements)==3 and index > start_index and index <= end_index:
                    #DEV NOTES 24/7/18: ignore index necessary to append a series 
                    df_measurement_all = df_measurement_all.append(
                    {submeasures_headers[0]:fname,
                     submeasures_headers[1]:line_elements[0:1][0],
                     submeasures_headers[2]:line_elements[1:2][0],
                     submeasures_headers[3]:line_elements[2:3][0]},ignore_index=True)
    if path_to_output!=None: 
        if time_in_filename==False:
            time_string = ""
        elif time_in_filename==True:
            time_string = "_"+ str(dt.datetime.strftime(dt.datetime.now(),"%y%m%d%H%M"))
        df_measurement_all.to_csv(path_to_output+"calibration_submeasures"+time_string+".csv")
    return df_measurement_all



def _results_to_dataframe(all_sections_positions,path_to_input,files_list,
                          path_to_output=None,time_in_filename=False):
    """ extract and reorder calibration results from txt files """
    """ based on defined info_positions (see related module) """
    """ info should be same type and positions """  
    """ note that voltage is in mV not uV (factor 1000 found) """
    #DEV NOTE 23/7/18: based on coenergy file (and improved?). info_id added as identifier
    #initialise empty dataframe    
    df_measurement_all = pd.DataFrame(columns=["filename","parameter","datetime","value"])
    for fname in files_list:
        file_path = str(PATH_TO_INPUT) + str(fname)      
        file_sections_positions = all_sections_positions[(all_sections_positions.filename==fname)]
        #next iteration if file not found 
        if file_sections_positions.shape[0] == 0:
            print(file_path+" not found")
            continue
        elif (file_sections_positions.shape[0] != 0):
            #extracting information for the reference 
            id_info = "Pyranometer Calibration Reference Meas Results"    
            ref_start_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)-1  
            ref_end_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["end_line"].values)
            id_info = "Pyranometer Calibration Sample Meas Results" 
            sample_start_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["start_line"].values)-1  
            sample_end_index=int(file_sections_positions[(file_sections_positions.info_value==id_info)]["end_line"].values)
            #DEV NOTE: check
            #print(start_line)
            fhand = open(file_path)
            #initialise empty lists
            datetimes_list = list()
            values_list = list()            
            for index, line in enumerate(fhand):
               #split line into elements
               line_elements = line.split()
               #check until end_index to not exclude last line 
               if line != None and index > ref_start_index and index <= ref_end_index:
                   for item in line_elements:
                       try:
                           dt_format="%y/%m/%d-%H:%M:%S.%f"
                           datetimes_list.append(dt.datetime.strptime(str(item),dt_format))
                       except:
                           try:
                               values_list.append(float(item))
                           except:
                               if values_list == []:
                                   parameter_tmp = item                                
                               if datetimes_list != [] and values_list != []:
                                   #creation of tmp array
                                   df_tmp = pd.DataFrame({
                                   'filename':[fname for x in range(len(datetimes_list))],
                                   'parameter':[parameter_tmp for x in range(len(datetimes_list))],
                                   'datetime':datetimes_list,
                                   'value':values_list})
                                   #appending to df measurements 
                                   df_measurement_all = df_measurement_all.append(df_tmp)
                                   # clearing tmp lists 
                                   datetimes_list = list()
                                   values_list = list() 
               elif line != None and index > sample_start_index and index <= sample_end_index:
                   for item in line_elements:
                       try:
                           dt_format="%y/%m/%d-%H:%M:%S.%f"
                           datetimes_list.append(dt.datetime.strptime(str(item),dt_format))
                       except:
                           try:
                               values_list.append(float(item))
                           except:
                               if values_list == []:
                                   parameter_tmp = item                                
                               if datetimes_list != [] and values_list != []:
                                   #creation of tmp array
                                   df_tmp = pd.DataFrame({
                                   'filename':[fname for x in range(len(datetimes_list))],
                                   'parameter':[parameter_tmp for x in range(len(datetimes_list))],
                                   'datetime':datetimes_list,
                                   'value':values_list})
                                   #appending to df measurements 
                                   df_measurement_all = df_measurement_all.append(df_tmp)
                                   # clearing tmp lists 
                                   datetimes_list = list()
                                   values_list = list()   
            #appending values for last series at end of file
            df_tmp = pd.DataFrame({
            'filename':[fname for x in range(len(datetimes_list))],
            'parameter':[parameter_tmp for x in range(len(datetimes_list))],
            'datetime':datetimes_list,
            'value':values_list})
            #appending to df measurements 
            df_measurement_all = df_measurement_all.append(df_tmp)
    if path_to_output!=None: 
        if time_in_filename==False:
            time_string = ""
        elif time_in_filename==True:
            time_string = "_"+ str(dt.datetime.strftime(dt.datetime.now(),"%y%m%d%H%M"))
        df_measurement_all.to_csv(path_to_output+"calibration_results"+time_string+".csv")
    return df_measurement_all




