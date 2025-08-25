# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:49:49 2018
@author: wsfm
analysis of raw datasets (built for a specific datasets)
HISTORY
Last modified 23/7/18: "rawformat" renamed as 


"""


import os
import pandas as pd
import numpy as np

#for testing time
import datetime as dtt
#for basic related operations 
import sqlalchemy as sqla



"""POSTGRESQL CONSTANTS """ 
_CONNECTION:str
 #connstring example: postgresql+psycopg2://postgres@localhost/[name_of_database]
_SCHEMA="pv_monitoring"
#id_device big serial key     
_DEVICE_DTYPES={'udid':sqla.types.VARCHAR,'upid':sqla.types.VARCHAR,
'config':sqla.types.INTEGER,'colcount':sqla.types.INTEGER,
'evcount':sqla.types.INTEGER,'active':sqla.types.BOOLEAN,
'virtual':sqla.types.BOOLEAN,'changed':sqla.types.TIMESTAMP,
'loggerid':sqla.types.INTEGER,'devtype':sqla.types.VARCHAR,
'alias':sqla.types.VARCHAR,'vendor':sqla.types.VARCHAR,
'created':sqla.types.TIMESTAMP,'ppeak':sqla.types.FLOAT,
'sparent':sqla.types.VARCHAR,'eparent':sqla.types.VARCHAR,
'deviceid':sqla.types.INTEGER,'devicegroup':sqla.types.VARCHAR,
'devicetype':sqla.types.VARCHAR,'has_cdb_alias':sqla.types.BOOLEAN,
'customalias':sqla.types.VARCHAR,'cyclic_table_att_num':sqla.types.INTEGER}
#record big serial key
_VALUE_DTYPES={'id_device':sqla.types.BIGINT,'uvid':sqla.types.VARCHAR,
'udid':sqla.types.VARCHAR,'spalte':sqla.types.VARCHAR,
'datatype':sqla.types.VARCHAR,'storagetype':sqla.types.VARCHAR,
'changed':sqla.types.TIMESTAMP,'valtype':sqla.types.VARCHAR,
'valueid':sqla.types.INTEGER,'sourceuvid':sqla.types.VARCHAR,
'tablename':sqla.types.VARCHAR,'unit':sqla.types.CHAR(10),
'alias':sqla.types.VARCHAR,'config':sqla.types.INTEGER,
'valuetype':sqla.types.VARCHAR,'activated':sqla.types.BOOLEAN,
'origin_spalte':sqla.types.INTEGER,'hpmode':sqla.types.VARCHAR}
#record big serial key
_EVENTS_EXT_DTYPES={'id_device':sqla.types.BIGINT,'date':sqla.types.TIMESTAMP,
'uvid':sqla.types.VARCHAR,'message':sqla.types.VARCHAR}
#record big serial key
#note precision for measure could be specified
_MEASUREMENT_DTYPES={'id_device':sqla.types.BIGINT,'uvid':sqla.types.VARCHAR,
'date':sqla.types.TIMESTAMP,'measure':sqla.types.FLOAT}           
    


def _files_info_positions(path_to_files,files_list,path_to_output=None,info_id='#',save=False):
    """ Return values and positions based on assumed types of info """                          
    #DEV NOTES 23/7/18: see variant/improved version on wacom 
    #DEV NOTES 8/6/18: faster (0:00:00.368514 vs 0:00:03.028550) than _files_info          
    start = dtt.datetime.now()       
    #return an overview of the positions for each type of information
    #created on 16/5/18: very specific for the type of structure, initialising table
    #DEV NOTES: 4,23 minutes for table    
    rows_list = []
    #DEV NOTE 23/7/18: "end_line" not used
    #defines type of information to be found. Following code depends on expected structure
    info_types = ('file_path','info_line','info_value',
    'header_line','header_value','sample_line','sample_value','end_line')
    header_ck = False
    sample_ck = False
    info_line = None
    info_value = None
    header_line = None
    header_value = None
    sample_line = None
    sample_value = None
    for fname in files_list:
        fhand = open(path_to_files+fname) 
        print(fhand)
        for index, line in enumerate(fhand):
            value = line.rstrip()              
            if line != None:      
                if value.startswith(info_id):
                    sections_tmp = {info_types[0]:str(path_to_files+fname),info_types[1]:info_line, 
                    info_types[2]:info_value,info_types[3]:header_line,info_types[4]:header_value,
                    info_types[5]:sample_line,info_types[6]:sample_value,info_types[7]:str(index)}
                    if sections_tmp != None: rows_list.append(sections_tmp)
                    #initialising new section info
                    header_line = None
                    header_value = None
                    sample_line = None
                    sample_value = None
                    header_ck = False
                    sample_ck = False
                    #assigning for new line 
                    info_line = str(index)
                    info_value = str(value)
                    continue
                elif header_ck == False:
                    header_ck = True
                    header_line = str(index)
                    header_value = str(value)
                    continue
                elif header_ck == True and sample_ck == False:
                    sample_ck = True
                    sample_line = str(index)
                    sample_value = str(value)
                    continue
        fhand.close()
    rowsdf = pd.DataFrame(rows_list,columns=info_types)  
    if save==True:
        rowsdf.to_csv(path_to_output+"info.csv")      
    print("processing time"+str(dtt.datetime.now()-start))
    return rowsdf


 
def _measurements_to_dataframe(file_info_positions,path_to_files,files_list,path_to_output,
    output_dictionary,datetime_start,datetime_end,save_to_csv=True): 
    """ extract measurements only from files and aggregate together in different dataframes per info type """
    """ info should be same type and positions """     
    """ different than transfer to database since filename and not generated id use """
    """ dataframe_info_positions from pd.read_csv(path_to_info,header=0) """
    #initialise empty dataframe 
    df_measurement_all = pd.DataFrame()
    for fname in files_list:
        file_path = str(path_to_files) + str(fname)      
        file_info_positions = file_info_positions[(file_info_positions.file_path==file_path)]
        #next iteration if file not found 
        if file_info_positions.shape[0] == 0:
            print(file_path+" not found")
            continue
        elif (file_info_positions.shape[0] != 0):
            #info positions for measurement            
            events_end = int(file_info_positions[(file_info_positions.info_value=="## events_ext ##")]["end_line"].values)   
            #DEV TEST  #records = sum(1 for line in open(file_path))
            
            #extract only the selected indexes of columns
            try:
                #read  for measurement
                df_measurement = pd.read_csv(file_path,header=events_end+1,delimiter=";",skip_blank_lines=False,
                low_memory=False)
                #,usecols=output_dictionary.keys()) REMOVE ON 8/6/18
            except:
                print("reading error for "+file_path)
                continue
           #condition that all basic type of informations are included 
            if df_measurement.shape[1] > 0:   
                #iteration to rename measurements header (under the same type)
                clm_enm = enumerate(df_measurement.columns)
                val_enm = enumerate(output_dictionary.values())
                # loop over the two lists
                for clm, val in zip(clm_enm,val_enm):
                    #take the value (pos1) when renaming
                    df_measurement.rename(columns={clm[1]:val[1]},inplace=True)                
                df_measurement_select = df_measurement[(df_measurement.date >= datetime_start) & (df_measurement.date <= datetime_end)]
                #create an array to store the file name 
                df_measurement_select =df_measurement_select.assign(
                filename=np.full(df_measurement_select.shape[0],str(fname)))  
                df_measurement_all = df_measurement_all.append(df_measurement_select,
                ignore_index=True)
                if save_to_csv == True: 
                    df_measurement_all.to_csv(path_to_output+"files_measurement.csv")    
    return df_measurement_all




def _general_info_to_dataframe(path_to_files,dataframe_info_positions,info_type,path_to_output,
    files_start_list,files_code_list,files_end='.csv',save_to_csv=True):    
    """ extract one type of info only from files """     
    """ different than format in database since filename parameter used and not id_device (generated by the database) """
    """ assume structure same for all files """        
    if (info_type == "device") | (info_type == "value") | (info_type == "events_ext"):
        files = [f for f in os.listdir(path_to_files) if (f.endswith(files_end) & f.startswith(files_start_list))]
        #initialise empty dataframe 
        df_info_all = pd.DataFrame()
        for fname in files:  
            file_path =  path_to_files + fname    
            file_info_positions = dataframe_info_positions[(dataframe_info_positions.file_path==file_path)]
            #next iteration if file not found 
            if file_info_positions.shape[0] == 0:
                print(fname+" not found")
                continue
            elif file_info_positions.shape[0] != 0:
                #define if id 
                info_id = str('## ')+info_type+str(' ##')
                info_header = int(file_info_positions[(file_info_positions.info_value==info_id)]["header_line"].values)  
                info_end = int(file_info_positions[(file_info_positions.info_value==info_id)]["end_line"].values)  
                try:
                    df_info = pd.read_csv(file_path,delimiter=";",skip_blank_lines=False,
                    header=info_header,nrows=info_end-info_header-1,low_memory=False)
                    df_info = df_info.dropna(axis=0,how="all")  
                except:
                    print("error in "+fname)
                if df_info.shape[1] > 0:   
                    #create an array to store the file name 
                    df_info["filename"]=str(fname)                
                    df_info_all = df_info_all.append(df_info,
                    ignore_index=True)
                    if save_to_csv == True: 
                        df_info_all.to_csv(path_to_output+"files_"+info_type+".csv")    
        return df_info_all
    elif (info_type != "device") & (info_type != "value") & (info_type != "events_ext"):
        return None


def _devices_uncopied_to_sql(dataframe_device_all,schema=_SCHEMA,connection=_CONNECTION):
    """ list devices not_copied on sql based on the identifier udid """   
    from sqlalchemy import engine as e
    engine_connection = e.create_engine(connection)  
    #create list of unique udid
    udid_list = dataframe_device_all["udid"].values    
    
    #create select query with all found udid 
    control_query = "SELECT tab.udid FROM "+str(schema)+".device AS tab WHERE "
    started = False
    for value in udid_list:
        if started == True: 
            control_query = control_query + " OR "
        control_query = control_query + "tab.udid ='" + str(value)+"'"
        started = True
    #launch select query
    connection = engine_connection.connect()
    # transform into array to give only results 
    udid_copied = np.asarray(connection.execute(control_query).fetchall())
    connection.close()
    #retrieve udid values which are in list but not copied 
    udid_not_copied = np.setdiff1d(udid_list,udid_copied,assume_unique=True)
    files_udid_not_copied = dataframe_device_all[dataframe_device_all["udid"].isin(udid_not_copied)][["filename","udid"]]      
    return files_udid_not_copied 

    
def _dataframe_to_sql(_path_to_files,files_udid_not_copied,dataframe_device_all,dataframe_value_all,
dataframe_events_ext_all,dataframe_measurement_all,schema=_SCHEMA,connection=_CONNECTION):
    """ transfer all info based on udid_list. Works only on already established (global)types"""  
    #DEV notes: could be modified to enlarge type of data ??
    #sql connection setup
    #if empty list return error
    if files_udid_not_copied.shape[0] == 0: return print("Error: empty list of udid")    
    #sqlalchemy setup
    from sqlalchemy import engine as e
    _engine_connection = e.create_engine(connection)
    #new empty list to import properly formatted devices
    udid_copied = list()
    #iteratively import all device info   
    for value in files_udid_not_copied["udid"]:
        #filter df_device for udid value and take only specified parameters 
        _df_device = dataframe_device_all[(dataframe_device_all.udid == value)][list(_DEVICE_DTYPES.keys())]       
        #if device not null
        if _df_device.shape[0] > 0:        
            _df_device.to_sql("device",_engine_connection,schema=schema,
                 if_exists="append",index=False,index_label=None,
                 chunksize=None,dtype=_DEVICE_DTYPES)
            udid_copied.append(value)
            print("Imported device with udid="+value)
        elif _df_device.shape[0] ==0:            
            print("Not imported device with udid="+value)
    #iteratively look for produced id_device   
    look_query = "SELECT tab.udid,tab.id_device FROM "+str(schema)+".device AS tab WHERE "
    started = False    
    for value in udid_copied:
        if started == True: 
                look_query =  look_query + " OR "
        look_query =     look_query + "tab.udid ='" + str(value)+"'"
        started = True
    _connection = _engine_connection.connect()
    _id_copied = np.asarray(_connection.execute(look_query).fetchall())
    _connection.close()
    #iteratively importing value for each udid
    #for value in udid_copied:
    for udid,device in _id_copied:
        #udid_start for required time processing  
        udid_start = dtt.datetime.now()   
        #extract filename and filter dataframes for the specific filename
        filename = files_udid_not_copied[files_udid_not_copied["udid"]==udid].loc[:,"filename"].values[0] 
        #start manipulating dataframes value_select and events_ext
        _df_value_select = dataframe_value_all[(dataframe_value_all.filename==filename)]   
        _df_events_ext_select = dataframe_events_ext_all[(dataframe_events_ext_all.filename==filename)]  
        #drop all null rows        
        _df_value_select = _df_value_select.dropna(axis=0,how="all")
        _df_events_ext_select = _df_events_ext_select.dropna(axis=0,how="all")  
        #assigning id device to value, events and measurements 
        _df_value_select = _df_value_select.assign(id_device = lambda x:str(device))
        _df_events_ext_select = _df_events_ext_select.assign(id_device = lambda x:str(device))

        #selecting columns for sql 
        _df_value_select =_df_value_select[list(_VALUE_DTYPES.keys())]    
        _df_events_ext_select = _df_events_ext_select[list(_EVENTS_EXT_DTYPES.keys())]                    
        #importing value information     
        _df_value_select.to_sql("value",_engine_connection,schema=schema,if_exists="append",index=False,
                                index_label=None,chunksize=None,dtype=_VALUE_DTYPES)
        #importing events ext information    
        _df_events_ext_select.to_sql("events_ext",_engine_connection,schema=schema,if_exists="append",index=False,
                                index_label=None,chunksize=None,dtype=_EVENTS_EXT_DTYPES)
        #preliminary filtering for measurement
        _df_measurement_select = dataframe_measurement_all[(dataframe_measurement_all.filename==filename)]        
        _df_measurement_select = _df_measurement_select.dropna(axis=0,how="all")
         #remove last line if necessary 
        _df_measurement_select = _df_measurement_select[_df_measurement_select.date != "## csv dump complete ##"]        #importing measurement information 
            #DEV NOTE: by asking both index & value it split into the two
        for index,value in enumerate(_df_measurement_select.columns):
            if (str(value) != "date") and (str(value) != "filename") and "Unnamed" not in value: 
                print(str(value)+" start processing "+str(dtt.datetime.now()))
                #DEV NOTES: sequence used instead of pd.Dataframe function (slower?)   
                #selection of date and value column
                _df_measure_tmp = _df_measurement_select.loc[:,('date',value)]
                #renaming value into generic "measure" label
                _df_measure_tmp.rename(columns={value:"measure"},inplace=True)
                #adding id_device
                _df_measure_tmp = _df_measure_tmp.assign(id_device = lambda x:str(device))
                #copying into sql 
                print(_df_measure_tmp)
                _df_measure_tmp.to_sql("measure",_engine_connection,schema=schema,
                if_exists="append",index=False,index_label=None,chunksize=None,dtype=_MEASUREMENT_DTYPES)
                print(str(value)+" end processing "+str(dtt.datetime.now()))
        lenght = dtt.datetime.now() - udid_start
        print(filename + "/" + udid + " processed in "+str(lenght))
        
        break
 
"""   
#DEV NOTE: controld code could be added 

            



########################################################################################################################
# OBSOLETE
########################################################################################################################


def _files_info(path_to_files,path_to_output,files_end='.csv',info_id='##',save_info=False):
    #return info about file content through headers positions and values 
    #DEV NOTES 8/6/18: faster (0:00:00.368514 vs 0:00:03.028550) than _files_info  
    start = dtt.datetime.now()
    #DEV NOTES: more than 20 minutes for table    
    all_loc = pd.DataFrame()    
    all_values = pd.DataFrame()     
    files = [f for f in os.listdir(path_to_files) if f.endswith(files_end)]
    for fname in files: 
        #read each file complying with the criteria defined       
        filepath = path_to_files+fname 
        csvimp = pd.read_csv(filepath,header=None,sep=None,delimiter=None,skiprows=0,
        squeeze=True,skip_blank_lines=True)
               
        srs_tmp = csvimp.iloc[:,0].map(lambda x: x.startswith('##'))
                 
        itr_comments =  srs_tmp[srs_tmp==True]         

        tables_loc = pd.DataFrame({'info':itr_comments.index})
        tables_values = pd.DataFrame()   
    
        for index in tables_loc.index:
            tables_loc.loc[index,'file'] = fname
            tables_values.loc[index,'file'] = fname
            info_loc = tables_loc.loc[index,'info']
            #evaluating if last info chunk
            info_str = ""
            for column in csvimp.columns:
                if csvimp.iloc[info_loc,column] != None:
                        info_str = info_str + str(csvimp.iloc[info_loc,column])
            tables_values.loc[index,'info_str'] = info_str
            if index != tables_loc.shape[0]-1:   
                end = int(tables_loc.loc[index+1,'info'])-1
            elif index == tables_loc.shape[0]-1:
                end = int(csvimp.shape[0]-1)-1       
            #defining which information available
            if info_loc + 1 > end:
                tables_loc.loc[index,'header'] = None
                tables_loc.loc[index,'data_end'] = None
            elif info_loc + 1 <= end:
                header_loc = tables_loc.loc[index,'info'] + 1
                tables_loc.loc[index,'header'] = header_loc
                #header line
                header_str = ""
                for column in csvimp.columns:
                    if csvimp.iloc[header_loc,column] != None:
                        header_str = header_str + str(csvimp.iloc[header_loc,column])
                tables_values.loc[index,'header_str'] = header_str
                if info_loc + 2 > end:
                    tables_loc.loc[index,'data_end'] = tables_loc.loc[index,'header']
                elif info_loc + 2 <= end:
                    tables_loc.loc[index,'data_end'] = end
                    #last data line
                    data_str = ""
                    for column in csvimp.columns:
                        if csvimp.iloc[end,column] != None:
                            data_str = data_str + str(csvimp.iloc[end,column])
                        tables_values.loc[index,'data_str'] = data_str
                    
        
        #tables_loc.to_csv(output_folder+fname)
        #tables_values.to_csv(output_folder+"val"+fname)
        
        csvimp = None                 
             
        all_loc = all_loc.append(tables_loc)
        all_values = all_values.append(tables_values)
        
    all_loc.to_csv(path_to_output+"Report_loc.csv")
    all_values.to_csv(path_to_output+"Report_values.csv")
    
    end = dtt.datetime.now()
     
    print("Processing time:"+str(end-start))
    #0:30:19.976000 all excel
    
    return all_loc






   

def _file_text_to_columns(path_to_file,path_to_output,split_id=";"):
    #OBSOLETE: NOT USED ANYMORE 17/5/18
    df = pd.read_csv(path_to_file,header=0)          
    #extract only one part of the string originally designed for the file info_values.csv
    df["info_str_mod"]=df["info_str"].apply(lambda x: str(x)[2:8])    
    #df2=df[(df.info_str_mod== "value#") | (df.info_str_mod == "device")]   
    
    df_dev=df[(df.info_str_mod == "device")]    
    data_df = pd.DataFrame()
    #DEV NOTES: not code effective      
    for index,value in enumerate(df_dev.loc[:,"header_str"].values):
        header_list_tmp = enumerate((str(value)).split(split_id))    
        for index2, value2 in header_list_tmp:
            #data_df.loc[index,index2*2] = header_list[index2]
            data_df.loc[index,index2*2+1] = value2  
    for index,value in enumerate(df_dev.loc[:,"data_str"].values):
        data_list_tmp = enumerate((str(value)).split(split_id))
        for index2, value2 in data_list_tmp:
            #data_df.loc[index,index2*2] = header_list[index2]
            data_df.loc[index,index2*2+2] = value2
    data_df.loc[:,"file"] = df_dev.loc[:,"file"]
    #not necessary if not iteration 
    #data_f2 = data_df.reindex_axis(sorted(data_df.columns), axis=1)
    data_df.to_csv(path_to_output+"text_to_columns.csv")
    return data_df
    

def _info_values_clean(path_to_file,path_to_output): 
    #OBSOLETE: NOT USED ANYMORE 17/5/18 
    #assign headers to the very specific info_values.csv file
    df = _file_text_to_columns(path_to_file,path_to_output)
    df2 = pd.DataFrame({"udid":df.loc[:,"1"].values,
    "devtype":df.loc[:,"19"].values,
    "alias":df.loc[:,"21"].values,  
    "vendor":df.loc[:,"23"].values,
    "ppeak":df.loc[:,"27"].values,  
    "sparent":df.loc[:,"29"].values,    
    "deviceid":df.loc[:,"33"].values,        
    "devicetype":df.loc[:,"37"].values,  
    "customalias":df.loc[:,"41"].values,
    "file":df.loc[:,"1"].values},
    index = df.loc[:,"1"].values)
    df2.to_csv(path_to_output+"info_values_clean.csv")       
"""        