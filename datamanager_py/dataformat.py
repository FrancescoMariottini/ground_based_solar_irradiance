# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:49:49 2018

@author: wsfm
"""

import os
#import pandas as pd #31/8/20 not used
    
def directories(path_to_files):
    directories=[name for name in os.listdir(path_to_files) 
                if os.path.isdir(path_to_files)]
    directories_WP2=[name for name in directories if "WP2" in name]
    directories_WP6=[name for name in directories if "WP6" in name]
    directories_other = [name for name in directories if ("WP6" not in name and "WP2" not in name)]
    
    return directories
    
path_to_files=r"C:\Users\wsfm\OneDrive - Loughborough University\_Personal_Backup\_Events\_SOLAR-TRAIN_events"

directories(path_to_files)

   

def unzip_tarfiles(path_to_tarfiles, unzip_where_folder):
    import tarfile
    # for this function only     
    """
    Fast unzipper for csv.tar.gz files
    Created on 14 Dec 2017
    @author: Elena Koumpli
    See more in https://stackoverflow.com/questions/30887979/i-want-to-create-a-script-for-unzip-tar-gz-file-via-python
    2  """ 
    print("Input Path: "+str(path_to_tarfiles))
    print("Output Path: "+str(unzip_where_folder))
    #DEV 
    files = [f for f in os.listdir(path_to_tarfiles) if f.endswith('.csv.tar.gz')]
    for fname in files:
        print(fname)            
        tar = tarfile.open(path_to_tarfiles+fname, "r:gz")
        tar.extractall(unzip_where_folder)
        tar.close()
    return None


    