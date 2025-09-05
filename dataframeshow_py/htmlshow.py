# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:10:20 2017

@author: wsfm
"""
import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)


def Head(title):
    _head = '''
    <html>
    <head>
        <link rel="stylesheet">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
    </head>
        <body>
        <h1>'''+title+'''</h1>
    '''
    return _head

def SectionImage(path_to_output,title,filename,outcomes=''):
    _url=path_to_output + filename
    _section='''
                <!-- *** Section 1 *** --->
                <h2>'''+title+'''</h2>
                <div style="width: 1000px; height: 500px;">
                    <img src='''+_url+''' style="height: 100%; width: 100%;">
                </div>         
                <p>'''+outcomes+'''</p>
    '''
    return _section 


"""
style="width:auto;
style="width:100%;height:100%;">
"""

def SectionSVG(path_to_output,title,filename,outcomes=''):
    #_url= os.getcwd() + '\OutputReport\\' + filename  
    _url=path_to_output + filename               
    _section='''
                <!-- *** Section 1 *** --->
                <h2>'''+title+'''</h2>
                <img
                    src='''+_url+'''
                    height='500'
                    width='1000'/>
                <p>'''+outcomes+'''</p>
    '''        
    return _section 



def CloseWritehtml(path_to_output,html_name,string):
    _tail = '''
        </body>
    </html>
    '''
    string = string + _tail
    f = open(path_to_output+html_name+".html",'w')
    f.write(string)
    f.close()
    return None




