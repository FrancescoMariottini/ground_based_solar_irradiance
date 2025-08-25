# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:54:05 2019
Last modified 12/5/2020

@author: wsfm
"""

#print("""Area under curve testing""")
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
norm.cdf(1.96)

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)  
  return np.roots([a,b,c])

m1 = 2.5
std1 = 3.0
m2 = 5
std2 = 1.0
#Get point of intersect
result = solve(m1,m2,std1,std2)
#Get point on surface
x = np.linspace(-5,9,10000)
plot1=plt.plot(x,norm.pdf(x,m1,std1))
plot2=plt.plot(x,norm.pdf(x,m2,std2))
plot3=plt.plot(result,norm.pdf(result,m1,std1),'o')
#Plots integrated area
r = result[0]
olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r],m1,std1),alpha=0.3)
olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r],m2,std2),alpha=0.3)
# integrate
area = norm.cdf(r,m2,std2) + (1.-norm.cdf(r,m1,std1))
print("Area under curves ", area)
plt.show()

"""


import pvdata.executable.data_handling as datah
import datetime as dt

_PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_PhD_Thesis/__Outputs/PhD03/"
_PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/data_handling_disc/"
_PATH_TO_OUTPUT = r"C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/filtering/"

# 25/8/25 Removed confidential information

_PROCESS_TYPE = "performance filtering"
#_PROCESS_TYPE = "alignment" 
_PROCESS_SENSOR = "Bolzano"
_PROCESS_NAMES = [_PROCESS_TYPE + " " + _PROCESS_SENSOR]

_PROCESS_NAMES= [
"alignment Bolzano"]
_PROCESS_NAMES= [
"alignment Bolzano"]
#_PROCESS_NAMES = ["alignment Bolzano"]
#_PROCESS_NAMES = "test4"
_PROCESS_NAMES= ["performance filtering ivpoints crest roof remote sql mms"]
#_PROCESS_NAMES= ["performance filtering ivpoints crest roof sql mms indoor"]


#"performance flagging mms crest roof remote sql mms indoor 2013",
#"performance flagging mms crest roof remote sql mms indoor 2014",
#"performance flagging mms crest roof remote sql mms indoor 2015",

_PROCESS_NAMES= [
"performance flagging mms crest roof remote sql mms indoor 2016",
"performance flagging mms crest roof remote sql mms indoor 2017",
"performance flagging mms crest roof remote sql mms indoor 2018",
"performance flagging mms crest roof remote sql mms indoor 2019",
"performance flagging mms crest roof remote sql mms indoor 2013",
"performance flagging mms crest roof remote sql mms indoor 2014"]


"""
_PROCESS_NAMES= ["performance flagging Bolzano",
                 "performance flagging mms crest roof mms indoor pregrouped"
                 ]
"""
#"performance flagging mms crest roof remote sql mms indoor 2014"]
#_PROCESS_NAMES= ["performance flagging mms crest roof remote sql mms indoor 2018"]


_TEST=False     
for value in list(_PROCESS_NAMES):
    process=value
    datah.run(process,test=_TEST,path_to_output=_PATH_TO_OUTPUT)


#TEST FUNCTIONS RELATED TO PHD03 MODULES

if "test time grouping" in _PROCESS_NAMES:
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
                       'Quantity': [1, 3, 5, 1, 8, 1, 9, 3],
                       'Test':[2,4,5,6,7,8,10,100],
                       'Date': [
                           dt.datetime(2013, 1, 2, 13, 0),
                           dt.datetime(2013, 1, 2, 13, 5),
                           dt.datetime(2013, 1, 2, 13, 10),
                           dt.datetime(2013, 1, 2, 13, 15),
                           dt.datetime(2013, 1, 2, 13, 20),
                           dt.datetime(2013, 1, 2, 13, 25),
                           dt.datetime(2013, 1, 2, 13, 30),
                           dt.datetime(2013, 1, 2, 13, 35)],
                        'Date2': [
                           dt.datetime(2013, 1, 2, 13, 0),
                           dt.datetime(2013, 1, 2, 13, 5),
                           dt.datetime(2013, 1, 2, 13, 10),
                           dt.datetime(2013, 1, 2, 13, 15),
                           dt.datetime(2013, 1, 2, 13, 20),
                           dt.datetime(2013, 1, 2, 13, 25),
                           dt.datetime(2013, 1, 2, 13, 30),
                           dt.datetime(2013, 1, 2, 13, 35)]
                        })
    print(df)
    dfm = df.groupby([pd.Grouper(freq='900s', key='Date')])
    df2 = dfm.aggregate([min])    
    print(df2)
    df3 = df.join(df.groupby([pd.Grouper(freq='900s',key='Date',label='left')]).min(),on="Date",rsuffix="_r")
    print(df3)
    df4 = df.join(df.groupby([pd.Grouper(freq='900s',key='Date',label='right')]).min(),on="Date",rsuffix="_r")
    print(df4)
    

    freq = 600

    def group(x:pd.DatetimeIndex):
        ts = (x.hour*3600+x.minute*60+x.second)//freq*freq
        h = ts//3600
        m = (ts-h*3600)//60
        s = ts-h*3600-m*60
        return dt.datetime(x.year,x.month,x.day,h,m,s)
   
    
    df["s"] = df["Date"].apply(lambda x: group(x))
    
    print(df)
    
    #df2 = dfm["Test"].aggregate([np.median,np.std])    
    
     
if "test solar series" in _PROCESS_NAMES: 
    LATITUDE = 41.841831
    LONGITUDE = 12.301606
    ALTITUDE = 0 #assumed 26/3/19
    TIMEZONE = 'Europe/Rome'
    TILT = 30
    AZIMUTH = 180     
    PRESSURE=101325.
    TRANSMITTANCE=0.5   #modeling
    ALBEDO = 0.25
    APPARENT_ZENITH_MODEL="kastenyoung1989"  #used for atm module 
    DIFFUSE_MODEL="allsitescomposite1990"
    POA_MODEL="isotropic"     #POA_MODEL="perez" DEV NOTE 3/19 run error
    PACKAGE="pvlib"
    import pandas as pd 
    import datetime as dt
    import pvdata.solarlibraries.pvlibinterface as pvlibint #importing pvlibinterface for solar zenith
    location = pvlibint.SolarLibrary(latitude=LATITUDE,
                                         longitude=LONGITUDE,
                                         altitude=ALTITUDE,
                                         surface_tilt=TILT,
                                         surface_azimuth=AZIMUTH,
                                         package=PACKAGE,
                                         diffuse_model=DIFFUSE_MODEL,
                                         transmittance=TRANSMITTANCE,
                                         pressure=PRESSURE,
                                         albedo=ALBEDO,
                                         poa_model=POA_MODEL,
                                         timezone=TIMEZONE)    
    if "getsolarseries" in _PROCESS_NAMES: 
        pd.set_option('display.max_columns', 20)
        datetimeindex_utc = [dt.datetime(2019,5,3,8,00),dt.datetime(2019,5,3,10,00),dt.datetime(2019,5,3,12,00),dt.datetime(2019,5,3,14,00)]
        dates = pd.DatetimeIndex(datetimeindex_utc,tz=None,dayfirst=True)   
        parameters = list(["clearskypoa","aoi"]) #define requested parameters
        df_par = location.getsolarseries(datetimeindex_utc,location.timezone,parameters) 
        


