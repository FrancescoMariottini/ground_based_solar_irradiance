# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:39:52 2019
@author: wsfm
22/2/19: new version to replace old figureshow. could be merged with Elena
"""
import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)


import matplotlib as mpl
plt = mpl.pyplot #import matplotlib.pyplot as plt
#importing pandas for using series ??
import pandas as pd

#DEV NOTE 23/2/19: conditions on acceptable values could be put in all function

'GLOBAL VARIABLES'
_PATH_TO_OUTPUT=r'C:/Users/wsfm/OneDrive - Loughborough University/Documents/Pyhton_Test/'
#standard values for all graphs
_STYLES= 'presentation_190223' #'ggplot', 'presentation_190223'
_STYLES= 'classic'
_FONTSIZE='x-large' #relative or absolute (number)font size: 'xxsmall','x-large'
'matplotlib.axes.Axes.pie'
_AUTOPCT='%1.1f%%'#if not None, string or function used to label the wedges with their numeric value
_FONTSIZE_PIE=_FONTSIZE
'matplotlib.axes.Axes.tick_params'
_LABELSIZE=_FONTSIZE
'matplotlib.axes.Axes.set_xticklabels'
_MINOR=True #set minor & not than major
'matplotlib.figure.Figure'
_ADD_AXES_PIE = [0.1, 0.1, 0.8, 0.8] # [left, bottom, width, height] of new axis as fractions of width & height
_DPI=100 #resolution for figures
'matplotlib.patches.Patch'
_EDGECOLOR = "none" #patch edge color DEV NOTE 26/3/19: done for pie not for scatter
_SHADOW_OX = -0.01 #DEV NOTE 26/3/19 Shadow not used 
_SHADOW_OY = -0.01 #DEV NOTE 26/3/19 Shadow not used 
'matplotlib.pyplot.figure'
_FIGSIZE_SCATTER=(7,5) #width, height in inches. If None 6.4 & 4.8
_FIGSIZE_PIE=(10,10)
'matplotlib.pyplot.legend' #same FONTSIZE used#   
_BBOX_TO_ANCHOR=(0,1.01) #box location (x,y) or (x, y, width, height)
_BORDERAXESPAD=0 #pad between axes and legend border
_FANCYBOX=None #round edges around the FancyBboxPatch @FM: no sensible difference?
_LOC=3 #location: 'best'=0,'upper left'=2,'lower left'=3,'lower center'=8
 #loc/bbox combos 3/(0,1.1) https://jdhao.github.io/2018/01/23/matplotlib-legend-outside-of-axes/
_MODE='Expand' # expand fill axes area
_NCOL=1 #number of columns 
'matplotlib.pyplot.plot'
_MARKER='v'
_MARKERSIZE=6  
'matplotlib.pyplot.savefig'
_FFORMAT='jpg' #raster(jpg,png..) or vector(eps,pdf,ps,svg...)
_BBOX_INCHES='tight' #if inches only given portion, if 'tight' try tight bbox. 'False' too.
#'matplotlib.pyplot.ylim' to be defined per function

### not described yet ###
bl = 18
leg = 16
tck = 16
leg_layout = 'horizzontal'
width = 50
        
        

class Format:
    #definition & storage of formatting information
    def __init__(self,styles=_STYLES,
                 dpi=_DPI,
                 bbox_inches=_BBOX_INCHES,
                 bbox_to_anchor=_BBOX_TO_ANCHOR,
                 borderaxespad=_BORDERAXESPAD,
                 fancybox=_FANCYBOX,
                 figsize=_FIGSIZE_SCATTER,
                 fformat =_FFORMAT,
                 fontsize=_FONTSIZE,
                 labelsize=_LABELSIZE,
                 loc=_LOC,
                 minor=_MINOR,
                 mode=_MODE,
                 ncol=_NCOL,
                 marker=_MARKER,
                 markersize=_MARKERSIZE,
                 path_to_output=_PATH_TO_OUTPUT,
                 bl = bl,
                 leg = leg,
                 tck = tck,
                 leg_layout = leg_layout,
                 width = width):
        #DEV NOTE 23/2/19: series instead of array since no np imported
        plt.style.use(styles)        
        """
        if styles != None:
            if (isinstance(styles,pd.Series)) == False:                 
                if styles in mpl.style.available: 
                    self.styles = styles 
                    plt.style.use(styles)
            #DEV NOTE 24/2/19: to be checked 
            elif (isinstance(styles,pd.Series)) == True:
                for item in styles:                    
                    if str(item) in mpl.style.available: plt.style.use(str(item))
        """
        #DEV NOTE 23/2/19: even if constant values, self for referencing
        self.bbox_inches=bbox_inches
        self.bbox_to_anchor=bbox_to_anchor
        self.borderaxespad=borderaxespad
        self.dpi=dpi
        self.fancybox=fancybox
        self.figsize=figsize
        self.fformat=fformat
        self.fontsize=fontsize
        self.labelsize=labelsize
        self.loc=loc
        self.minor=minor
        self.mode=mode
        self.ncol=ncol
        self.marker=marker
        self.markersize=markersize
        #not described yet 
        self.bl = bl
        self.leg = leg
        self.tck = tck
        self.fformat = fformat
        self.leg_layout = leg_layout
        self.path_to_output=path_to_output
        self.styles = styles
        self.width = width               
        #default dictionary from Elena's function plotDynGraphsOnXY(plotGraphs.py)   
        #self.pprefs = {'lbl':18,'leg':16, 'tck':16,'res': 300, 
        #'leg_loc':'upper right','fformat':'jpg',
        #'leg_layout':'horizzontal','width':50}      
          
def save(f:Format,filename:str,
         bbox_inches=None,dpi=None,fformat=None,path_to_output=None):
    if path_to_output==None: path_to_output=f.path_to_output
    #plot saving, 23/2/19
    with plt.style.context(f.styles):        
        if bbox_inches==None: bbox_inches=f.bbox_inches
        if dpi==None: dpi=f.dpi
        if fformat==None: fformat=f.fformat
        fn = str(path_to_output) + filename + "." + str(fformat)
        plt.savefig(fn,dpi=dpi,format=fformat,bbox_inches=bbox_inches)
    #close plot after saving
    #plt.close() 
  
"""
def savepie(f:Format,filename:str,labels,path_to_output=None,fformat=None): #??? 26/3/19 not clear how works exactly  
    if path_to_output==None: path_to_output=f.path_to_output
    if fformat==None: fformat=f.fformat
    from io import BytesIO            
    f = BytesIO()
    plt.savefig(f, format=_FFORMAT)
    import xml.etree.cElementTree as ET #https://docs.python.org/2/library/xml.etree.elementtree.html       
    filter_def = ""
    tree, xmlid = ET.XMLID(f.getvalue())
    # insert the filter definition in the svg dom tree.
    tree.insert(0, ET.XML(filter_def))
    for i, pie_name in enumerate(labels):
        pie = xmlid[pie_name]
        pie.set("filter", 'url(#MyFilter)')
        shadow = xmlid[pie_name + "_shadow"]
        shadow.set("filter", 'url(#dropshadow)')
    fn = path_to_output + filename + "." + str(fformat)
    ET.ElementTree(tree).write(fn)
    #DEV NOTE 26/10/18 replaced plt.close() with plt.show() and it didn't work
    #plt.close()
"""    
                
def scatter(f:Format,x,y,label,
             marker=None,markersize=None,styles=None):
    #DEV NOTE 24/02/19: keep explicit label name, no dataset
    #DEV NOTE 24/02/19: could be replaced with scatter? How change into function
    #DEV NOTE 23/2/19: works for multiple but iterations should be done 
    #DEV NOTE 23/2/19: if shorter doesn't work. 
    #MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
    #warnings.warn(message, mplDeprecation, stacklevel=1)        
    with plt.style.context(f.styles):
        if marker==None: marker=f.marker
        if markersize==None: markersize=f.markersize 
        pi = plt.plot(x,y,marker,label=label,markersize=markersize)
        p, = pi    
        #legend
        plt.legend(loc=f.loc,bbox_to_anchor=f.bbox_to_anchor,ncol=f.ncol,
                   mode=f.mode,borderaxespad=f.borderaxespad,fontsize=f.fontsize)   
    return pi                
    #mpl.axes.Axes.tick_params(labelsize=_LABELSIZE)              
    #plt.show() 
    
def pie(f:Format,labels,fracs,explode):
    l=[]
    f=[]
    e=[]
    for index,value in enumerate(fracs):
        if value != 0:
            l.append(labels[index])
            f.append(fracs[index])
            e.append(explode[index])
    fig1 = plt.figure(1, figsize=_FIGSIZE_PIE)
    ax = fig1.add_axes(_ADD_AXES_PIE)
    patches, texts, autotexts = ax.pie(f, explode=e, labels=l, autopct=_AUTOPCT)
    for w in patches:
            w.set_gid(w.get_label()) # set the id with the label.
            w.set_edgecolor(_EDGECOLOR) # we don't want to draw the edge of the pie
    for t in texts:
        t.set_size(_FONTSIZE_PIE)
        t.set_wrap(True) #wrap text
    #DEV NOTE 27/3/19: testing
    plt.legend(loc=f.loc,bbox_to_anchor=f.bbox_to_anchor,ncol=f.ncol,
    mode=f.mode,borderaxespad=f.borderaxespad,fontsize=f.fontsize)
    return fig1
    
    
def plotvstime(dataframe,time_label,frequency="h",merged_y=False,title=None,bbox_to_anchor=_BBOX_TO_ANCHOR,path_to_output=None):
    #DEV NOTE 7/5/19: format to be implemented    
    """show scatters plot for all variables in a df based on time column (used as x axis)"""
    # 28/7/17 created by fm    
    # 13/2/18 including "merge" option 
    # 26/7/18 renamed "plotvstime" from "PlotVsTime", path_to_output, frequency parameters
    # frequencey parameters month (M), seconds (s)
    # 22/2/19 before plotvstime, now lineplotvstime 
    if merged_y == False:
        for i in dataframe.columns:
            if i != time_label:
                #copied from the cookbook at                
                #https://stackoverflow.com/questions/12945971/pandas-timeseries-plot-setting-x-axis-major-and-minor-ticks-and-labels               
                #create series of values
                series=pd.Series(data=dataframe.loc[:,i].values,index=dataframe.index)
                #define figure size, DPI and framework
                ax = plt.figure(figsize=_FIGSIZE_SCATTER,dpi=_DPI).add_subplot(111)                               
                #define min and max of intervals and frequency of x ticks
                xticks = pd.date_range(start=min(dataframe.loc[:,time_label].values),
                end=max(dataframe.loc[:,time_label].values),freq=frequency)    
                series.plot(ax=ax,style=".",label='second line',xticks=xticks.to_pydatetime())
                #format for tick label, can use also \n
                ax.set_xticklabels([x.strftime('%d\n%m\n%Y') for x in xticks]);
                ax.set_xticklabels([], minor=_MINOR)
                #label for y
                plt.ylabel(i)
                #label for x
                plt.xlabel("datetime")
                plt.show()
    elif merged_y != False:
        #DEV NOTE: to be improve like other cases
        columns = []
        for i in dataframe.columns:
            if i != time_label:
                columns.append(i)
                p,= plt.plot(dataframe[time_label],dataframe[i],"o",markersize=_MARKERSIZE)    
        
        dataframe.plot(x=time_label,y=columns,title=title)                  
        plt.ylabel(merged_y)
        plt.legend(loc=_LOC,bbox_to_anchor=_BBOX_TO_ANCHOR,
        ncol=_NCOL, mode=_MODE, borderaxespad=_BORDERAXESPAD,fontsize=_FONTSIZE) 
                
                
          
        """
        #FROM SUMMER SCHOOL
        dataframe.plot(x=time_label,y=columns,title="\n".join(twrp.wrap(title,70))) 
        """
        
        
              
        
    #26/7/18 modified: 
    if path_to_output!=None:
        graph = Graph(str(title))    
        graph.Save(path_to_output)

