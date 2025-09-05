# -*- coding: utf-8 -*-
"""
@author: Francesco Mariottini 
Created on 14/8/17
more information on the readme file 
22/2/19: old version to be replaced by figureshow after all graphs plotted

"""
import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)



import matplotlib.pyplot as plt
#import shadows for the pie chart
from matplotlib.patches import Shadow
#manipulation of text
import textwrap as twrp
# importing pandas for using series ??
import pandas as pd


#import axes for limit
#import matplotlib.axes as plt_ax
#import for date tick labes
import matplotlib.dates as plt_d
#time difference for data completeness
import datetime as dtt


"""GLOBAL VARIABLES """
DPI = 100


class Graph:
    def __init__(self,name, 
        bl = 18,
        leg = 16,
        tck = 16,
        res = 300,
        leg_loc = 'upper right',
        fformat = 'jpg',
        leg_layout = 'horizzontal',
        width = 50):
        self.name = name
        self.bl = bl
        self.leg = leg
        self.tck = tck
        self.res = res
        self.leg_loc = leg_loc 
        self.fformat = fformat
        self.leg_layout = leg_layout
        self.width = width
        #default dictionary from Elena's function plotDynGraphsOnXY(plotGraphs.py)   
        #self.pprefs = {'lbl':18,'leg':16, 'tck':16,'res': 300, 
        #'leg_loc':'upper right','fformat':'jpg',
        #'leg_layout':'horizzontal','width':50}
        #DEV NOTE 6/9/18: "Save" should be "save"
    def save(self,path_to_output):
        fn = str(path_to_output) + self.name + "." + str(self.fformat)
        plt.savefig(fn,dpi = self.res,fformat = self.fformat,bbox_inches="tight",)
        plt.close()
    class PieChart:
        
        def __init__(self,SVGname,labels,fracs,explode):
            self.SVGname = SVGname        
            self.labels = []
            self.fracs = []
            self.explode = []
            for index,value in enumerate(fracs):
                if value != 0:
                    self.labels.append(labels[index])
                    self.fracs.append(fracs[index])
                    self.explode.append(explode[index])   
        def save(self,path_to_output):
            """IMPROVE: only save without showing"""
            # make a square figure and axes
            #https://matplotlib.org/devdocs/gallery/misc/svg_filter_pie.html#sphx-glr-gallery-misc-svg-filter-pie-py
            #"Save" renamed as "save" on 31/8/18. Maybe some code should be updated. 
            fig1 = plt.figure(1, figsize=(10, 10))
            ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
            patches, texts, autotexts = ax.pie(self.fracs, explode=self.explode, labels= self.labels, autopct='%1.1f%%')
            #, autopct='%1.1f%%')
            for w in patches:
                # set the id with the label.
                w.set_gid(w.get_label())
                # we don't want to draw the edge of the pie
                w.set_ec("none")
            for w in patches:
                # create shadow patch
                s = Shadow(w, -0.01, -0.01)
                s.set_gid(w.get_gid() + "_shadow")
                s.set_zorder(w.get_zorder() - 0.1)
                ax.add_patch(s)
            for t in texts:
                #t.set_size('smaller')
                #wrap text
                t.set_wrap(True)
            # save
            from io import BytesIO
            f = BytesIO()
            plt.savefig(f, format="svg")
            import xml.etree.cElementTree as ET
            # filter definition for shadow using a gaussian blur
            # and lightening effect.
            # The lightening filter is copied from http://www.w3.org/TR/SVG/filters.html
            # I tested it with Inkscape and Firefox3. "Gaussian blur" is supported
            # in both, but the lightening effect only in the Inkscape. Also note
            # that, Inkscape's exporting also may not support it.
            filter_def = """
              <defs  xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>
                <filter id='dropshadow' height='1.2' width='1.2'>
                  <feGaussianBlur result='blur' stdDeviation='2'/>
                </filter>
            
                <filter id='MyFilter' filterUnits='objectBoundingBox' x='0' y='0' width='1' height='1'>
                  <feGaussianBlur in='SourceAlpha' stdDeviation='4%' result='blur'/>
                  <feOffset in='blur' dx='4%' dy='4%' result='offsetBlur'/>
                  <feSpecularLighting in='blur' surfaceScale='5' specularConstant='.75'
                       specularExponent='20' lighting-color='#bbbbbb' result='specOut'>
                    <fePointLight x='-5000%' y='-10000%' z='20000%'/>
                  </feSpecularLighting>
                  <feComposite in='specOut' in2='SourceAlpha' operator='in' result='specOut'/>
                  <feComposite in='SourceGraphic' in2='specOut' operator='arithmetic'
                k1='0' k2='1' k3='1' k4='0'/>
                </filter>
              </defs>
            """
            tree, xmlid = ET.XMLID(f.getvalue())
            # insert the filter definition in the svg dom tree.
            tree.insert(0, ET.XML(filter_def))
            for i, pie_name in enumerate(self.labels):
                pie = xmlid[pie_name]
                pie.set("filter", 'url(#MyFilter)')
                shadow = xmlid[pie_name + "_shadow"]
                shadow.set("filter", 'url(#dropshadow)')
            fn = path_to_output + self.SVGname + ".svg"
            print("Saving '%s'" % fn)
            ET.ElementTree(tree).write(fn)
            #DEV NOTE 26/10/18 replaced plt.close() with plt.show() and it didn't work
            plt.close()
            #plt.show()
    
    
    
    def plotvstime(dataframe,time_label,frequency="h",merged_y=False,title=None,legend_anchor=-0.4,markersize=6,path_to_output=None):
        """show scatters plot for all variables in a df based on time column (used as x axis)"""
        # 28/7/17 created by fm    
        # 13/2/18 including "merge" option 
        # 26/7/18 renamed "plotvstime" from "PlotVsTime", path_to_output, frequency parameters
        # frequencey parameters month (M), seconds (s)
        # 22/2/19 into 
        if merged_y == False:
            for i in dataframe.columns:
                if i != time_label:
                    #copied from the cookbook at                
                    #https://stackoverflow.com/questions/12945971/pandas-timeseries-plot-setting-x-axis-major-and-minor-ticks-and-labels               
                    #create series of values
                    series=pd.Series(data=dataframe.loc[:,i].values,index=dataframe.index)
                    #define figure size, DPI and framework
                    ax = plt.figure(figsize=(7,4),dpi=DPI).add_subplot(111)                               
                    #define min and max of intervals and frequency of x ticks
                    xticks = pd.date_range(start=min(dataframe.loc[:,time_label].values),
                    end=max(dataframe.loc[:,time_label].values),freq=frequency)    
                    series.plot(ax=ax,style=".",label='second line',xticks=xticks.to_pydatetime())
                    #format for tick label, can use also \n
                    ax.set_xticklabels([x.strftime('%d\n%m\n%Y') for x in xticks]);
                    ax.set_xticklabels([], minor=True)
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
                    p,= plt.plot(dataframe[time_label],dataframe[i],"o",markersize=markersize)    
            
            dataframe.plot(x=time_label,y=columns,title=title)                  
            plt.ylabel(merged_y)
            plt.legend(loc=8,bbox_to_anchor=(0,legend_anchor,1,legend_anchor),
            ncol=1, mode="expand", borderaxespad=0.) 
                    
                    
              
            """
            #FROM SUMMER SCHOOL
            dataframe.plot(x=time_label,y=columns,title="\n".join(twrp.wrap(title,70))) 
            """
            
            
                  
            
        #26/7/18 modified: 
        if path_to_output!=None:
            graph = Graph(str(title))    
            graph.Save(path_to_output)

