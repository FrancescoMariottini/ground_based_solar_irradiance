# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:10:20 2017

@author: wsfm
"""

import sys
PATH_TO_MODULES = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/ground_based_solar_irradiance/"
#adding path to the tailored made modules
sys.path.append(PATH_TO_MODULES)


import os
import matplotlib
import matplotlib.pyplot as plt 



""" GLOBAL VARIABLES (starting setup) """
#define and create output folder if it doesn't exist already           
if os.path.isdir(os.getcwd()+'\output') == False:
    os.mkdir(os.getcwd()+r"\output")
    
    
output_folder = os.path.join(os.getcwd(), r'output/')   
#output_images = os.path.join(os.getcwd(), r'output/',r'images/')   

               


#functions for the PieChart class 
#TO IMPROVE: use on matplotlib generate 1405 user warning
matplotlib.use("Svg")

from matplotlib.patches import Shadow


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
   
    def Save(self):
                
        fn = output_folder + self.name + "." + str(self.fformat)
        
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
        
    def Save(self):
        """IMPROVE: only save without showing"""
        # make a square figure and axes
        #https://matplotlib.org/devdocs/gallery/misc/svg_filter_pie.html#sphx-glr-gallery-misc-svg-filter-pie-py
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
        
        fn = output_folder + self.SVGname + ".svg"
        print("Saving '%s'" % fn)
        ET.ElementTree(tree).write(fn)
        plt.close()
                     
                     
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

def SectionImage(title,filename,outcomes=''):
    _url=output_folder + filename
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

def SectionSVG(title,filename,outcomes=''):
    #_url= os.getcwd() + '\OutputReport\\' + filename  
    _url=output_folder + filename               
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



def CloseWritehtml(string):
    _tail = '''
        </body>
    </html>
    '''
    string = string + _tail
    f = open(output_folder+"Report.html",'w')
    f.write(string)
    f.close()
    return None




