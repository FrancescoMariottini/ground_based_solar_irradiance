import datetime as dt
import functools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
import os
import numpy as np

#DEV NOTE 13/6/21: TBC which parameters already in
# https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams


# 29/7/23 template *.mplstyle to be imported inside the environment 
# C:\ProgramData\Anaconda3\envs\pv_module_modeling\Lib\site-packages\matplotlib\mpl-data\stylelib
#TIPS
# matplotlibrc locates all the mplstyle located in the same folder (stylelib)
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files
# Available parameters showed through mpl.rcParams
# Could be modified and checked through mpl.rcParams["savefig.format"]= 'jpg'
# https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
# rcParams can be directly used as args inside funcs but only some funcs arg are in rcParams
# mplstyle allow only one value type per parameter (e.g. 'upper center) while func may use shortcut (e.g. 2)

#issue solved by reinstall it

#20/10/25 quick fix
mpl.style.use(['thesis'])
# at C:\Users\wsfm\OneDrive - Loughborough University\_Personal_Backup\python_repositories\venv\Lib\site-packages\matplotlib\mpl-data\stylelib

#loc giving freedom since change depending how many graphs
# Loc can also be a 2-tuple giving the coordinates of the lower-left corner of the legend in axes coordinates
# (in which case bbox_to_anchor will be ignored). @FM does not happen
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
_DEFAULT_KWARGS ={plt.legend: {'loc': (0, 0), #'best' 0, 'upper right' 1, 'upper left' 2, 'lower left' 3, 'lower right' 4,
                 #'right' 5, 'center left'	6, 'center right' 7, 'lower center'	8, 'upper center' 9, 'center' 10
                 #loc position change starting from bbox_to_anchor. l/r & u/l seems inverted when bbox_to_anchor
                 'bbox_to_anchor': (0, -1)  # box location (x,y) or (x, y, width, height)
                 }}

#DEV NOTE 16/6/21 bbox_to_anchor before: (1,-0.2) loc 1

_ITERABLE_CLASSES =[dict, list, tuple, np.ndarray, pd.Series, pd.DataFrame, pd.DatetimeIndex]

_FIG_FOLDER =  os.path.join(os.path.dirname(os.getcwd())+r"/outputs/figures/")
#os.getcwd()

def change_savefig(func):
    #DEV NOTE 8/4/21 in Jupyter f = change_savefig(plt.savefig) always required to catch title
    #DEV NOTE 8/4/21 plt.show does not seems to change that. use_default has not same behaviour
    ax = plt.gca()
    @functools.wraps(func)
    def get_name_from_title(folder:str=None, fname:str=None, *args, **kwargs):
        #DEV NOTE 6/4/21 not needed since needed parameters already in matplotlib.rcParams
        """kwargs_format = _DEFAULT_KWARGS[plt.savefig]
        for k, v in kwargs_format.items():
            if k not in kwargs.keys():
                kwargs[k] = kwargs_format[k]"""
        #DEV NOTE 14/5/21 old part could be removed
        """if 'fname' not in kwargs.keys():
            if folder is None:
                folder = _FIG_FOLDER
            #if title wrapped remove \n, not working if using '
            fname = ax.get_title().replace("\n", " ") + "." + kwargs['format']
            kwargs['fname'] = os.path.join(folder, fname)"""
        if fname is None:
            fname = ax.get_title().replace("\n", " ") + "." + mpl.rcParams['savefig.format']
        if folder is None:
            folder = _FIG_FOLDER
        # if title wrapped remove \n, not working if using '
        kwargs['fname'] = os.path.join(folder, fname)
        value = func(*args, **kwargs)
        print(f"{fname} saved at {folder}")
        #DEV NOTE 9/6/21 removing blank plot
        plt.show()
        return value
    return get_name_from_title


def use_default(func):
    @functools.wraps(func)
    # default decorators for some functions, e.g. matplotlib
    # for matplotlib not used if args already available through plt.style.use
    # https: // matplotlib.org / stable / tutorials / introductory / customizing.html  # customizing-with-matplotlibrc-files
    def func_default_args(*args, **kwargs):
        if func in _DEFAULT_KWARGS.keys():
            kwargs_format = _DEFAULT_KWARGS[func]
            # DEV NOTE 3/4/21 considering only kwargs not positional arguments
            for k, v in kwargs_format.items():
                #modify only if not explicitely specified
                if k not in kwargs.keys():
                    kwargs[k] = kwargs_format[k]
        value = func(*args, **kwargs)
        return value
    return func_default_args

def inoutput(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        value = func(*args, **kwargs)
        r = {}
        a = {}
        k = {}

        def get_info(x):
            c = x.__class__
            if c in _ITERABLE_CLASSES:
                v = f'len {len(x)}'
            elif c not in _ITERABLE_CLASSES:
                v = x
            return c, v
        for arg in args:
            c, v = get_info(arg)
            a[c]=v
        for val in value:
            c, v = get_info(val)
            r[c]=v
        for k, v in kwargs:
            c, v = get_info(v)
            k[k+" "+c]=v

        print(f'{func.__name__}'+
              f'('+",".join([f'{k}:{v}' for k, v in a.items()])+
              ",".join([f'{k}:{v}' for k, v in k.items()])+')='+
              ",".join([f'{k}:{v}' for k, v in r.items()]))
        return value
    return wrapper_timer


"""def get_info(x):
    c = x.__class__
    if c in _ITERABLE_CLASSES:
        v = f'len {len(x)}'
    elif c not in _ITERABLE_CLASSES:
        v = x
    return v


for x in value:
    o[x] = get_info(x)
for x in kwargs:
    i[x] = get_info(x)
print(f'{func.__name__!r}(' + ",".join([f'{k}:{v}' for k, v in i.items()]) + ")" +
      ",".join([f'{k}:{v}' for k, v in o.items()]))
return value"""


#https://realpython.com/primer-on-python-decorators/#a-few-real-world-examples
def timer(func):
    #preserving original function attributes
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = dt.datetime.now() # time.perf_counter()    # 1
        print(f"Started {func.__name__!r} at {str(start_time)}")  # {run_time:.4f} secs"
        value = func(*args, **kwargs)
        #end_time = time.perf_counter()      # 2
        run_time = dt.datetime.now() - start_time #end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {str(run_time)} secs") #{run_time:.4f} secs"
        return value
    return wrapper_timer

def add_tukey_fences(func):
        #DEV NOTE 7/4/21 counting of null and duplicates could be added as additional decorator. Check first describe source:
        #https://github.com/pandas-dev/pandas/blob/v1.2.3/pandas/core/generic.py#L10025-L10412
        @timer
        @functools.wraps(func)
        def describe_with_tukey_fences(df:pd.DataFrame, percentiles: List = [0.25, 0.75], *args, **kwargs):
            """
            Replace describe with addition of tukey fences creating a function
            :param df:
            :param percentiles:
            :param args:
            :param kwargs:
            :return:
            """
            p_factors: dict = {"25%": -1, "75%": +1}
            p_labels: dict = {"25%": 'fence_tukey_min', "75%": 'fence_tukey_max'}
            [percentiles.append(p) for p in [0.25, 0.75] if p not in percentiles]
            if isinstance(df, pd.DataFrame):
                df_desc = pd.DataFrame.describe(df, percentiles, *args, **kwargs)
            elif isinstance(df, pd.Series):
                df_desc = pd.Series.describe(df, percentiles, *args, **kwargs)
            df_index = df_desc.index.to_list()
            for k, v in p_factors.items():
                if isinstance(df, pd.DataFrame):
                    ft:List  = [df_desc.loc[k, c] + v * 1.5 * (df_desc.loc["75%", c] - df_desc.loc["25%", c]) for c in df_desc.columns]
                    df_desc = df_desc.append(dict(zip(df_desc.columns, ft)), ignore_index=True)
                elif isinstance(df, pd.Series):
                    ft = df_desc[k] + v * 1.5 * (df_desc["75%"] - df_desc["25%"])
                    df_desc = df_desc.append(pd.Series([ft]), ignore_index=True)
                df_index.append(p_labels[k])
                df_desc.index = df_index
            return df_desc
        return describe_with_tukey_fences



"""t = lambda x: [5, [x+4,x+5], x+6, pd.DataFrame([3,3])]
t1 = timer(inoutput(t))
t1(3)
"""


"""#testing
from textwrap import wrap
df = pd.DataFrame({"a":[1, 2, 3], "b":[5, 6, 7]})
title = str(dt.datetime.now().second)
title = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
df.plot(title='\n'.join(wrap(title,60)))

legend = use_default(plt.legend)
legend()
savefig = change_savefig(plt.savefig)
savefig() #fname='test.jpg')"""



# testing different positions
"""for i in range(0,11):
    df.plot()
    legend = use_default(plt.legend)
    legend(bbox_to_anchor= (1,-0.02*i))
    plt.title(i)
    #legend(bbox_to_anchor=(1.01, 0.89))
    #plt.legend()
    plt.show()"""



"""df = pd.DataFrame({"a":[1, 2, 3, 4, 5], "b":[7, 8, 9, 0, 11]})
df = pd.Series([1, 2, 3, 4, 5])
describe_with_tukey_fences = add_tukey_fences(pd.DataFrame.describe)
print(describe_with_tukey_fences(df, percentiles=[0.25, 0.75, 0.05, 0.95]))"""




