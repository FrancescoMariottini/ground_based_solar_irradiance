"""
Generic processes on csv & dataframe
Modified on 27/3/21
"""

#showing few specific methods imported although entire library imported anyway
from os import getcwd, path, listdir


#showing importing of entire library for readiness since many used
import pandas as pd
#DEV NOTE just for timedelta
import numpy as np

import datetime

#wraps for specific decorators
from functools import wraps
#importing decorators for generic ones
import decorators

#for showing visualisation when testing
import matplotlib.pyplot as plt
from typing import List, Dict

#for day saving time
from pytz import timezone

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None
#pd.options.display.max_colwidth = None

CSVS_STARTWITH = 'ch'
DATETIME_COLUMN = 'tmstamp'
_DATETIME_FORMATS = ['%d-%m-%y %H:%M:%S', '%d-%m-%y %H:%M', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M']
COLUMNS_TO_EXCLUDE = ['pt1000_ppuk_sensol']
GMIN = 20

#folder for py file testing
ISC_IRRADIANCE_FILES_FOLDER = path.join(path.dirname(getcwd())+r"/assets/isc_irradiance_files/")




dt_to_hour = lambda x: x.hour + x.minute / 60 + x.second / 3600

#17/5/21 IRRADIANCE PRELIMINARY ANALYSIS FILE TO SAVE CACHE ?


#month strings for visualisation
month_strs =[datetime.datetime(1,n,1).strftime("%B")[:3] for n in range(1,13)]

def tdhour(td: np.timedelta64):  # td function to get hours
    return (td.total_seconds() / 3600)

#older version
"""def add_hour(df:pd.DataFrame):
    for c in df.columns.to_list():
        t = df[c].dtype
        if str(t) == 'datetime64[ns]':
            df[c + '_h'] = df[c].apply(lambda x: x.hour + x.minute / 60 + x.second / 3600)
    return df"""

def add_hour(datetimes):
    if isinstance(datetimes, pd._libs.tslibs.timestamps.Timestamp):
        datetimes = dt_to_hour(datetimes)
    if isinstance(datetimes, pd.DataFrame):
        for c in datetimes.columns.to_list():
            t = str(datetimes.loc[:,c].dtype)
            if "datetime" in t:
                datetimes[c+'_h'] = datetimes[c].apply(dt_to_hour)
    elif isinstance(datetimes, pd.Series):
        datetimes =datetimes.apply(dt_to_hour)
    elif isinstance(datetimes, pd.DatetimeIndex):
        datetimes = dt_to_hour(datetimes)
    return datetimes



def reset_tzi_convert_tze(dtis, tzi:str=None, tze:str=None, dti_columns:List[str]=None):
    def dti_reset_tzi_convert(dti):
        #DEV NOTE 17/5/21 if None input not provided
        #tranform into dt if not already
        dti = dti if isinstance(dti, pd.DatetimeIndex) else pd.DatetimeIndex(dti)
        #if dti.tz != tze:
        if dti.tz != tzi:
            #https://stackoverflow.com/questions/16628819/convert-pandas-timezone-aware-datetimeindex-to-naive-timestamp-but-in-certain-t
            #removint timezone before localizing into the desidered starting timezone
            dti = dti.tz_localize(tz=None) if tzi is not None else dti
            dti = dti.tz_localize(tz=tzi, ambiguous='NaT', nonexistent='NaT') if tzi is not None else dti
        if tzi != tze:
            # no need ambiguous = 'NaT', nonexistent='NaT' since None ?
            dti = dti.tz_convert(tz=tze) if tzi is not None else dti.tz_localize(tz=tze)
        return dti
    if isinstance(dtis, pd.DataFrame):
        if dti_columns is None: dti_columns = dtis.columns.to_list()
        for c in dti_columns:
            dtis[c] = dti_reset_tzi_convert(dtis[c])
    else:
        dtis = dti_reset_tzi_convert(dtis)
    return dtis


def get_dst_steps(dt: pd.Series, tz: str, years=List[int]):
    df = pd.DataFrame({"dt": dt, "date": pd.DatetimeIndex(dt).date}, index=dt.index)
    tz = timezone(tz)
    utc_tt = tz._utc_transition_times
    dst_steps = {}
    dst_steps_dt = {}
    for y in years:
        tt_y = [tt for tt in utc_tt if tt.year == y]

        def get_max_step_before_after(df, date):
            df_ba = df[(df.date == date) | (df.date == date - datetime.timedelta(days=1))].copy()
            if len(df_ba) > 1:
                df_ba['step'] = df_ba.dt - df_ba.dt.shift(1)
                df_ba.step = df_ba.step.apply(lambda x: x.total_seconds())
                return df_ba.step.max(skipna=True), df_ba.loc[df_ba.step.idxmax(skipna=True), "dt"]
            else:
                return None, None

        dst_steps[f"{y}_dst"], dst_steps_dt[f"{y}_dst"] = get_max_step_before_after(df, tt_y[0].date())
        dst_steps[f"{y}_ndst"], dst_steps_dt[f"{y}_ndst"] = get_max_step_before_after(df, tt_y[1].date())

    return dst_steps, dst_steps_dt

#def df_split_dst(df:pd.DataFrame, sl:SolarLibrary, tzi='utc'):
def df_split_dst(df:pd.DataFrame, sl_tz:str, tzi='utc', dt_column=None):
    #index must be datetime
    #splitting between summer and winter time
    if sl_tz.upper() == 'UTC': raise AttributeError("No DST split for utc")
    df_s = pd.DataFrame()
    df_w = pd.DataFrame()
    tz = timezone(sl_tz)
    utc_tt = tz._utc_transition_times
    #storing previous index
    df["index_o"] = df.index
    #if dt column provided used instead of index
    if dt_column is None:
        #if dates shift since converting as starting of date
        if isinstance(df.index.to_list()[0], datetime.date):
            s = datetime.timedelta(hours=12)
        else:
            s = datetime.timedelta(hours=0)
        if isinstance(df.index,pd.DatetimeIndex) is False:
            df.index = pd.DatetimeIndex(df.index)
    elif dt_column is not None:
        #DEV NOTE 9/6/21 added
        #remove name from dt before converting into index to avoid conflicts
        df.index = pd.DatetimeIndex(df[dt_column].rename(None))
        #assumed no date at dt
        s = datetime.timedelta(hours=0)
    #16/5/21 assigning tzi if not already (e.g. date) removing tz for comparison with utc without timezone
    dti = reset_tzi_convert_tze(df.index, tzi=tzi, tze=None)
    df.index = dti + s
    #for y in np.unique(df.index.year):
    #adding year to avoid warning on index.year
    df["year"] = dti.year
    #DEV NOTE 9/6/21 not clear why many nan added but removed
    for y in np.unique(dti.year.dropna()):
        tt_y = [tt for tt in utc_tt if tt.year==y]
        dst_t = df[(df.year==y)&(df.index >= tt_y[0])&(df.index < tt_y[1])]
        ndst_t = df[(df.year==y)&((df.index < tt_y[0])|(df.index >= tt_y[1]))]
        if len(dst_t) > 1:
            if len(df_s) > 1:
                df_s = df_s.append(dst_t)
            else:
                df_s = dst_t
        if len(ndst_t) > 1:
            if len(df_w) > 1:
                df_w = df_w.append(ndst_t)
            else:
                df_w = ndst_t
    for dfs in [df_s, df_w]:
        #not for empty df
        if len(dfs) > 0:
            dfs.set_index(keys="index_o", drop=True, inplace=True)
            dfs.drop(columns="year", inplace=True)
    return df_s, df_w
    #to be checked with reset_index could be useful
    #return df_dst.reset_index(drop=True), df_ndst.reset_index(drop=True)


@decorators.timer
def clean_df(df: pd.DataFrame, dt_format='%d/%m/%Y %H:%M', cln_order=["dt", "gpoa"]):
    df.rename(dict(zip(df.columns.to_list(), cln_order)), inplace=True, axis=1)
    df.drop(columns=[c for c in df.columns.to_list() if c not in ["dt", "gpoa"]], inplace=True)
    #extracting raw duplicates before any transformation
    dup_condition = df.duplicated(subset='dt', keep=False) == True
    df_duplicates = df.loc[dup_condition,:].sort_values(by='dt')
    # kept last based on post-analysis
    # DEV NOTE 8/4/21 Is a decorator to remove duplicates with na values even before dropna useful ?
    df.drop_duplicates(subset='dt', keep='last', inplace=True)
    #converting to nan instead of raising error
    df.dt = pd.to_datetime(arg=df.dt, format=dt_format, errors="ignore")
    #tz slower processing but still useful
    #DEV NOTE overkill since renaming not working
    print(f'{len(df_duplicates)} duplicates') if len(df_duplicates) > 0 else None
    return df.sort_index(), df_duplicates

#fast no timer needed
def groupby_with_utc(dfc:pd.DataFrame, tzi='utc', max_hour=12, gmin=GMIN):
    #17/5/21 for irradiance but could be generalised
    #fast approach for easy identification of dt or tz issues
    #could not be merged with data completeness since filter 20 used
    #utc_hour_max approximation but should be a precise datetime
    dfv = dfc.copy(deep=True)
    #whatever datetime is provided reset to assumed tz before converting to utc
    dfv.index  = reset_tzi_convert_tze(dfv.dt.rename("dt_utc"), tzi, 'utc')
    #removing all na after conversion
    dfv.dropna(inplace=True)
    dfv = dfv[dfv.gpoa > gmin]
    dfv['date'] = dfv.index.date
    dfv['dt_h'] = add_hour(dfv.index)
    #not clear why not conflict with index name
    dfv['dt_utc'] = dfv.index
    #only essential aggregation
    #DEV NOTE 9/5/21 overlapping groups to ensure maximum is around 12 not working if original dataset not utc
    #DEV NOTE 20/5/21 no need for max precision since misalignment not checked in this function
    max_h_precision = 0
    dfm_mm = dfv[dfv.dt_h <= max_hour+max_h_precision].groupby(["date"]).agg(
        {'dt_utc':'first', 'gpoa': ['idxmin', 'idxmax', 'max']})
    dfa_mm = dfv[dfv.dt_h >= max_hour-max_h_precision].groupby(["date"]).agg(
        {'dt_utc':'last', 'gpoa': ['idxmin', 'idxmax', 'max']})
    #introducing max for morning and afternoon
    dly_mm = pd.concat([dfm_mm.rename(columns={'first':'dtfirst', 'idxmin':'dtmmin', 'idxmax': 'dtmmax', 'max':'mmax'}),
                       dfa_mm.rename(columns={'last':'dtlast', 'idxmin':'dtamin', 'idxmax': 'dtamax', 'max':'amax'})],
                      join='inner', axis=1).droplevel(level=0, axis=1)
    dly_mm["dtmax"] = dly_mm.apply(lambda x: x['dtmmax'] if x['mmax'] > x['amax'] else x['dtamax'], axis=1)
    return dly_mm

#elements not in the first series
def series_differences(s1: pd.Series, s2: pd.Series):
    #initialise empty Series rather than None (not valid index)
    dff = pd.Series(dtype=float, name=s2.name)
    if s2 is not None and len(s2)>0:
        if s1 is None or len(s1) == 0:
            dff = s2
        elif len(s1) > 0:
            s2_name = s2.name
            s1.name = 's1'
            s2.name = 's2'
            df = pd.concat([s2, s1], axis=1, join='outer')
            df.dropna(subset=['s2'], inplace=True)
            df["dff"] = df.apply(lambda x: x['s2'] if x['s2'] != x['s1'] else None, axis=1)
            df.dropna(subset=['dff'], inplace=True)
            if len(df) > 0:
                dff = df.dff
                dff.name = s2_name
    return dff

### OLD data operations functions



@decorators.timer
def merge_datetime_csvs(csvs_folderpath: str,
               csvs_startwith: str = CSVS_STARTWITH,
               datetime_column: str = DATETIME_COLUMN,
               datetime_formats: List[str] = _DATETIME_FORMATS,
               columns_to_exclude: List[str] = COLUMNS_TO_EXCLUDE
                ) -> pd.DataFrame:
    """
    Retrieve and merge different csv having a datetimecolumn.
    :param csvs_folderpath:
    :param csvs_startwith:
    :param datetime_column:
    :param datetime_formats:
    :param columns_to_exclude:
    :return:
    """
    filenames = [f for f in listdir(csvs_folderpath) if f.endswith(r".csv") and f.startswith(csvs_startwith)]
    df = pd.DataFrame()
    for filename in filenames:
        suffix = '_'+filename.replace(csvs_startwith,'').replace(".csv", '')
        errors = []
        #DEV NOTE 6/4/21 format dt not used anymore relying on to_dt. Function still in decorators_partial_tst
        #df_tmp = read_csv_format_datetime(filepath= path.join(csvs_folderpath, filename),
        #                             datetime_column=datetime_column, datetime_formats=datetime_formats)
        df_tmp = pd.read_csv(filepath_or_buffer=path.join(csvs_folderpath, filename))
        df_tmp[datetime_column] = pd.to_datetime(df_tmp[datetime_column])
        if len(errors) > 0:
            print(f'{filename} {datetime_column} column not compatible with formats: {",".join([f for f in errors])}')
        if len(errors) == 0:
            df_tmp.drop(columns=[c for c in columns_to_exclude if c in df_tmp.columns], inplace=True)
            c_to_rename = [c for c in df_tmp.columns.tolist() if c != datetime_column]
            df_tmp.rename(columns=dict(zip(c_to_rename, [c + suffix for c in c_to_rename])), inplace=True)
            if len(df) > 0:
                try:
                    df = df.merge(df_tmp, left_on=datetime_column, right_on=datetime_column) #, suffixes=(None, suffix))
                except KeyError:
                    print(f'{filename} could not be merged')
                    next
            elif len(df) == 0:
                df = df_tmp.copy(deep=True)
            print(f'{filename} adding columns: {",".join([c for c in df_tmp.columns])}')

    return df

@decorators.timer
def get_min_max_by_date(values_datetime_indexed: pd.Series,
                        parameter: str = "value",
                        derivative2_only: bool = True,
                        resolution_s=3600):
    """
    Get min and max values and derivatives (hourly resolution) for a series having a datetime index
    :param values:
    :param parameter:
    :param derivative2_only:
    :return:
    """
    #define y column
    y_cln = parameter
    yg_cln = y_cln+"_grd"
    y2d_cln = y_cln+"_2d"
    #define columns for group by
    gb_clns = [y_cln, yg_cln , y2d_cln, "datetime", "date"]
    #not only positive values since variation matters rather then minimum
    #values = values[values>0]
    dates = [dt.date() for dt in values_datetime_indexed.index]
    df = pd.DataFrame({y_cln:values_datetime_indexed, "date":dates, "datetime":values_datetime_indexed.index}, index=values_datetime_indexed.index)
    df.loc[:, "hour"] = [dt.hour for dt in df.index]
    #identify values for which previous hrly value missing (returning them?)
    #df.loc[(df.date == df.date.shift(1)) &
    #        (df["datetime"] - df["datetime"].shift(1) != pd.Timedelta(1, unit='h'))]
    #splittig morning and afternoon because gradient calculated differently (due to data availability)
    #for the morning based on next hour, for the afternoon based on previous hour 
    #thus initial overlapping intervals treated later
    df_m = df.loc[df.hour<= 13,:]
    df_a = df.loc[df.hour>= 11,:]
    #keeping only values for which previous/next value is available respectively for morning/afternoon
    df_m = df_m.loc[df_m["datetime"] - df_m["datetime"].shift(1) == pd.Timedelta(resolution_s, unit='s')]
    df_a = df_a.loc[df_a["datetime"].shift(-1) - df_a["datetime"] == pd.Timedelta(resolution_s, unit='s')]
    #gradient is calculated only within the same day
    df_m.loc[df_m.date == df_m.date.shift(1), yg_cln] = df_m.loc[:, y_cln] - df_m.loc[:, y_cln].shift(1)
    df_a.loc[df_a.date == df_a.date.shift(-1), yg_cln] = df_a.loc[:,y_cln].shift(-1) - df_a.loc[:,y_cln]
    #second derivative is calculated to identify elbow
    #next/previous value are retrieved for morning/afternoon respectively
    df_m.loc[df_m.date == df_m.date.shift(1), y2d_cln] = df_m.loc[:, yg_cln].shift(-1) - df_m.loc[:, yg_cln]
    df_a.loc[df_a.date == df_a.date.shift(-1), y2d_cln] = df_a.loc[:, yg_cln] - df_a.loc[:, yg_cln].shift(1)

    """
    aggregators = [max, min, 'idxmax']
    if derivative2_only:
        aggregators += ['idxmin']
    g_m = df_m.loc[df_m.hour<= 12, gb_clns].groupby(["date"]).agg(aggregators)
    g_a = df_a.loc[df_a.hour>= 12, gb_clns].groupby(["date"]).agg(aggregators)
    #22/12/20 tmp grouping to identify daily max only (redundant with m and a max)
    g = df.groupby(["date"]).agg(['max', 'idxmax', 'len'])
    """

    g_m = df_m.loc[df_m.hour <= 12, gb_clns].groupby(["date"]).agg({y2d_cln: ['max', 'idxmax']})
    g_a = df_a.loc[df_a.hour >= 12, gb_clns].groupby(["date"]).agg({y2d_cln: ['max', 'idxmax']})
    g = df.groupby(["date"]).agg({'datetime':['min', 'max', len], y_cln: ['max', 'idxmax', lambda x: x.isnull().sum()]})

    """
    def get_hour_day_max(x):
        if g_m.loc[x, (y2d_cln, 'max')] >= g_a.loc[x, (y2d_cln, 'max')]:
            idxmax = g_m.loc[x, (y_cln, 'idxmax')]
        elif g_m.loc[x, (y2d_cln, 'max')] < g_m.loc[x, (y2d_cln, 'max')]:
            idxmax = g_a.loc[x, (y_cln, 'idxmax')]
        return idxmax.hour
    """

    df_hrs = pd.DataFrame({'date': g.index,
                         'len': g.loc[:, ('datetime', 'len')],
                         'null': g.loc[:, (y_cln, '<lambda_0>')],
                         'hour_min': g.loc[:, ('datetime', 'min')].transform(lambda x: x.hour),
                          'hour_' + y2d_cln + '_max_morning': g_m.loc[:, (y2d_cln, 'idxmax')].transform(
                          lambda x: x.hour),
                          y2d_cln + '_max_morning': g_m.loc[:, (y2d_cln, 'max')],
                          #'hour_'+y_cln+'_max_morning': g_m.loc[:, (y_cln, 'idxmax')].transform(lambda x: x.hour),
                          #y_cln+'_max_morning': g_m.loc[:, (y_cln, 'max')],
                          'hour_'+y_cln+'_max': g.loc[:, (y_cln, 'idxmax')].transform(lambda x: x.hour),
                          y_cln+'_max': g.loc[:, (y_cln, 'max')],
                          #'hour_'+y_cln+'_max_afternoon': g_a.loc[:, (y_cln, 'idxmax')].transform(lambda x: x.hour),
                          #y_cln+'_max_afternoon': g_a.loc[:, (y_cln, 'max')],
                          'hour_' + y2d_cln + '_max_afternoon': g_a.loc[:, (y2d_cln, 'idxmax')].transform(lambda x: x.hour),
                          y2d_cln + '_max_afternoon': g_a.loc[:, (y2d_cln, 'max')],
                         'hour_max': g.loc[:, ('datetime', 'max')].transform(lambda x: x.hour)})

    """
    df_hrs = pd.DataFrame({'date': g_a.index,
                         'hour_min': g_m.loc[:, ('datetime', 'min')].transform(lambda x: x.hour),
                          'hour_' + y2d_cln + '_max_morning': g_m.loc[:, (y2d_cln, 'idxmax')].transform(
                          lambda x: x.hour),
                          y2d_cln + '_max_morning': g_m.loc[:, (y2d_cln, 'max')],
                          #'hour_'+y_cln+'_max_morning': g_m.loc[:, (y_cln, 'idxmax')].transform(lambda x: x.hour),
                          #y_cln+'_max_morning': g_m.loc[:, (y_cln, 'max')],
                          'hour_'+y_cln+'_max': g.loc[:, (y_cln, 'idxmax')].transform(lambda x: x.hour),
                          y_cln+'_max': g.loc[:, (y_cln, 'max')],
                          #'hour_'+y_cln+'_max_afternoon': g_a.loc[:, (y_cln, 'idxmax')].transform(lambda x: x.hour),
                          #y_cln+'_max_afternoon': g_a.loc[:, (y_cln, 'max')],
                          'hour_' + y2d_cln + '_max_afternoon': g_a.loc[:, (y2d_cln, 'idxmax')].transform(lambda x: x.hour),
                          y2d_cln + '_max_afternoon': g_a.loc[:, (y2d_cln, 'max')],
                         'hour_max': g_a.loc[:, ('datetime', 'max')].transform(lambda x: x.hour)})
    """

    if derivative2_only is False:
        #optional values not used
        #18/12/20 sign inversion to be checked
        # optional min since may be related to diffuse only component (bad clear sky model performance)
        df_hrs.loc['hour_'+y_cln+'_min_morning'] = g_m.loc[:, (y_cln, 'idxmin')].transform(lambda x: x.hour)
        df_hrs.loc[y_cln + '_min_morning'] = g_m.loc[:, (y_cln, 'min')]
        df_hrs.loc['hour_'+y_cln+'_min_afternoon'] = g_a.loc[:, (y_cln, 'idxmin')].transform(lambda x: x.hour)
        df_hrs.loc[y_cln+'_min_afternoon'] = g_a.loc[:, (y_cln, 'min')]
        # optional min grad since may be related to temporary stable cloudy condition
        df_hrs.loc['hour_' + yg_cln + '_min_morning'] = g_m.loc[:, (yg_cln, 'idxmin')].transform(lambda x: x.hour)
        df_hrs.loc[yg_cln + '_min_morning'] = g_m.loc[:, (yg_cln, 'min')]
        df_hrs.loc['hour_' + yg_cln + '_min_afternoon'] = g_a.loc[:, (yg_cln, 'idxmin')].transform(lambda x: x.hour)
        df_hrs.loc[yg_cln + '_min_afternoon'] = g_a.loc[:, (yg_cln, 'min')]
        # optional gradient since do not identify elbows
        df_hrs.loc['hour_' + yg_cln + '_max_morning'] = g_m.loc[:, (yg_cln, 'idxmax')].transform(lambda x: x.hour)
        df_hrs.loc[yg_cln + '_max_morning'] = g_m.loc[:, (yg_cln, 'max')]
        df_hrs.loc['hour_' + yg_cln + '_max_afternoon'] = g_a.loc[:, (yg_cln, 'idxmax')].transform(lambda x: x.hour)
        df_hrs.loc[yg_cln + '_min_afternoon'] = -g_a.loc[:, (yg_cln, 'max')]
    return df_hrs

@decorators.timer
def datacompleteness(values_datetime_indexed: pd.Series,
                    parameter: str = "value",
                    time_resolution_s_max=1,
                    datetime_column='datetime',
                    get_hour=True):
    """
    Data completeness check. Higher resolution than clear sky identification
    :param values_datetime_indexed:
    :param time_resolution_s_max:
    :param datetime_column:
    :return:
    """
    # assign series name if not given
    if values_datetime_indexed.name is None:
        values_datetime_indexed.name = 'value'
    #only as reminder steps to merge dataframe with a complete time series
    #datetimeindex_utc = pd.DatetimeIndex(dataframe.loc[:, datetimecolumn].values, ambiguous='NaT', name="datetime")
    #dataframe.index = datetimeindex_utc.to_series(name="index")  # 26/2/20 .index remove the time
    d = values_datetime_indexed.index.date
    #extracting complete time range, offset to cover until 0:00 of next day
    _ = pd.date_range(start=min(d), end=max(d) + pd.DateOffset(1), freq=str(time_resolution_s_max) + "s", name="datetime_all",
                      tz=values_datetime_indexed.index.tz)
    #removing 0:00 of next day
    #FutureWarning: The default of the 'keep_tz' keyword will change to True in a future release.
    utc_all = _.to_series(index=None, name=datetime_column, keep_tz=True)[:-1]
    df = pd.concat([values_datetime_indexed, utc_all], join='outer', sort=True, axis=1)
    #index already datetimeindex no need to transform it
    #date for grouping
    df["date"] = df.index.date
    def my_agg(x):
        s = x[values_datetime_indexed.name]
        dt_notnull_x = x.loc[s.notnull(), datetime_column]
        if len(dt_notnull_x) == 0:
            x_first, x_last, x_max = None, None, None
        elif len(dt_notnull_x) > 0:
            x_first, x_last, x_max = min(dt_notnull_x), max(dt_notnull_x), s.idxmax()
        #DEV NOTE 18/4/21 TBI: null is union from original resolution and requested one
        names = {"null":len(x[x[values_datetime_indexed.name].isnull()]),
                "first_datetime": x_first,
                "last_datetime": x_last,
                "max_datetime": x_max}
        return pd.Series(names, index=['null', 'first_datetime', 'last_datetime', 'max_datetime'])
    df_daily = df.groupby('date').apply(my_agg)
    # storing daily na to recover them after
    df_daily_na = df_daily[df_daily.isna().any(axis=1)]
    #removing na to calculate disconnections on multiple days
    df_daily.dropna(subset=["first_datetime","last_datetime"], inplace=True)
    df_daily["after_last_hours"] = df_daily["first_datetime"].shift(-1) - df_daily["last_datetime"]  # 28/3/19 DEV no shift immediately
    df_daily["before_first_hours"] = df_daily["first_datetime"] - df_daily["last_datetime"].shift(1)
    # df_daily["after_off_hours"]=df_daily["after_off_hours"].apply(lambda x: tdhour(x))
    df_daily["after_last_hours"] = df_daily["after_last_hours"].transform(lambda x: tdhour(x))
    # df_daily["before_on_hours"]=df_daily["before_on_hours"].apply(lambda x: tdhour(x)) #End of parameters calculation
    df_daily["before_first_hours"] = df_daily["before_first_hours"].transform(lambda x: tdhour(x))
    if get_hour:
        def get_hour(x):
            return x.hour + x.minute / 60 + x.second /3600
        df_daily["first_hour"] = df_daily["first_datetime"].apply(get_hour)
        df_daily["last_hour"] = df_daily["last_datetime"].apply(get_hour)
        df_daily["max_hour"] = df_daily["max_datetime"].apply(get_hour)
    # FutureWarning: Sorting because non-concatenation axis is not aligned.
    # A future version of pandas will change to not sort by default.
    df_daily = pd.concat([df_daily, df_daily_na], sort=True).sort_index(ascending=True)
    #DEV NOTE 18/4/21: daily disconnection ('different than null') could be added
    return df_daily

"""ARCHIVED FUNCTIONS"""
DATETIME_FORMATS = ['%d-%m-%y %H:%M:%S', '%d-%m-%y %H:%M', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M']
def csv_read_datetime_format(filepath: str,
                             datetime_column: str,
                             datetime_formats: List[str] = DATETIME_FORMATS):
    wrong_formats = []
    df_tmp = pd.DataFrame()
    for dt_f in datetime_formats:
        try:
            converters = {datetime_column: lambda x: pd.to_datetime(x, format=dt_f, errors='raise')}
            df_tmp = pd.read_csv(filepath, delimiter=",", converters=converters)
            wrong_formats = []
            break
        except ValueError:
            wrong_formats += [dt_f]
    return df_tmp, wrong_formats

def datetimeindex_to_utc(datetimeindex: pd.DatetimeIndex) -> np.ndarray:
    #DEV NOTE 2/4/21 is pandas.core.indexes.numeric.Float64Index
    # DatetimeIndex conversion into numpy array [pvlib spa.py]
    # specific returned type is pandas.core.indexes.numeric.Float64Index
    return datetimeindex.astype(np.int64)/10**9

#equivalent to the one above but still used separately in clear sky
def datetime_to_utc(datetime: datetime) -> float:
    # see timestamp at https://docs.python.org/3/library/datetime.html
    return datetime.replace(tzinfo=timezone.utc).timestamp()




"""TESTING"""

#WINDOWS TESTING
_WINDOWS_TESTING = 'completeness_rm'
_WINDOWS_TESTING = 'minmax_lb'
_WINDOWS_TESTING = ''

if _WINDOWS_TESTING == 'minmax_lb':
    RM_FOLDER = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/BAYWARE/doc_shared/_ST/Fiumicino/"
    RM_PYR_C1 = "Pyranometer Cab 1_fm.csv"
    RM_PYR_C4 = "Pyranometer Cab 4_fm.csv"
    df_rm1 = pd.read_csv(filepath_or_buffer=path.join(RM_FOLDER, RM_PYR_C1))
    df = df_rm1.copy(deep=True)
    df.rename(dict(zip(df.columns.to_list(), ["dt", "gpoa"])), inplace=True, axis=1)
    dup_condition = df.duplicated(subset='dt', keep=False) == True
    df_duplicates = df.loc[dup_condition, :].sort_values(by='dt')
    df.drop_duplicates(subset='dt', keep='last', inplace=True)
    df.dt = pd.to_datetime(df.dt, format='%d/%m/%Y %H:%M')
    df.index = pd.DatetimeIndex(df.dt.values)
    df_dly_mm = get_min_max_by_date(values_datetime_indexed=df.gpoa)



if _WINDOWS_TESTING == 'completeness_rm':
    RM_FOLDER=  r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/BAYWARE/doc_shared/_ST/Fiumicino/"
    RM_PYR_C1="Pyranometer Cab 1_fm.csv"
    RM_PYR_C4="Pyranometer Cab 4_fm.csv"
    df_rm1 = pd.read_csv(filepath_or_buffer=path.join(RM_FOLDER, RM_PYR_C1))
    df = df_rm1.copy(deep=True)
    df.rename(dict(zip(df.columns.to_list(), ["dt", "gpoa"])), inplace=True, axis=1)
    # extracting raw duplicates before any transformation
    dup_condition = df.duplicated(subset='dt', keep=False) == True
    df_duplicates = df.loc[dup_condition, :].sort_values(by='dt')
    # kept last based on post-analysis
    # DEV NOTE 8/4/21 Is a decorator to remove duplicates with na values even before dropna useful ?
    df.drop_duplicates(subset='dt', keep='last', inplace=True)
    df.dt = pd.to_datetime(df.dt)
    # tz not considered for faster processing
    df.index = pd.DatetimeIndex(df.dt.values)

    dly_rm4c = datacompleteness(df.gpoa, time_resolution_s_max=3600)



if _WINDOWS_TESTING == 'completeness':
    from os.path import join

    ISC_IRRADIANCE_FILES_FOLDER = join(path.dirname(getcwd()) + r"/assets/isc_irradiance_files/")
    gpoa = pd.read_csv(filepath_or_buffer=join(ISC_IRRADIANCE_FILES_FOLDER, 'chpoa_y15-20.csv'))
    gpoa.columns
    # %%
    gpoa.date_trunc = pd.to_datetime(gpoa.date_trunc)
    gpoa.index = pd.DatetimeIndex(gpoa.date_trunc, ambiguous='NaT', name="datetime")
    print(min(gpoa.index), max(gpoa.index))


    #filenames = [f for f in listdir(ISC_IRRADIANCE_FILES_FOLDER) if f.endswith(r".csv") and f.startswith(r"chpoa_y")]
    #filename = filenames[0]
    filename = "chpoa_y15-20.csv"

    gpoa_y = pd.read_csv(filepath_or_buffer=join(ISC_IRRADIANCE_FILES_FOLDER, filename))
    gpoa_y.date_trunc = pd.to_datetime(gpoa_y.date_trunc)

    gpoa_y.index = pd.DatetimeIndex(gpoa_y.date_trunc, ambiguous='NaT', name="datetime")

    test = datacompleteness(gpoa_y.g_cmp11_ppuk, time_resolution_s_max=3600)

    PV_FOLDER = r"C:/Users/wsfm/OneDrive - Loughborough University/_Personal_Backup/_Research_Master/_Data Sources/Skytron_Data_TEST/"
    PV_FILE = r"devicedata_0921.csv"

    #csvimp = pd.read_csv(filepath_or_buffer=PV_FOLDER + PV_FILE, delimiter=";", skip_blank_lines=True, header=11, nrows=5)
    #csvimp.date = pd.to_datetime(csvimp.date)




if _WINDOWS_TESTING == 'years':
    from os.path import join

    filenames = [f for f in listdir(ISC_IRRADIANCE_FILES_FOLDER) if f.endswith(r".csv") and f.startswith(r"chpoa_y")]
    gpoa = pd.DataFrame()

    for f in filenames:
        gpoa_y = pd.read_csv(filepath_or_buffer=join(ISC_IRRADIANCE_FILES_FOLDER, f))
        gpoa_y.date_trunc = pd.to_datetime(gpoa_y.date_trunc)
        gpoa = gpoa.append(gpoa_y, ignore_index=True)

    gpoa.sort_values(by='tmstamp', inplace=True)
    gpoa_dly = get_min_max_by_date(values=pd.Series(gpoa.g_cmp11_ppuk.values, index=gpoa.date_trunc.values))
    # overview of first and last sun ray on the sensor
    gpoa_dly.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])


if _WINDOWS_TESTING == 'merge':
    df = merge_datetime_csvs(csvs_folderpath=ISC_IRRADIANCE_FILES_FOLDER,
                            columns_to_exclude=['pt1000_ppuk_sensol'],
                         datetime_column= "date_trunc")


if _WINDOWS_TESTING == 'df_hrs':
    #generating overview
    CLEAR_SKY_CREST_EXAMPLES =["2019-02-14", "2019-04-21", "2019-06-26", "2019-06-27", "2019-06-28", "2019-06-29", "2019-11-29"]

    import re
    df_gpoa = pd.read_csv(filepath_or_buffer=join(ISC_IRRADIANCE_FILES_FOLDER, 'chpoa.csv'))
    df_gpoa.date_trunc = pd.to_datetime(df_gpoa.date_trunc)

    #columns_to_drop = [c for c in df_gpoa.columns.tolist() if len(re.findall('mt|cha|tms|iv_sweep', c))>0]
    #negative also included since difference considered and not minimum
    df_hrs = get_min_max_by_date(values= pd.Series(df_gpoa.g_cmp11_ppuk.values, index = df_gpoa.date_trunc.values))
    print(df_hrs)

