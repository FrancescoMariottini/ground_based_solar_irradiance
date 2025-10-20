import pandas as pd
import datetime

import pvlib

# 20/10/25 quick fix
# import data_operations as dtop
# import clear_sky as csky
# import decorators
import irradiance_source_py.data_operations as dtop
import irradiance_source_py.clear_sky as csky
import irradiance_source_py.decorators as decorators

from typing import List, Dict


#fast enough no @decorators.timer
def compare_with_sunpath(dly_mm:pd.DataFrame, sl:csky.SolarLibrary, tzi='utc', sun_rise_set_transit:pd.DataFrame=None,
                         delays_difference_max_h:int=None, utc_out=True):
    #dly_mmf is usually in utc due to DST splitting. dly_mm in dtop
    dly_mmf = dly_mm.copy(deep=True)
    #simple filter condition that minimum are extreme
    dly_mmf = dly_mmf[(dly_mmf.dtfirst == dly_mmf.dtmmin)&(dly_mmf.dtlast == dly_mmf.dtamin)]
    if len(dly_mmf) > 0:
        #26/5/21 better keep utc
        """for c in dly_mmf.columns.to_list():
            if "dt" in c:
                try:
                    dly_mmf[c] = dtop.reset_tzi_convert_tze(dly_mmf[c], tzi, sl.timezone)
                except TypeError:
                    print(f'{c} not converted from {tzi} to {sl.timezone}')"""
        if sun_rise_set_transit is not None:
        # since filter before not necessary due to merge: sun_rise_set_transit.index.equals(dly_mmf.index)
            srst = sun_rise_set_transit
        else:
            #None index of dates localize utc before converting timezone
            dti = dtop.reset_tzi_convert_tze(pd.DatetimeIndex(dly_mmf.index), tzi, sl.timezone)
            srst = pvlib.solarposition.sun_rise_set_transit_spa(dti,
                    sl.latitude, sl.longitude, how='numpy', delta_t=sl.delta_t, numthreads=sl.numthreads)
            #26/5/21 keeping utc time
            for c in ["sunrise", "sunset", "transit"]:
                #converting to utc time
                dti = pd.DatetimeIndex(srst[c]).tz_convert('utc')
                srst[c] = dti
                if c == "transit": srst.index = dti.date
                #using transit as reference for the date

        try:
            cmp_sp = dly_mmf.merge(srst, left_index=True, right_index=True, how='inner')
        except:
            print(r"error at dly_mmf.merge(srst, left_index=True, right_index=True, how='inner')")
            raise pd.errors.MergeError
        #sunrise, sunset and suntransit analysis fit more horizontal south oriented than tilted
        #however if obstacles in particular direction (mountain for eurac) delay still depends on day
        cmp_sp["delay_first_h"] = cmp_sp.apply(lambda x: (x["dtfirst"]-x["sunrise"]).total_seconds()/3600, axis=1)
        cmp_sp["delay_last_h"] = cmp_sp.apply(lambda x: (x["dtlast"]-x["sunset"]).total_seconds()/3600, axis=1)
        cmp_sp["delay_transit_h"] = cmp_sp.apply(lambda x: (x["dtmax"]-x["transit"]).total_seconds()/3600, axis=1)
        #filtering
        #sun transit not similar delay but at least should be symmetrical minus misalignment
        # ideally: transit = sunrise + (sunset - sunrise)/2
        # practically: centre = sunrise + err(sunrise) + (sunset - sunrise + delay_diff_err) /2
        cmp_sp["delay_centre_h"] =  cmp_sp.apply(lambda x:
        (x["dtmax"]-(x["dtmmin"]+(x["dtamin"]-x["dtmmin"])/2)).total_seconds()/3600, axis=1)
        if delays_difference_max_h is not None:
            #4/6/21 complex filter seems to not impact results (not in a positive way anyway)
            #4/6/21 better to simply use the additional parameter of centre
            #condition not used since not working well if shifted dataset due to not being centered at 12:00
            #(dly_mmf.dtmmax == dly_mmf.dtamax)]
            #DEV NOTE 16/5/21 TBR: expected not working well if not symmetric delay due to geographical features (e.g. mountain)
            #even if timezone shift and/or mountain, expected having same direction and magnitude minus error on delays
            #err = misalignment + geographical features
            cmp_sp = cmp_sp[abs(cmp_sp.delay_last_h-cmp_sp.delay_first_h)<=delays_difference_max_h]
            #20/5/21 too strong filter removing entire months, not used
            #3/6/21 not used since too difficult to explain in thesis
            #cmp_sp = cmp_sp[abs(cmp_sp.delay_centre_h)<=delays_difference_max_h*3/2]
        #keep sun to be used for delays
        columns_dt= ["dtfirst","dtlast","dtmax","sunrise","sunset","transit"]
        cmp_sp.drop(columns=[c for c in cmp_sp.columns.to_list() if (c not in columns_dt and "delay" not in c)], inplace=True)
    else:
        cmp_sp = None
    #remove hour extract done separately if needed
    """    if hour_extract:
        #DEV NOTE 14/5/21 lazy iteration
        for c in columns_dt:
            cmp_sp[c+'_h'] = add_hour(cmp_sp[c])
        for c in ["sunrise","sunset","transit"]:
            cmp_sp[c+'_h'] = add_hour(cmp_sp[c])"""
    return cmp_sp


def reset_cmp_utc_to_tz(cmp_tz:pd.DataFrame, tzi:str, tze:str, timeshift_h:int=0, columns_dt:List[str]=None, drop_hour=True,
             extract_hour=False, delays_recalculate=False):
    #application: take the cmp calculated based on utc assumption and reset to a different tz
    #18/5/21 recalculate still to be tested
    #copy to avoid strange behaviour
    df = cmp_tz.copy(deep=True)
    #DEV NOTE 13/5/21 verifed with equals it works only up to some rows of the entire df, TBC
    if drop_hour:
        #removing hour extractions to be recalculated later
        columns_h = [c for c in df.columns.to_list() if ("hour" in c) or ("_h" in c)]
        df.drop(columns=columns_h, inplace=True)
    #replacing results of groupby_with_utc analysis with utc with one using time zone
    #17/5/21 new to be tested
    if delays_recalculate:
        columns_dt:List[str]=["dtfirst","dtlast","dtmax"]
    df = dtop.reset_tzi_convert_tze(df, tzi=tzi, tze='utc', dti_columns=columns_dt)
    for c in columns_dt:
        df[c] = df[c].apply(lambda x: x+datetime.timedelta(seconds=int(60*60*timeshift_h)))
        if extract_hour:
            if "datetime" not in c:
                l = c+"_h"
            else:
                l = c.replace("datetime","hour")
            df[l] = dtop.add_hour(df[c])
        #17/5/21 new part to be tested
        if delays_recalculate:
            c = 0
            columns_delays_sp:Dict[str,str] = {
                "delay_first_h":"sunrise", "delay_last_h":"sunset", "delay_transit_h":"transit"}
            for k, v in columns_delays_sp.items():
                dt_c = columns_dt[c]
                df[k] = df.apply(lambda x: (x[dt_c]-x[v]).total_seconds()/3600, axis=1)
                c+=1
    return df


def get_sunpath_delays(cmp_sp_tz:pd.DataFrame, shifts:List[int], tzi:List[str], sl:csky.SolarLibrary, sunpath_filter=False,
                       dates=False):
    #reset based on cmp_sp_tz initially calculated assuming sl.tz as tz
    columns = ['tz','time','shift_h','delay_first_h_mdn','delay_first_h_min','delay_last_h_mdn',
    'delay_last_h_max', 'delay_transit_h_mdn', 'delay_product_mdn','len','d_product_min_date', 'd_product_min_value']
    if dates:
        columns.append("dates")
    delays = pd.DataFrame(columns=columns)
    for tz in tzi:
        for s in shifts:
            cmp_t = cmp_sp_tz.copy(deep=True)
            columns_dt = ["dtfirst","dtlast","dtmax"]
            #reset need tz compare
            cmp_t = reset_cmp_utc_to_tz(cmp_tz=cmp_t, tzi=tz, tze=sl.timezone, timeshift_h=s, columns_dt=columns_dt,
                          drop_hour=True, extract_hour=False)
            #recalculating delays
            cmp_t.loc[:,"delay_first_h"] = cmp_t.apply(lambda x: (x["dtfirst"]-x["sunrise"]).total_seconds()/3600, axis=1)
            cmp_t.loc[:,"delay_last_h"] = cmp_t.apply(lambda x: (x["dtlast"]-x["sunset"]).total_seconds()/3600, axis=1)
            cmp_t.loc[:,"delay_transit_h"] = cmp_t.apply(lambda x: (x["dtmax"]-x["transit"]).total_seconds()/3600, axis=1)
            cmp_s, cmp_w = dtop.df_split_dst(cmp_t, sl_tz=sl.timezone, tzi=sl.timezone)
            for time, cmp_time in zip(["summer", "winter"], [cmp_s, cmp_w]):
                if len(cmp_time)>0:
                    if sunpath_filter:
                        #following filter not used since still negative values
                        #if min(cmp_time.delay_first_h) < 0 or max(cmp_time.delay_last_h) >0: cmp_time = pd.DataFrame()
                        cmp_time = cmp_time.copy().loc[(cmp_time.delay_first_h>0) & (cmp_time.delay_last_h<0),:]
                    if len(cmp_time)>0:
                        cmp_time["delay_product"] =cmp_time.apply(lambda x:x["delay_first_h"]*
                                                                      x["delay_last_h"]*x["delay_transit_h"], axis=1)
                        sp =  cmp_time.loc[cmp_time.delay_product>=0,"delay_product"]
                        sn =  cmp_time.loc[cmp_time.delay_product<=0,"delay_product"]
                        #DEV NOTE 5/5/21 rough solution for None
                        spm = abs(min(sp)) if len(sp) > 0 else 9999
                        snm = abs(max(sn)) if len(sn) > 0 else 9999
                        m, d = (spm, sp.idxmin()) if spm < snm else (-snm, sn.idxmax())
                        dsc = cmp_time.describe(percentiles=[0.5])
                        first, last, transit, prd = dsc.loc['50%',['delay_first_h','delay_last_h','delay_transit_h',
                                                                   'delay_product']]
                        first_min = dsc.loc['min','delay_first_h']
                        last_max = dsc.loc['max','delay_last_h']
                        values = [tz,time,s,first,first_min,last,last_max,transit,prd,len(cmp_time),d,m]
                        if dates:
                            values.append(cmp_time.index.to_list())
                        delays = delays.append(dict(zip(delays.columns.to_list(), values)),ignore_index=True)
    return delays


"""8/6/21 not used in preliminary analysis, to be checked"""

#8/6/21 not used in preliminary analysis, to be checked
def get_completeness_cs_dly(df_c:pd.DataFrame, sl:csky.SolarLibrary, tzi='utc', time_resolution_s_max=3600 ):
    #dti = reset_tzi_convert_tze(df_c.dt)
    #dummy dti but could be utc also to keep provided data as they are, timezone is enabled
    df_c.index = pd.DatetimeIndex(df_c.dt)
    dly = dtop.datacompleteness(df_c.gpoa, time_resolution_s_max=time_resolution_s_max, get_hour=False)
    #transform min and max into timezone for comparison
    columns_dt = ["first_datetime", "last_datetime", "max_datetime"]
    #CONVERTING (NOT reset) into tzi
    for c in columns_dt: dly[c] = pd.DatetimeIndex(dly[c]).tz_convert(tz=sl.timezone)
    # dly =  dtop.columns_to_tz(dly, tzi=tzi, tze=sl.timezone, columns=columns_dt, index=False)
    # reset_tzi_convert_tze(dtis, tzi:str=None, tze:str=None, dti_columns:List[str]=None)
    for c in columns_dt: dly[c.replace("datetime","hour")]=dtop.add_hour(dly[c])
    #transform dti into timezone after since dly grouped with less data (i.e. less data losses)
    #dti = dtop.reset_tzi_convert_tze(dly.index, tzi=tzi, tze=sl.timezone)
    dti = dly.index.tz_convert(tz=sl.timezone)
    dly.index = dti
    cs_dly = csky.get_sun_rise_set_transit(dti, sl)
    dly = dly.merge(cs_dly.loc[:, ['sunrise_h','sunset_h', 'transit_h']], how='inner', left_index=True, right_index=True)
    return dly


#8/6/21 not used in preliminary analysis, to be checked
@decorators.timer
#def convert_completeness_cs_tz(dly:pd.DataFrame, tzi:str, tze:str): #waiting for completeness TBU
def convert_completeness_to_tz(dly:pd.DataFrame, tzi:str, tze:str='utc'):
    #DEV NOTE 17/5/21 could be transferred as generic ones to data_operations module
    #DEV NOTE 13/5/21 verifed with equals it works only up to some rows of the entire df, TBC
    columns_h = ["first_hour","last_hour","max_hour"]
    #drop columns calculated with previous tzi
    dly.drop(columns=columns_h, inplace=True)
    columns_dt = ["first_datetime","last_datetime","max_datetime"]
    #replacing results of groupby_with_utc analysis with utc with one using time zone
    for c in columns_dt:
        dti = pd.DatetimeIndex(dly[c])
        #utc transformed to time zone, inversing the process
        #print(dti[0]) if c == "first_datetime" else None
        dti = dtop.reset_tzi_convert_tze(dti, tzi=tzi, tze=tze)
        #print(dti[0]) if c == "first_datetime" else None
        #dti = dtop.reset_tzi_convert_tze(dti, tzi=None, tze=tze)
        #print(dti[0]) if c == "first_datetime" else None
        dly[c]= dti
    h = dict(zip(columns_dt, columns_h))
    for k, v in h.items():
        dly[v] = dtop.add_hour(dly[k])
    return dly

#8/6/21 not used in preliminary analysis, to be checked
def get_daylight_disconnection_dates(dly:pd.DataFrame) -> pd.Series:
    #necessary since a clear days could be a disconnection and affect statistical evaluation
    #fences not necessary since one hour usually
    #dly_dsc = describe_with_tukey_fence(df_dly, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    #hours_limit = dly_dsc.loc['fence_tukey_max','before_first_hours']
    dly_dsc = dly[(dly.first_hour > dly.sunrise_h) | (dly.sunset_h > dly.last_hour)]
    return dly_dsc.index