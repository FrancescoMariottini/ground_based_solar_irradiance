import pandas as pd
import os

CREST_IRRADIANCE_FOLDER = r"C:\Users\wsfm\OneDrive - Loughborough University\_Personal_Backup\python_repositories\ground-based-solar-irradiance\assets\isc_irradiance_files"
CREST_IRRADIANCE_FILE = r"chpoa_y15.csv"
CREST_IRR_FILEPATH = os.path.join(CREST_IRRADIANCE_FOLDER, CREST_IRRADIANCE_FILE)
# 7/8/23 hourly spot value created through:
"""select date_trunc('hour',w.tmstamp) as date_trunc,
(w.g_cmp11_ppuk) as g_cmp11_ppuk,
w.tmstamp as tmstamp
FROM w_meas.mms1_cr1000_tom as w
where w.tmstamp > '2015-01-01 00:00:00.000' and w.tmstamp < '2015-01-01 23:00:00.000' 
and w.tmstamp = date_trunc('hour',w.tmstamp)"""
# should be rather average, it won't affect a lot directional response if represented as absolute

# 26/3/23 power output in W
PVSYST_HOURLY_FILE = r"Project_test_221012_Project_VC4_HourlyRes_250_1m.CSV"

# PVsyst project exported meteo file
PVSYST_HOURLY_METEO = r"Loughborough_meteo.csv"
PS_WEATHER_FILEPATH = os.path.join(os.getcwd().replace("examples","data"),PVSYST_HOURLY_METEO)
PS_HOURLY_FOLDERPATH = r"C:\Users\wsfm\PVsyst7.0_Data\UserHourly"

PS_HOURLY_FILEPATHS = [os.path.join(PS_HOURLY_FOLDERPATH,PVSYST_HOURLY_FILE)]

CREST_FILENAME_HOUR = r"_select_m1_hour_trunc_m1_g_cmp11_ppuk_avg_m1_t_ambient_avg_m1_wi_202308081755.csv"

DATA_FOLDERPATH = r"C:\Users\wsfm\OneDrive - Loughborough University\_Personal_Backup\pv_module_model\data"
IV_FILENAME = r"_select_m1_hour_trunc_m1_g_cmp11_ppuk_avg_m2_chp_02_avg_iv_impp_2018_202308072113.csv"
WEATHER_FILENAME = r"_select_date_trunc_hour_m1_tmstamp_as_hour_trunc_avg_t_ambient_a_2018_202308081627.csv"
CREST_FILENAME_MINUTE = r"_select_m1_minute_trunc_m1_g_cmp11_ppuk_avg_m1_t_ambient_avg_m1_2018_202308182232.csv"

FILENAMES = [""]


def get_merge_data(folderpath:str=None, filenames:list= None, #[IV_FILENAME,WEATHER_FILENAME], 
                   datetime_column="minute_trunc",datetime_format=r"%Y-%m-%d %H:%M:%S")-> pd.DataFrame:
    if folderpath is None:
        folderpath = os.path.join(DATA_FOLDERPATH,datetime_column)
        print(folderpath)
        filenames = []
        for f in os.scandir(os.path.join(DATA_FOLDERPATH,datetime_column)):
            filenames.append(f.name)
                
    """if filenames is None:
        if datetime_column == "minute_trunc": filenames = [CREST_FILENAME_MINUTE]
        elif datetime_column == "hour_trunc": filenames = [CREST_FILENAME_HOUR"""

    # 20/8/23 to be updated with list of files
    # data extracted from dbeaver
    df = pd.DataFrame()
    for f in filenames:
        filepath = os.path.join(folderpath,f)
        if len(df) == 0:
            df = pd.read_csv(filepath_or_buffer=filepath,encoding="utf-8-sig",delimiter=",", encoding_errors="replace")
        else:
            df_tmp = pd.read_csv(filepath_or_buffer=filepath,encoding="utf-8-sig",delimiter=",", encoding_errors="replace")
            clmns = [datetime_column] + [c for c in df_tmp.columns.to_list() if c not in df.columns.to_list()]
            # outer using both
            df = df.merge(df_tmp[clmns], left_on=datetime_column, right_on=datetime_column, how="outer", suffixes=('_x', '_y'))
    df[datetime_column] = df[datetime_column].apply(lambda x: pd.to_datetime(x, format=datetime_format, utc=False))
    # assumed utc
    df.index = pd.DatetimeIndex(df[datetime_column], ambiguous='NaT', name="datetime")
    df.sort_values(by="datetime", inplace=True)
    return df




def get_ps_prms(folderpath, filename) -> dict:
    """
    wrap-up function for pvsystem moduel parameters
    :
    :
    :return: parameters
    
    Parameters not included in PAN files to be added manually:

    #The dark or diode reverse saturation current at reference conditions, in amperes.
    "I_o_ref" : 0.040 / (10 ** 9)

    #The energy bandgap at reference temperature in units of eV. 1.121 eV for crystalline silicon. EgRef must be >0.
    "EgRef" : 1.12, #1

    # Reference irradiance in W/m^2.
    "temp_ref" : 25
    """
    ps_prms = {}
    f = open(os.path.join(folderpath,filename))
    for r in f.readlines():
        s = r.split("=")
        if len(s) > 1:
            ps_prms[s[0].strip()] = s[1].rstrip()
    f.close()
    return ps_prms


def get_weather_df(crest_irr_filepath:str=CREST_IRR_FILEPATH, 
                   ps_weather_filepath:str=PS_WEATHER_FILEPATH, ps_hourly_filepaths:list=PS_HOURLY_FILEPATHS) -> pd.DataFrame:
    """
    wrap-up to create a weather file
    :param crest_filepath: CREST_IRRADIANCE_FOLDER,CREST_IRRADIANCE_FILE
    :param ps_weather_filepath: os.path.join(os.getcwd().replace("examples","data"),PVSYST_HOURLY_METEO)
    :param ps_hourly_filepath: PVSYST_FOLDER
    """

    #Data for pvsyst imported from:
    #C:\Users\wsfm\OneDrive - Loughborough University\_Personal_Backup\model_predict\ground-based-solar-irradiance\assets\isc_irradiance_files\*.*
    crest_irr = pd.read_csv(filepath_or_buffer=crest_irr_filepath, 
    encoding="utf-8-sig",delimiter=",", encoding_errors="replace") #, usecols=[0,1,2,3,4,5,6]) #nrows=20,

    # 22/12/22 UTC false initially to compare with PVsyst
    # using date since also in pvsyst
    crest_irr["date"] = crest_irr["date_trunc"].apply(lambda x: pd.to_datetime(x, format=r"%d/%m/%Y %H:%M:%S", utc=False))

    crest_irr["year"] = crest_irr["date"].dt.year

    # reading meteo data
    pvs_mt = pd.read_csv(filepath_or_buffer=ps_weather_filepath, 
    encoding="utf-8-sig",delimiter=";", encoding_errors="replace", usecols=[0,1,2,3,4,5,6]) #nrows=20,
    pvs_mt["Interval beginning"] = pvs_mt["Interval beginning"].apply(lambda x: pd.to_datetime(x, format=r"%d/%m/%y %H:%M", utc=False))
    pvs_mt.rename(columns={"Interval beginning":"date"}, inplace=True)
    # Direct normal irradiance (DNI or BeamNor)  are also available for concentrating systems.

    # reading output data
    # C:\Users\wsfm\PVsyst7.0_Data\Models

    pvss = {}
    pvs_slcs = {}
    # 20/3/23 for standard analysis only first file will be considered

    for f in ps_hourly_filepaths:
        # f = PVSYST_HOURLY_FILE
        pvs_out = pd.read_csv(filepath_or_buffer=f, encoding="utf-8-sig",
        skiprows=10,   delimiter=",", encoding_errors="replace") #nrows=20,
        pvs_out = pvs_out.iloc[1:]
        for c in pvs_out.columns.to_list():
            if c != "date": 
                pvs_out[c] = pvs_out[c].astype(float)
        # PVsyst use local time
        # https://forum.pvsyst.com/topic/175-time-zone-in-met-data-file-and-site-file/
        pvs_out["date"] = pvs_out["date"].apply(lambda x: pd.to_datetime(x, format=r"%d/%m/%y %H:%M", utc=False))

        #MERGING DATAFRAMES

        # merging to get solar elevation for extracting I0
        # using out as base since mt has only up to decimal
        pvs_mt_mrg_columns = ["date"]+[c for c in pvs_mt.columns.to_list() if c not in pvs_out.columns.to_list()]

        # different merging to check df well aligned
        # pvs_out_mrg_columns2 = [c for c in pvs_out.columns.to_list()]

        pvs = pvs_out.merge(pvs_mt.loc[:,pvs_mt_mrg_columns], left_on="date", right_on="date", how="left", suffixes=('_x', '_y'))

        # merging for comparison
        pvs = pvs.merge(crest_irr[["g_cmp11_ppuk","date"]], left_on="date", right_on="date")

        pvs.rename(columns={"date":"datetime"}, inplace=True)
        pvs["year"] = pvs["datetime"].dt.year
        pvs["date"] = pvs["datetime"].dt.date
        pvs["hour"] = pvs["datetime"].dt.hour
        # test day previously selected
        pvss[f] = pvs
        
        # test selection
        #pvs_slc = pvs.loc[(pvs["datetime"].dt.day==22) & (pvs["datetime"].dt.month==3),:].copy(deep=True)
        #pvs_slcs[f] = pvs_slc
        return pvs