# most popular PostgreSQL database adapter (package) to be installed: psycopg2
from sqlalchemy import engine as e
#import os
from typing import List
from typing import TextIO
from typing import Dict
from typing import Tuple
#importing for parallel queries
import time
from threading import Thread
from numpy import squeeze

from pandas_profiling import ProfileReport

import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = None


_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

_COLUMNS_INFORMATION_SCHEMA = ['table_schema', 'table_name', 'column_name', "data_type", "timestamp_column"]
_QUERY_EXAMPLES = {"sensors_metadata": r"""SELECT s.sensorname, s.comments as sensorcomment,
s.manufacturer, s.manufacturermodelnumber, s.manufacturerserialnumber, s.lastcalibrationdate,
t.systemname, t.nameoflocation, t.latitude, t.longitude, t.altitude,
t.systemazimuth, t.systeminclination,t.systemcomment
FROM w_meta.metsensors as s
JOIN (SELECT i.*, m.systemname, m.comments as systemcomment FROM
(SELECT i.metsensorid, l.nameoflocation, l.comments, l.systemazimuth,
 l.systeminclination, i.dateremoved, i.measurementsystemid,
 l.latitude, l.longitude, l.altitude
 FROM w_meta.metsensorinstallations as i
 JOIN w_meta.outdoor_location as l ON i.locationid = l.locationid
 WHERE i.dateremoved IS NULL) as i JOIN w_meta.measurementsystems as m
 ON i.measurementsystemid = m.measurementsystemid) as t ON
 s.sensorid = t.metsensorid""",
"columns_information": r"""SELECT i.table_schema, i.table_name, i.column_name, i.data_type, p.description 
FROM information_schema.columns as i LEFT OUTER JOIN pg_catalog.pg_description as p ON i.ordinal_position =p.objsubid"""}


#"columns_information": r"""SELECT table_schema, table_name, column_name, data_type
#FROM information_schema.columns"""} #15/11/20 check if still used


class DatabaseEngine:
    def __init__(self, connection_type: str, user: str, password: str,
                 host: str, database: str) -> None:
        """
        Initialise database engine
        :param connection_type:
        :param user:
        :param password:
        :param host:
        :param database:
        """
        # if connection_type == "postgresql":
        # connection info at https://www.postgresql.org/docs/current/static/libpq-connect.html#LIBPQ-CONNSTRING
        _ = f"{connection_type}://{user}:{password}@{host}/{database}"
        self.connection = e.create_engine(_).connect()
        # storing empty columns, see method get_columns_description
        self.columns_dropped = pd.DataFrame(columns=["schemas", "table", "column", "cause"])

    def query_to_dataframe(self, query: str, close_connection_after: bool = False) -> pd.DataFrame:
        """
        Obtaining query results as dataframe. Results depend on granted select permissions.
        :param query:
        :param close_connection_after:
        :return:
        """
        print(f"Query started at {time.asctime(time.localtime())}")
        dataframe = pd.read_sql(query, self.connection)
        print(f"Query ended at {time.asctime(time.localtime())}")
        if close_connection_after:
            self.connection.close()
        return dataframe

    def get_isc_with_poa_irradiance(self,
                                channels: List[int] = _CHANNELS,
                                 test: bool = True) -> pd.DataFrame: #long query test provided
        #code to be improved to add dates
        #i.tmstamp >= '01/01/2019  00:00:00'
        #i.tmstamp < '01/01/2020  00:00:00'


        #TECHNICAL SOURCES
        #https://stackoverflow.com/questions/16469424/how-can-i-select-one-row-of-data-per-hour-from-a-table-of-time-stamps
        #https://stackoverflow.com/questions/8779918/postgres-multiple-joins
        channels = [str(c) for c in channels]
        query = "SELECT i.g_cmp11_ppuk, i.tmstamp, i.sensol_ppuk, pt1000_ppuk_sensol,"
        query += ",".join([f"ch{c}.isc as isc{c}, "
                           f"ch{c}.moduletemperature1 as mt1_{c}, "
                           f"ch{c}.moduletemperature1 as mt2_{c}, "
                           f"ch{c}.iv_sweep_endtime as set{c}" for c in channels])
        query += " FROM w_meas.mms1_cr1000_tom as i "
        join_part = ("LEFT JOIN (SELECT DISTINCT ON(date_trunc('hour', pv.iv_sweep_endtime)) pv.isc,"+
                        "pv.moduletemperature1, pv.moduletemperature2,"+
                        "pv.iv_sweep_endtime, pv.channelnumber, date_trunc('hour', pv.iv_sweep_endtime) as date_trunc "+
                        "FROM w_meas.pv_ivt_tom AS pv WHERE pv.channelnumber=")
        order_part = r" ORDER BY date_trunc('hour', pv.iv_sweep_endtime), pv.iv_sweep_endtime) as ch"
        query += " ".join([join_part+c+order_part+c+f" ON i.tmstamp=ch{c}.date_trunc" for c in channels])
        query += " WHERE EXTRACT(minute FROM i.tmstamp)=0 AND EXTRACT(second FROM i.tmstamp) = 0 "
        if test:
            #query += "LIMIT 10"
            query += "AND i.tmstamp >= '01/01/2019  00:00:00' AND i.tmstamp < '01/01/2020  00:00:00'"
        return self.query_to_dataframe(query=query)

    def get_sensors_info(self, sensor_types: List[str] = ['%hermopile%'],
                         parameters_included: List[str] = ['%PoA%'],
                         parameters_excluded: List[str] = ['%racker%']) -> pd.DataFrame:
        """
        Extracting sensors information from metsensors, metsensorinstallations, outdoor_location & measurementsystems
        :param sensor_types: identifiers for sensor types
        :param parameters_included: identifiers for parameters to be monitored
        :param parameters_excluded: identifiers for parameters to be EXCLUDED
        :return: table of sensor information
        """
        query = _QUERY_EXAMPLES["sensors_metadata"] + " WHERE ("
        query += " AND ".join([f"(s.sensortype LIKE '{st}')" for st in sensor_types])
        query += " ".join([f"AND (s.measuredparameter LIKE '{pi}')" for pi in parameters_included])
        query += " ".join([f"AND (s.measuredparameter NOT LIKE '{pe}')" for pe in parameters_excluded])
        query += ")"
        return self.query_to_dataframe(query=query)

    def get_table_information(self, table_name: str = 'pv_ivt_tom',
                                     table_schema: str = "w_meas") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Return information_schema for the table.
        :param table_name:
        :param table_schema:
        :return: Different values in a Dataframe, common values in a dictionary.
        """
        query = (f"SELECT i.*, p.description FROM information_schema.columns as i LEFT OUTER JOIN ("+
                "SELECT p.objsubid, p.description FROM pg_catalog.pg_description as p WHERE p.objoid ="+
                f"(SELECT oid FROM pg_class WHERE relname = '{table_name}')) as p ON i.ordinal_position ="+
                f"p.objsubid WHERE(i.table_name = '{table_name}' AND i.table_schema = '{table_schema}')")

        df = self.query_to_dataframe(query)
        df.dropna(axis=1, how='all', inplace=True)
        info_all_columns = {}
        for c in df.columns:
            _ = df.loc[:, c].unique()
            if len(_) == 1:
                info_all_columns[c] = _[0]
        df = df.drop(columns=info_all_columns.keys())
        return df, info_all_columns

    def get_columns_information(self, column_numbers_included: List[str] = ["22", "11"],
                     column_strings_included: List[str] = ["cm"],
                     table_strings_excluded: List[str] = ["ourly", "view", "ashu", "day"],
                     schemas_strings_included: List[str] = ["meas"]) -> pd.DataFrame:
        """
        Extracting columns_information from Postgresql information_schema
        :param column_numbers_included: possible numeric identifiers for sensors
        :param column_strings_included: all requested word identifiers for sensors
        :param table_strings_excluded: all excluded tables from the searching
        :param schemas_strings_included: possible schemas for the searching
        :return: tables location information
        """

        query = _QUERY_EXAMPLES["columns_information"] + " WHERE ("
        _ = []
        query += " AND ".join([" OR ".join([f"(column_name LIKE '%{n}%')" for n in column_numbers_included]),
                               " AND ".join([f"(column_name LIKE '%{c}%')" for c in column_strings_included]),
                               " AND ".join([f"(table_name NOT LIKE '%{t}%')" for t in table_strings_excluded]),
                               " OR ".join([f"(table_schema LIKE '%{s}%')" for s in schemas_strings_included])])
        query += ")"

        columns_information = self.query_to_dataframe(query=query)
        #removing double results due to pg_catalog.pg_description
        columns_information.drop_duplicates(inplace=True)
        #deep copy to avoid indexing warnings
        clm_inf = columns_information.copy()
        clm_inf.loc[:, "timestamp_column"] = None

        for t in clm_inf["table_name"].unique():
            schema = clm_inf.loc[clm_inf["table_name"] == t, "table_schema"].unique()[0]
            query = (f"SELECT t.column_name as timestamp_column FROM information_schema.columns AS t " +
                     f"WHERE (t.data_type = 'timestamp without time zone' OR t.data_type = 'timestamp with time zone')" +
                     f" AND t.table_schema = '{schema}' AND t.table_name = '{t}'  LIMIT 1")
            _ = self.query_to_dataframe(query)
            if not _.empty:
                timestamp_column = _.loc[:, "timestamp_column"].values[0]
                clm_inf.loc[clm_inf["table_name"] == t, "timestamp_column"] = timestamp_column

        return clm_inf

    def store_dropped_column(self, schemas: str, table: str, column: str, cause: str) -> pd.DataFrame:
        """
        Store dropped columns
        :param schemas:
        :param table:
        :param column:
        :param cause: message to be registred inside the class
        :return:
        """
        self.columns_dropped = self.columns_dropped.append({"schemas": schemas, "table": table, "column": column,
                                                          "cause": cause}, ignore_index=True)
        self.columns_dropped.drop_duplicates(inplace=True)
        return self.columns_dropped

    def get_columns_description(self, columns_information: pd.DataFrame = pd.DataFrame(),
                                column_numbers_included: List[str] = ["22"],
                                column_strings_included: List[str] = ["cm"],
                                table_strings_excluded: List[str] = ["ourly", "view", "ashu", "day"],
                                schemas_strings_included: List[str] = ["meas"],
                                aggregate_functions: List[str] = ["count", "min", "max", "sum", "avg"],
                                text_stream: TextIO = open("../examples/isc_columns_description.csv", 'a')
                                ) -> pd.DataFrame:
        """
        Preliminary evaluation of values for each find column
        :param columns_information: columns location information (obtainable through find_columns function)
        :param column_numbers_included: possible numeric identifiers for sensors
        :param column_strings_included: all requested word identifiers for sensors
        :param table_strings_excluded: all excluded tables from the searching
        :param schemas_strings_included: possible schemas for the searching
        :param output_csv_path: path for file creation
        :return: columns description including aggregators and first/last non-null value
        """
        if len(columns_information) > 0 and all([c in columns_information.columns for c in _COLUMNS_INFORMATION_SCHEMA]):
            clm_inf = columns_information[_COLUMNS_INFORMATION_SCHEMA]
        else:
            clm_inf = self.get_columns_information(column_numbers_included, column_strings_included,
                                        table_strings_excluded, schemas_strings_included)
        # copy to avoid slicing warning later. See "Evaluation order matters" at:
        # https://pandas.pydata.org/pandas-docs/version/0.22.0/indexing.html#indexing-view-versus-copy
        clm_inf.loc[:, ["timestamp_min", "timestamp_max"]] = None
        clm_inf.loc[:, aggregate_functions] = None
        text_stream.write(",".join([v for v in clm_inf.columns.values])+"\n")
        threads = []
        columns_description = pd.DataFrame(columns=clm_inf.columns)
        for i in clm_inf.index.values:
            schema, table, column, timestamp_column = clm_inf.loc[
                i, _COLUMNS_INFORMATION_SCHEMA]
            if timestamp_column is None:
                self.store_dropped_column(schema, table, column, "no timestamp column")
            elif timestamp_column is not None:
                threads.append(ColumnDescriptionWriter(engine = engine,
                               columns_information = dict(zip(_COLUMNS_INFORMATION_SCHEMA,[schema, table, column, timestamp_column])),
                               text_stream= text_stream))
                columns_description.append(threads[i].start(), ignore_index= True)
        for i in clm_inf.index.values:
            threads[i].join()
        columns_description.dropna(subset=aggregate_functions, how='all', inplace=True)
        return columns_description

class QueryResultsWriter(Thread): #27/11/20 not used to be removed ?
    def __init__(self, query_str: str, query_engine: DatabaseEngine, thread_identifier: str, csv_output_path: str):
        Thread.__init__(self)
        self.query = query_str
        self.engine = query_engine
        self.csv_output_path = csv_output_path
        self.thread_identifier = thread_identifier
    def run(self):
        print(f"{self.thread_identifier} started at {time.asctime(time.localtime())}")

class ColumnDescriptionWriter(Thread):
    def __init__(self, engine: DatabaseEngine, columns_information: dict,
                 aggregate_functions: List[str] = ["count", "min", "max", "sum", "avg"],
                 text_stream: TextIO = open("../examples/isc_columns_description.csv", 'a')) -> dict:
        Thread.__init__(self)
        self.engine = engine
        self.cd = columns_information
        self.aggregate_functions = aggregate_functions
        self.text_stream = text_stream

    def run(self) -> dict:
        print(f"{' '.join([str(v) for v in self.cd.values()])} started on {time.asctime(time.localtime())} \n")
        query = f"SELECT t.{self.cd['timestamp_column']} FROM {self.cd['table_schema']}.{self.cd['table_name']} AS t " \
                f"WHERE t.{self.cd['column_name']} IS NOT NULL ORDER BY t.{self.cd['timestamp_column']} LIMIT 1"
        _ = self.engine.query_to_dataframe(query)
        if _.empty:
            self.cd.update(zip(["timestamp_min", "timestamp_max"],[None]*2))
            self.cd.update(zip(self.aggregate_functions, [None] * len(self.aggregate_functions)))
        elif not _.empty:
            self.cd["timestamp_min"] = _.loc[:,self.cd['timestamp_column']].values[0]
            query = f"SELECT t.{self.cd['timestamp_column']} FROM {self.cd['table_schema']}.{self.cd['table_name']} AS t " \
                    f"WHERE t.{self.cd['column_name']} IS NOT NULL ORDER BY t.{self.cd['timestamp_column']} DESC LIMIT 1"
            self.cd["timestamp_max"] = self.engine.query_to_dataframe(query).loc[:,self.cd['timestamp_column']].values[0]
            query = "SELECT " + ", ".join([f"{f}(t.{self.cd['column_name']})" for f in self.aggregate_functions]) + \
                    f" FROM {self.cd['table_schema']}.{self.cd['table_name']} AS t"
            self.cd.update(zip(self.aggregate_functions,
                               squeeze(self.engine.query_to_dataframe(query).loc[:, self.aggregate_functions].values)))
        self.text_stream.write(",".join([str(v) for v in self.cd.values()])+"\n")
        print(f"{' '.join([str(v) for v in self.cd.values()])} written on {time.asctime(time.localtime())} \n")
        return self.cd


class ModuleIscWriter(Thread):
    def __init__(self, engine: DatabaseEngine,
                 columns_information: dict,
                 aggregate_functions: List[str] = ["count", "min", "max", "sum", "avg"],
                 text_stream: TextIO = open("../examples/isc_columns_description.csv", 'a')) -> dict:
        Thread.__init__(self)
        self.engine = engine
        self.cd = columns_information
        self.aggregate_functions = aggregate_functions
        self.text_stream = text_stream

    def run(self) -> dict:
        print(f"{' '.join([str(v) for v in self.cd.values()])} started on {time.asctime(time.localtime())} \n")
        query = f"SELECT t.{self.cd['timestamp_column']} FROM {self.cd['table_schema']}.{self.cd['table_name']} AS t " \
                f"WHERE t.{self.cd['column_name']} IS NOT NULL ORDER BY t.{self.cd['timestamp_column']} LIMIT 1"
        _ = self.engine.query_to_dataframe(query)
        if _.empty:
            self.cd.update(zip(["timestamp_min", "timestamp_max"],[None]*2))
            self.cd.update(zip(self.aggregate_functions, [None] * len(self.aggregate_functions)))
        elif not _.empty:
            self.cd["timestamp_min"] = _.loc[:,self.cd['timestamp_column']].values[0]
            query = f"SELECT t.{self.cd['timestamp_column']} FROM {self.cd['table_schema']}.{self.cd['table_name']} AS t " \
                    f"WHERE t.{self.cd['column_name']} IS NOT NULL ORDER BY t.{self.cd['timestamp_column']} DESC LIMIT 1"
            self.cd["timestamp_max"] = self.engine.query_to_dataframe(query).loc[:,self.cd['timestamp_column']].values[0]
            query = "SELECT " + ", ".join([f"{f}(t.{self.cd['column_name']})" for f in self.aggregate_functions]) + \
                    f" FROM {self.cd['table_schema']}.{self.cd['table_name']} AS t"
            self.cd.update(zip(self.aggregate_functions,
                               squeeze(self.engine.query_to_dataframe(query).loc[:, self.aggregate_functions].values)))
        self.text_stream.write(",".join([str(v) for v in self.cd.values()])+"\n")
        print(f"{' '.join([str(v) for v in self.cd.values()])} written on {time.asctime(time.localtime())} \n")
        return self.cd


# TESTING AREA
"""
import os
import pandas as pd

ISC_IRRADIANCE_FILES_FOLDER = os.path.join(os.path.dirname(os.getcwd())+r"/assets/isc_irradiance_files/")
filenames = [f for f in os.listdir(ISC_IRRADIANCE_FILES_FOLDER) if f.endswith(r".csv") and f.startswith(r"ch")]
df = pd.DataFrame()


for filename in filenames:
    errors = []
    try:
        dt_f = '%d-%m-%y %H:%M:%S'
        converters = {'tmstamp': lambda x: pd.to_datetime(x, format=dt_f, errors='raise')}
        df_tmp = pd.read_csv(os.path.join(ISC_IRRADIANCE_FILES_FOLDER, filename),
                             delimiter= ",", converters= converters)
    except ValueError:
        errors = [dt_f]
        try:
            dt_f = '%d-%m-%y %H:%M'
            converters = {'tmstamp': lambda x: pd.to_datetime(x, format=dt_f, errors='raise')}
            df_tmp = pd.read_csv(os.path.join(ISC_IRRADIANCE_FILES_FOLDER, filename),
                                 delimiter=",", converters=converters)
        except ValueError:
            errors += [dt_f]
            print(f'{filename} not valid for formats: {[f for f in errors]}')
    if len(errors) != 2:
        if len(df) > 0:
            columns_excluded = ['g_cmp11_ppuk', 'sensol_ppuk', 'pt1000_ppuk_sensol']
            df_tmp.drop(columns=columns_excluded, inplace=True)
            print(f'{filename} adding columns: {",".join([c for c in df_tmp.columns])}')
            try:
                df = df.merge(df_tmp, left_on='tmstamp', right_on='tmstamp')
            except KeyError:
                print(f'{filename} could not be merged')
        elif len(df) == 0:
            df = df_tmp.copy(deep=True)

for c in df.columns.tolist():
    if c.startswith("mt") or c!= 'tmstamp' or c!= 'pt1000_ppuk_sensol':
        df.drop(columns=c, inplace=True)

profile = ProfileReport(df, title='Pandas Profiling Report')
"""

"""
engine = DatabaseEngine()

engine.get_isc_with_poa_irradiance()

engine.connection.close()


#" OR ".join([f"l.nameoflocation = '{l}'" for l in locations])



ARCHIVED
#first column not always datetime
query = (_QUERY_EXAMPLES["columns_information"] + " WHERE ordinal_position = 1 AND " +
f"table_schema = {schema} AND table_name = {table} FROM information_schema.columns")
column_first = self.query_to_dataframe(query).column_name


SELECT t.*, l.systemazimuth, l.systeminclination FROM
(SELECT d.deviceid, d.devicename, count(p.deviceid)as records,
min(p.measurementdatetime) as start, max(p.measurementdatetime) as end,
d.outdoorlocation FROM w_meta.devices as d LEFT OUTER JOIN w_meas.mppt as p
ON d.deviceid = p.deviceid GROUP BY d.deviceid ORDER BY records DESC) as t
LEFT OUTER JOIN w_meta.outdoor_location as l ON t.outdoorlocation = l.locationid
WHERE t.records > 0
"""
