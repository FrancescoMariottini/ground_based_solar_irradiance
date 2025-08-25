# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:50:35 2018

@author: wsfm
"""

#DEV NOTE 25/10/18: sql connection must be closed !! 
# Last time modified 3/11/2020

class sql:
    #solar part to be removed 16/5/18
    #TO BE IMPROVED, SPLITTING SQL AND SQLALCHEMY
    #beware of modules using it (e.g. in data quality)
    """functions exclusively related to sql operation"""
    #Created 27/07/17 
    #Last updated 27/07/17
    
    _pyhton_dtypes=('datetime64[ns]',
    'float64')
    _sqla_dtypes=('DateTime(timezone=False)',
    'float()') 
    dtypes_dict = dict(zip(_pyhton_dtypes,_sqla_dtypes))
    import sqlalchemy as sqla
    meteo_tab_dtype={'measys_id':sqla.types.INTEGER,
    'datetime_start':sqla.types.DateTime,
    'datetime_end':sqla.types.DateTime,
    'global_swdn':sqla.types.FLOAT(2,10),
    'diffuse_sw':sqla.types.FLOAT(2,10),
    'direct_normal_sw':sqla.types.FLOAT(2,10),
    'sw_up':sqla.types.FLOAT(2,10),
    'lw_dn':sqla.types.FLOAT(2,10),
    'lw_up':sqla.types.FLOAT(2,10),
    't_a':sqla.types.FLOAT(2,10),
    'measys_id_ck':sqla.types.INTEGER,
    'datetime_start_ck':sqla.types.INTEGER,
    'datetime_end_ck':sqla.types.INTEGER,
    'global_swdn_ck':sqla.types.INTEGER,
    'diffuse_sw_ck':sqla.types.INTEGER,
    'direct_normal_sw_ck':sqla.types.INTEGER,
    'sw_up_ck':sqla.types.INTEGER,
    'lw_dn_ck':sqla.types.INTEGER,
    'lw_up_ck':sqla.types.INTEGER,
    't_a_ck':sqla.types.INTEGER}
    
    def connstring(conntype,user,pw,host,db):
        if conntype == "postgresql":
            #https://www.postgresql.org/docs/current/static/libpq-connect.html#LIBPQ-CONNSTRING
            connstring=conntype+"://"+user+":"+pw+"@"+host+"/"+db
        return connstring
    
    class alchemy:
        def engine(connstring):
            from sqlalchemy import engine as e
            """set up and return an engine to connect to a database"""
            #Created 27/07/17 
            #connstring example: postgresql+psycopg2://postgres@localhost/[name_of_database]
            sqlengine = e.create_engine(connstring)
            return sqlengine
        def table_valid (connstring,schemaname,tablename):
            """check and return the status of a table as string """
            #Created 27/07/17 
            #Improved 22/11/17: query results
            # IMPROVE: better management of exception       
            if sql.alchemy.engine(connstring).has_table(tablename,schemaname) == True:        
                sqlquery = r"SELECT infoc.column_name, infoc.table_name FROM information_schema.columns AS infoc WHERE infoc.table_name ='"+tablename+"' AND infoc.table_schema ='"+schemaname+"'"        
                if sql.alchemy.query(sqlquery,connstring) == []:            
                    connectioncheck  = "Table exists but no columns defined."
                    queryresult = []
                else:
                    connectioncheck = "Table exists and columns are identified"  
                    queryresult = sql.alchemy.query(sqlquery,connstring)
            else:
                connectioncheck = "Table does not exist or is not visible to a connection."     
            print(connectioncheck)
            return queryresult
        def query(querystring,connstring):
            """launch a sql query and return the result""" 
            #Created 27/07/17 
            #DEV NOTE 10/10/18: beware of redundancy, there is already pd.read_sql
            conn = sql.alchemy.engine(connstring).connect()
            result = conn.execute(querystring).fetchall()
            conn.close()
            return result 
    
        def table_overview(sqlengine,schemaname,tablename):
            """ return an overview of the table object (from sql as as string"""
            #Created 27/07/17 
            # sqlengine is a database connection
            # IMROVE: merge together the different information in single table 
            from sqlalchemy import Table, MetaData
            try:
                table = Table(tablename,MetaData(),autoload=True,autoload_with=sqlengine,schema=schemaname)                  
                tableoverview = "TABLE OVERVIEW"
                tableoverview = tableoverview +2*"\n"+"COLUMNS NAMES"+"\n"+str(table.columns.keys())
                tableoverview = tableoverview +2*"\n"+"DETAILS"+"\n"+repr(table)
            except:
                tableoverview = "An error has occured"
            return tableoverview
        def select_on_list(parameter,table,schema,variable,lista,connection):
            """ Select parameter list on schema.table where variable is in list"""
            from sqlalchemy import engine as e
            import numpy as np 
            _engine_connection = e.create_engine(connection)
            select_query = "SELECT tab."+parameter+" FROM "+str(schema)+"."+table+" AS tab WHERE "
            started = False
            for value in lista:
                if started == True: 
                    select_query = select_query + " OR "
                select_query = select_query + "tab."+variable+"='" + str(value)+"'"
                started = True
            #launch select query
            connection = _engine_connection.connect()
            # transform into array to give only results 
            results = np.asarray(connection.execute(select_query).fetchall())
            connection.close()
            return results

        def dataframe_to_sql(dataframe,connection,schema,table,id_header,sqa_dtypes):
            """transer a dataframe into SQL avoiding copies"""
            #10/10/18: based on extended file created in coenergy module 
            #sqlalchemy setup
            from sqlalchemy import engine as e
            #import numpy as np
            _engine_connection = e.create_engine(connection)
            
            """
            #check all values already copied             #base query
            #look_query = ( "SELECT "+str(table)+"."+str(id_header)+
            look_query = ( "SELECT "+r'tabella."'+id_header+
            r'" FROM '+schema+"."+table+" AS tabella WHERE ")
            started = False             #adding condition for all     
            for value in list(dataframe.loc[:,id_header].values):
                if started == True:
                    look_query =  look_query + " OR "
            #DEV NOTE 19/10/18: beware on use of "column" and 'value' in Postgresql
                look_query =     look_query + r'tabella."'+id_header+r'"='+"'"+str(value)+"'"
                started = True
            """
                
            _connection = _engine_connection.connect()  
            #_connection.close()   
            
    
            
            #_id_copied = np.asarray(_connection.execute(look_query).fetchall())
            #_connection.close()    
            #DEV NOTE 10/10/18: selection part to be developed
            """
            print(_id_copied)
            #selection of dataframe without already copied values
            df = dataframe[dataframe[:,id_header].isin(_id_copied)]
            #copy the entire dataframe
            """
            #try
            
            
            dataframe.to_sql(name=table,
                              con=_engine_connection,
                              schema=schema,
                              if_exists="append",
                              index=False,
                              index_label=None,
                              chunksize=None,
                              method="multi")
            #except: 
            #    print("err")                
            _connection.close()   
            
   
#TEST 3/11/2020
connstring = sql.connstring()
print(connstring)          
                
sqlengine = sql.alchemy.engine(connstring)


query = """ SELECT
            s.sensorid,
            s.sensorname,
            s.sensortype,
            s.measuredparameter,
            s.rawmeasurementunits,
            s.calibrated_measurement_units,
            s.conversionformula,
            s.comments,
            t.*
            FROM
            w_meta.metsensors as s
            JOIN
            (SELECT
             i.locationid,
             i.metsensorid,
             l.nameoflocation,
             l.comments,
             l.systemazimuth,
             l.systeminclination,
             i.dateremoved
             FROM
            w_meta.metsensorinstallations as i
             JOIN
            w_meta.outdoor_location as l
             ON
             i.locationid = l.locationid
              WHERE i.locationid='19')
             as t
             ON
             s.sensorid = t.metsensorid"""



connection = sqlengine.connect()
results = connection.execute(query).fetchall()
