import logging
import gc
import datetime
from sqlalchemy import create_engine
import time
import pandas as pd
import psycopg2 as pg
import sys
sys.path.append('../src')
import credentials
reload(credentials)

datalake_logger = logging.getLogger('model.datalake')

class PostgreSQLBase():

    connection = None

    def __init__(self):
        self.cursor = None
        self.SQL_QUERIES_PATH = '../sql_queries/'

    @classmethod
    def _db_connect(cls):
        """ Set connection to the DataHub """
        try:
            # connect to the PostgreSQL server
            cls.connection = pg.connect(host=credentials.HOST_NAME, dbname=credentials.DB_NAME,
                                    user=credentials.USER_NAME, password=credentials.PASS)
            # datalake_logger.debug('Database connection opened')
        except (Exception, pg.DatabaseError) as error:
            datalake_logger.error(error)
            cls._db_disconnect()
        # return cls.connection

    @classmethod
    def _db_disconnect(cls):
        """ Close connection to the DataHub """
        if cls.connection is not None:
            cls.connection.close()
            # datalake_logger.debug('Database connection closed')

    def _db_reconnect(self):
        pass

    def test_connection(self):
        """ Test connection to the DataHub """
        try:
            PostgreSQLBase._db_connect()
            self.cursor = PostgreSQLBase.connection.cursor()
            self.cursor.execute('SELECT version()')
            # display the PostgreSQL database server version
            db_version = self.cursor.fetchone()
            self.cursor.close()
            datalake_logger.debug('PostgreSQL database version: {}'.format(db_version))
            # close the communication with the PostgreSQL
        except (Exception, pg.DatabaseError) as error:
            datalake_logger.error(error)
        finally:
            PostgreSQLBase._db_disconnect()

    def get_data(self, query_file, param=None, param_type='none'):
        """ Execute sql query """
        df = None
        with open(self.SQL_QUERIES_PATH + query_file, 'r') as f:
            query = f.read()

        try:
            PostgreSQLBase._db_connect()
            self.cursor = PostgreSQLBase.connection.cursor()

            if param_type == 'tuple':
                self.cursor.execute(query.format(*param))
            elif param_type == 'none':
                self.cursor.execute(query)
            else:
                self.cursor.execute(query.format(param))
            first_iteration = True
            for chunk in self._iter_chunks(self.cursor):
                if first_iteration is True:
                    df = pd.DataFrame(chunk)
                    first_iteration = False
                else:
                    df = pd.concat([df, pd.DataFrame(chunk)], ignore_index=True)
            df.columns = [desc.name for desc in self.cursor.description]
            self.cursor.close()

        except (Exception, pg.DatabaseError) as error:
            datalake_logger.error(error)
        finally:
            PostgreSQLBase._db_disconnect()
            gc.collect()
        return df

    def _iter_chunks(self, cursor, size=40000):
        while True:
            chunk = cursor.fetchmany(size)
            if not chunk:
                break
            yield chunk


class ToDatahubWriter:

    def __init__(self, topics, schema='wpm_anlt'):
        self.start_time = datetime.datetime.now()
        self.run_id = int(time.time())
        self.topic = topics
        self.schema = schema

    def write_info(self, df, task_name='default', date_of_processing=True):

        if date_of_processing == True:
            df.insert(loc=len(df.columns), column='date_of_processing', value=df.shape[0] * [datetime.datetime.now()])
        df.insert(loc=len(df.columns), column='run_id', value=df.shape[0] * [self.run_id])

        engine = create_engine(
            'postgresql://' + credentials.USER_NAME + ':' + credentials.PASS + '@' + credentials.HOST_NAME_FOR_RESULTS +
            ':' + credentials.PORT_NAME + '/' + credentials.DB_NAME, echo=False)

        table_name = 'epm_wpm_ds_staff_' + self.topic

        if task_name == 'default':
            task_name = 'preprocessor_' + self.topic

        df.to_sql(name=table_name, con=engine, schema=self.schema,
                  if_exists='append')

        self.end_time = datetime.datetime.now()

        datalake_logger.info('date info written into datahub successfully')

        run_info = pd.DataFrame([{
            'run_id': self.run_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'code_version': '1.0',
            'comment': 'from OpenShift',
            'success': True,
            'task_type': task_name
        }]).set_index('run_id')

        run_info.to_sql(name='epm_wpm_ds_staff_run_info', con=engine, schema='wpm_anlt',
                        if_exists='append')

        engine.dispose()