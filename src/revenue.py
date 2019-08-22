from datahub import PostgreSQLBase, ToDatahubWriter
import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import datetime
import auxiliary as aux
import json
import xgboost as xgb
reload (aux)
import variable_selection
reload(variable_selection)
import data_splitting
reload(data_splitting)
import pipeline_classes
reload(pipeline_classes)
import pickle
import features
reload (features)


logger = logging.getLogger('rampdown')

class RevenueInStaffing(features.Features):

    def __init__(self, job='making_predictions', date='none'):

        """
        Initiation of the class will query base dataset for different purposes. One of them is model training. Another one is making predictions.
        :param job:str
        Possible job values:
            - 'making_predictions' with date=['YYYY-mm-dd', 'YYYY-mm-dd', ...] - list of dates to make predictions;
            - 'model_training' with date='YYYY-mm-dd' - for specific date;

        :param date: str for 'model_training' job or list for 'making_predictions' job
        Possible date valued:
            - ['YYYY-mm-dd', 'YYYY-mm-dd', ...]. To make predictions for specific days;
            - 'YYYY-mm-dd' - to train model by the specific date.
        """

        postgres = PostgreSQLBase()
        self.job = job
        self.date = date
        if self.job == 'model_training':
            self.key = 'revenue_' + self.job + '_' + self.date
        else:
            self.key = 'revenue_' + self.job + '_' + self.date[-1][:7]

        logger.info('{} dataset creation query started'.format(self.key))

        if job == 'making_predictions':

            self.df = postgres.get_data('active_positions_to_make_predictions.sql',  param=self.date, param_type='list')

        elif job == 'model_training':

            self.df = postgres.get_data('active_positions_per_date_dataset_snaps.sql', param=self.date, param_type='day')
            self.df = self.df[self.df.timesheet_date.notnull()]

        elif job == 'none':
            self.df = pd.DataFrame()

        logger.info('{} dataset was created successfully'.format(self.key))


    def feature_processing(self):

        """
        This method prosesses additional features for positions in dataset.
        :return: pd.DataFrame
        """

        logger.info('{} feature processing script started'.format(self.key))

        df_proc = aux.DataFrameProcessor()
        logger.debug(self.df.shape)

        # self.df['key'] = self.df.position_id.astype(str) + self.df.planned_end_date.astype(str)
        # self.df = self.df.sample(frac=1)
        # self.df = self.df.drop_duplicates('key')
        # self.df = self.df.drop(['key'], 1)

        self.processor_wkl_start_date()

        self.df = df_proc.start_processing(self.df)

        if self.job == 'model_training':

            self.df = self.df[self.df.timesheet_date.notnull()]

            self.df = df_proc.datetime_cols(self.df, ['date', 'timesheet_date'])
            # self.df = df_proc.time_distance(self.df, 'date', 'timesheet_date', 'distance')

            self.df['cur_month'] = self.df.date.map(lambda d: datetime.datetime.strftime(d, '%Y-%m-%d')[:7])
            self.df['next_1_month'] = self.df.date.map(
                lambda d: datetime.datetime.strftime(d + relativedelta(months=1), '%Y-%m-%d')[:7])
            self.df['next_2_month'] = self.df.date.map(
                lambda d: datetime.datetime.strftime(d + relativedelta(months=2), '%Y-%m-%d')[:7])

            self.df['report_month'] = self.df.timesheet_date.map(lambda d: datetime.datetime.strftime(d, '%Y-%m-%d')[:7])

            self.df['next_plus_months'] = np.where(
                self.df.report_month == self.df.cur_month, 0, np.where(
                    self.df.report_month == self.df.next_1_month, 0, np.where(
                        self.df.report_month == self.df.next_2_month, 0, 1
                    )
                )
            )
            self.df['cur_month'] = np.where(self.df.cur_month == self.df.report_month, 1, 0)
            self.df['next_1_month'] = np.where(self.df.next_1_month == self.df.report_month, 1, 0)
            self.df['next_2_month'] = np.where(self.df.next_2_month == self.df.report_month, 1, 0)

            # targets:
            # 0 - first date in the current month
            # 1 - first date in the next 1 month
            # 2 - first date in the next 2 month
            # 3 - first date after the next 2 month
            # 4 - shouldn't be in dataset at all, if any the check is needed

            self.df['target'] = np.where(
                self.df.cur_month == 1, 0, np.where(
                    self.df.next_1_month == 1, 1, np.where(
                        self.df.next_2_month == 1, 2, np.where(
                            self.df.next_plus_months == 1, 3, 4
                        )
                    )
                )
            )

            self.df = self.df.drop(['cur_month', 'next_1_month', 'next_2_month', 'next_plus_months', 'report_month'], 1)

            max_date = pd.DataFrame(self.df.groupby(['position_id'])['date'].max()).reset_index()
            max_date['position_id'] = max_date['position_id'].astype(str)
            self.df['position_id'] = self.df['position_id'].astype(str)

            self.df = pd.merge(
                self.df,
                max_date.set_index('position_id').rename(columns={'date': 'max_date'}),
                how='left', left_on='position_id', right_index=True
            )

            self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        elif self.job == 'making_predictions':

            self.df['target'] = 0

        logger.debug(self.df.shape)
        self.processor_project_information()
        logger.debug(self.df.shape)
        self.processor_bench_buckets()
        logger.debug(self.df.shape)
        self.processor_active_positions_counting()
        logger.debug(self.df.shape)
        self.processor_time_reporting()
        logger.debug(self.df.shape)
        self.processor_adding_proposals_per_positions()
        logger.debug(self.df.shape)
        self.processor_total_proposals_counting()
        logger.debug(self.df.shape)
        self.processor_adding_current_proposals_per_positions()
        logger.debug(self.df.shape)
        self.processors_from_first_proposal_booking_onboarding()
        logger.debug(self.df.shape)
        self.processors_from_last_proposal_booking_onboarding()
        logger.debug(self.df.shape)
        self.processor_active_positions_per_customers()
        logger.debug(self.df.shape)
        self.processor_active_positions_per_projects()
        logger.debug(self.df.shape)
        self.processor_unit_ids()
        logger.debug(self.df.shape)
        self.processor_time_relations_features()
        logger.debug(self.df.shape)
        self.processor_on_bench_counter()
        logger.debug(self.df.shape)
        self.processing_total_active_positions()
        logger.debug(self.df.shape)
        self.processor_supply_vs_demand()
        logger.debug(self.df.shape)
        self.processor_veiws_per_prev_N_days_div_total_views()
        logger.debug(self.df.shape)
        self.processor_date_of_max_views()
        logger.debug(self.df.shape)
        self.processor_date_of_starting_views_growth()
        logger.debug(self.df.shape)
        self.processor_date_of_starting_version_growth()
        logger.debug(self.df.shape)


    def crm_info_processing(self):

        logger.info('{} CRM info processing script started'.format(self.key))

        self.processor_crm_info()
        logger.info('{} CRM info were processed successfully'.format(self.key))


    def pickle_df(self):

        self.df.to_pickle('../data/df_{}.pkl'.format(self.key))
        logger.info('df_{}.pkl was pickled'.format(self.key))


    def get_df(self):

        return self.df


    def model_best_params(self, task_type='cancellation_probability', sorting='run_id'):
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        df = postgres.get_data('select_best_parameters.sql')

        df.metrics = df.metrics.map(lambda x: json.loads(x))
        df.best_params = df.best_params.map(lambda x: json.loads(x))

        df = df[df.task_type == task_type]

        if sorting == 'run_id':
            df = df_proc.datetime_cols(df, [sorting])

        else:
            df[sorting] = df.metrics.map(lambda x: x[sorting])

        df = df.set_index(sorting)

        self.best_params = df.at[max(df.index), 'best_params']


    def model_training(self):
        logger = logging.getLogger(__name__ + ' : model_training: {}'.format(self.key))

        self.feature_processing()

        logger.info('model_training: {}'.format(self.key))

        train = self.df.copy()

        text_data_columns, categorical_data_columns, multicategorical_data_columns, numeric_data_columns = \
            variable_selection.variable_selection(train)

        train[categorical_data_columns] = train[categorical_data_columns].astype(str)

        X_train, y_train = data_splitting.X_y_split(train)

        model = pipeline_classes.PipelineEx([
            ('features', pipeline_classes.features(categorical_data_columns=categorical_data_columns,
                                                   multicategorical_data_columns=multicategorical_data_columns,
                                                   numeric_data_columns=numeric_data_columns)),
            ('classifier', xgb.XGBClassifier(n_jobs=-1, objective='multi:softmax', num_class=3))
        ])

        self.model_best_params()
        model.set_params(**self.best_params)

        model.fit(X_train, y_train)

        filename = '../data/revenue_model_{}.sav'.format(self.date)
        pickle.dump(model, open(filename, 'wb'))

        logger.info('{} probability model trained successfully and dumped to pickle file'.format(self.key))


    def making_predictions(self, model_file, dates_type='none'):

        logger.info('{} making predictions script started'.format(self.key))

        self.feature_processing()

        logger.debug('date parameter in the revenue object are {}'.format(str(self.date)))

        if dates_type != 'none':
            self.df = self.df[self.df.date == max(self.df.date)]

        logger.info('{} making predictions script started'.format(self.key))

        filename = '../data/{}'.format(model_file)
        model = pickle.load(open(filename, 'rb'))

        to_datahub = ToDatahubWriter('{}_predictions'.format('revenue')) #rename to table name

        self.df.date = self.df.date.astype(str)

        for d in self.df.date.unique():

            logger.debug(d)

            X_predict, y = data_splitting.X_y_split(self.df[self.df.date == d])

            revenue_prediction = model.predict_proba(X_predict)

            prd = X_predict.copy()

            prd.insert(len(prd.columns), column='revenue_current_month_probability', value=revenue_prediction[:, 0])
            prd.insert(len(prd.columns), column='revenue_next_month_probability', value=revenue_prediction[:, 1])
            prd.insert(len(prd.columns), column='revenue_after_next_month_probability', value=revenue_prediction[:, 2])

            assert len(
                prd) > 3000, 'workload_extension_predictions data set has a problem! Too small data set!!!'

            prd = prd[['revenue_prediction', 'position_id']]
            prd.insert(len(prd.columns), column='date', value=d)

            prd = prd.set_index('position_id')

            to_datahub.write_info(prd, task_name='revenue')

            logger.info('{} predictions per date={} are successfully written to datahub'.format(self.key, d))