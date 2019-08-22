import auxiliary as aux
from datahub import PostgreSQLBase
import logging
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle
import datetime
from pandas.tseries.offsets import BDay


class Features:


    def __init__(self):
        self.df = pd.DataFrame()
        self.key = ''


    def processor_wkl_start_date(self):
        logger = logging.getLogger(__name__ + ' : processor_wkl_start_date: {}'.format(self.key))
        logger.info('processor_wkl_start_date: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        wkl_start_date = postgres.get_data('select_wkl_start_date.sql', tuple(self.df['date'].astype(str).unique()),
                                          param_type='days')

        self.df.position_id = self.df.position_id.astype(str)
        wkl_start_date.position_id = wkl_start_date.position_id.astype(str)

        wkl_start_date = wkl_start_date[wkl_start_date['position_id'].isin(self.df.position_id.unique())]

        self.df = pd.merge(
            self.df,
            wkl_start_date.set_index(wkl_start_date.position_id.astype(str) + wkl_start_date.date.astype(str))[[
                'workload_start_date'
            ]],
            how='left', left_on=self.df.position_id.astype(str) + self.df.date.astype(str), right_index=True
        )

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_project_information(self):
        logger = logging.getLogger(__name__ + ' : processor_project_information: {}'.format(self.key))
        logger.info('comments and processor_project_information: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)

        df_proj = postgres.get_data('project_info.sql')
        df_comments = postgres.get_data('comments.sql')

        df_comments = df_comments[df_comments.id.notnull()]
        df_comments['id'] = df_comments['id'].astype(int)
        #df.comment_id = df.comment_id.map(lambda x: 0 if x == '' else x).astype(int)
        self.df.comment_id = self.df.comment_id.astype(int)
        df_comments = df_comments[df_comments['id'].isin(self.df.comment_id.unique())]

        self.df = pd.merge(
            self.df,
            df_comments.set_index('id').rename(columns={'text': 'comments'}),
            how='left', left_on='comment_id', right_index=True
        )
        self.df['comments'] = self.df['comments'].fillna('')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge with comment_id operation!'

        df_proj.drop_duplicates(['project_code'], inplace=True)

        self.df = pd.merge(
            self.df,
            df_proj.set_index('project_code')[[
                'proj_description', 'proj_start_date', 'proj_end_date', 'proj_is_billable', 'is_using_template']],
            how='left', left_on='container_name', right_index=True
        )
        self.df[['proj_description']] = self.df[['proj_description']].fillna('')

        self.df[['proj_is_billable', 'is_using_template']] = self.df[['proj_is_billable', 'is_using_template']].fillna('__UNDEFINED__')

        assert length == len(
            self.df), 'Assertion!!! The dataframe length was changed after merge with projects info operation!'

        self.df = df_proc.text_cols(self.df, ['comments', 'proj_description'])
        self.df = df_proc.datetime_cols(self.df, ['proj_start_date', 'proj_end_date', 'created', 'planned_start_date'])
        self.df = df_proc.time_distance(self.df, 'proj_start_date', 'proj_end_date', 'proj_duration')
        self.df = df_proc.time_distance(self.df, 'proj_start_date', 'created', 'proj_before_created')
        self.df = df_proc.time_distance(self.df, 'created', 'proj_end_date', 'proj_after_created')
        self.df = df_proc.time_distance(self.df, 'proj_start_date', 'planned_start_date', 'start_pos_in_proj')
        self.df = df_proc.time_distance(self.df, 'planned_start_date', 'proj_end_date', 'end_pos_in_proj')

        self.df.drop(['comment_id'], 1, inplace=True)

        # self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert self.df.proj_duration.sum() > 0, 'Feature problem! Project information features were not joined!!!'


    def processor_bench_buckets(self):
        logger = logging.getLogger(__name__ + ' : processor_bench_buckets: {}'.format(self.key))
        logger.info('processor_bench_buckets: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()
        loc_ids = aux.City_Country()

        length = len(self.df)
        cols = self.df.columns

        bench_buckets = postgres.get_data('select_bench_buckets.sql', tuple(self.df['date'].astype(str).unique()),
                                          param_type='days')
        bench_buckets = bench_buckets.rename(columns={'location_id': 'staffing_locations_city_id'})
        bench_buckets = loc_ids.vlookuper(bench_buckets)
        bench_buckets = bench_buckets.dropna()

        self.df = df_proc.loc_skill_target_processing(
            self.df, bench_buckets, target_column='bench_buckets', agregate_column='number_of_employees', prefix='bench_buckets')
        df_proc.feature_join_result_check(self.df, bench_buckets, target_column='bench_buckets',
                                          agregate_column='number_of_employees', prefix='bench_buckets')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_active_positions_counting(self):
        logger = logging.getLogger(__name__ + ' : processor_active_positions_counting: {}'.format(self.key))
        logger.info('processor_active_positions_counting: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        df_active_positions = postgres.get_data(
            'select_active_positions_counting.sql', tuple(self.df['date'].astype(str).unique()), param_type='days')

        df_active_positions = df_active_positions.rename(
            columns={
                'city_id': 'staffing_locations_city_id',
                'country_id': 'staffing_locations_country_id'
            }
        )

        self.df = df_proc.loc_skill_target_processing(
            self.df, df_active_positions, target_column='staffing_status', agregate_column='number_of_positions', prefix='active_positions')
        df_proc.feature_join_result_check(self.df, df_active_positions, target_column='staffing_status',
                                          agregate_column='number_of_positions', prefix='active_positions')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_time_reporting(self):

        logger = logging.getLogger(__name__ + ' : processor_time_reporting: {}'.format(self.key))
        logger.info('processor_time_reporting: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()
        loc_ids = aux.City_Country()

        length = len(self.df)
        cols = self.df.columns

        df_time_reporting = postgres.get_data(
            'select_time_reporting.sql', tuple(self.df['date'].astype(str).unique()), param_type='days')

        df_time_reporting = df_time_reporting.rename(columns={'location_id': 'staffing_locations_city_id'})
        df_time_reporting = loc_ids.vlookuper(df_time_reporting)

        df_time_reporting.insert(
            loc=len(df_time_reporting.columns), column='total_hours_col',
            value=df_time_reporting.shape[0] * ['total_hours']) #to use method loc_skill_target_processing of df_proc object
        df_time_reporting.insert(
            loc=len(df_time_reporting.columns), column='in_trip_hours_col',
            value=df_time_reporting.shape[0] * ['in_trip_hours']) #to use method loc_skill_target_processing of df_proc object
        df_time_reporting.insert(
            loc=len(df_time_reporting.columns), column='billable_hours_col',
            value=df_time_reporting.shape[0] * ['billable_hours']) #to use method loc_skill_target_processing of df_proc object

        self.df = df_proc.loc_skill_target_processing(
            self.df, df_time_reporting, target_column='total_hours_col', agregate_column='total_hours', res_type=float, prefix='time_reporting')
        self.df = df_proc.loc_skill_target_processing(
            self.df, df_time_reporting, target_column='in_trip_hours_col', agregate_column='in_trip_hours', res_type=float, prefix='time_reporting')
        self.df = df_proc.loc_skill_target_processing(
            self.df, df_time_reporting, target_column='billable_hours_col', agregate_column='billable_hours', res_type=float, prefix='time_reporting')

        df_proc.feature_join_result_check(self.df, df_time_reporting, target_column='total_hours_col', agregate_column='total_hours', prefix='time_reporting')
        df_proc.feature_join_result_check(self.df, df_time_reporting, target_column='in_trip_hours_col',
                                          agregate_column='in_trip_hours', prefix='time_reporting')
        df_proc.feature_join_result_check(self.df, df_time_reporting, target_column='billable_hours_col',
                                          agregate_column='billable_hours', prefix='time_reporting')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_adding_proposals_per_positions(self):
        logger = logging.getLogger(__name__ + ' : processor_adding_proposals_per_positions: {}'.format(self.key))
        logger.info('processor_adding_proposals_per_positions: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        df_proposals = postgres.get_data('select_proposals_per_positions.sql')

        self.df = df_proc.datetime_cols(self.df, ['date'])
        df_proposals = df_proc.datetime_cols(df_proposals, ['date'])

        self.df.position_id = self.df.position_id.astype(int)
        df_proposals.position_id = df_proposals.position_id.astype(int)

        statuses = {
            'rejected_proposals': 'Rejected',
            'cancelled_proposals': 'Cancelled',
            'proposed_proposals': 'Proposed',
            'booked_proposals': 'Booked',
            'onboarding_proposals': 'Onboarding',
            'preselected_proposals': 'Preselected'
        }

        tmp = df_proposals.sort_values('date')

        tmp['total_proposals_changes'] = tmp.groupby(['position_id'])['proposal_id'].cumcount() + 1
        tmp.total_proposals_changes = tmp.total_proposals_changes.astype(int)
        tmp['key'] = tmp.date.astype(str) + tmp.position_id.astype(str)
        tmp = tmp.drop_duplicates('key', keep='last')
        tmp = tmp.set_index('key').drop(['proposal_id', 'position_id', 'staffing_status', 'date'], 1)

        df_proposals = pd.merge(df_proposals, tmp, how='left',
                                left_on=df_proposals.date.astype(str) + df_proposals.position_id.astype(str),
                                right_index=True).fillna(0)

        df_proposals = df_proposals.set_index(df_proposals.date.astype(str) + df_proposals.position_id.astype(str))

        days = pd.concat([self.df[['date', 'position_id']], df_proposals[['date', 'position_id']]])
        days['key'] = days.index
        days.drop_duplicates('key', inplace=True)
        days.drop(['key'], 1, inplace=True)

        df_proposals = pd.merge(days, df_proposals.drop(['date', 'position_id'], 1),
                                how='left', left_index=True, right_index=True)

        df_proposals = df_proposals[df_proposals.position_id.isin(self.df.position_id.unique())]

        df_proposals.total_proposals_changes = df_proposals.total_proposals_changes.fillna(0)
        df_proposals.total_proposals_changes = df_proposals.total_proposals_changes.map(lambda x: 1 if x > 0 else 0)
        df_proposals.sort_values('date', inplace=True)
        df_proposals.total_proposals_changes = df_proposals.groupby(['position_id'])['total_proposals_changes'].cumsum()

        for col in statuses.keys():
            tmp = df_proposals[
                df_proposals.staffing_status == statuses[col]
                ].sort_values('date')

            tmp[col] = tmp.groupby(['position_id'])['proposal_id'].cumcount() + 1
            tmp['key'] = tmp.date.astype(str) + tmp.position_id.astype(str) + tmp.staffing_status.astype(str)
            tmp = tmp.drop_duplicates('key', keep='last')
            tmp = tmp.set_index('key')
            tmp = tmp[[col]]

            df_proposals = pd.merge(df_proposals, tmp, how='left',
                                    left_on=df_proposals.date.astype(str) + df_proposals.position_id.astype(
                                        str) + df_proposals.staffing_status.astype(str),
                                    right_index=True)

            df_proposals[col] = df_proposals[col].fillna(0).astype(int)

            df_proposals[col] = df_proposals[col].map(lambda x: 1 if x > 0 else 0)

            df_proposals.sort_values('date', inplace=True)

            df_proposals[col] = df_proposals.groupby(['position_id'])[col].cumsum()

        df_proposals['key'] = df_proposals.position_id.astype(str) + df_proposals.date.astype(str)
        df_proposals = df_proposals.sort_values('date', ascending=False)
        df_proposals = df_proposals.drop_duplicates(['key'])

        df_proposals.set_index(df_proposals['key'], inplace=True)

        df_proposals.drop(['proposal_id', 'staffing_status', 'position_id', 'date', 'key'], 1, inplace=True)

        self.df = pd.merge(
            self.df, df_proposals,
            how='left', left_on=self.df.position_id.astype(str) + self.df.date.astype(str), right_index=True
        ).fillna(0)

        self.df['total_proposals_changes'] = self.df.proposed_proposals + self.df.onboarding_proposals + self.df.preselected_proposals + \
                                             self.df.booked_proposals + self.df.cancelled_proposals + self.df.rejected_proposals

        for col in statuses.keys():
            assert self.df[col].sum() > 0, ' Feature problem! Proposals_per_positions were not joined!!!'

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_total_proposals_counting(self):
        logger = logging.getLogger(__name__ + ' : processor_total_proposals_counting: {}'.format(self.key))
        logger.info('processor_total_proposals_counting: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        df_proposals = postgres.get_data('select_total_proposals.sql', tuple(self.df['date'].astype(str).unique()),
                                          param_type='days')
        df_proposals = df_proposals.rename(
            columns={
                'city_id': 'staffing_locations_city_id',
                'country_id': 'staffing_locations_country_id'
            }
        )

        self.df = df_proc.loc_skill_target_processing(
            self.df, df_proposals, target_column='staffing_status', agregate_column='number_of_proposals', prefix='total_proposals')
        df_proc.feature_join_result_check(self.df, df_proposals, target_column='staffing_status',
                                          agregate_column='number_of_proposals', prefix='total_proposals')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_adding_current_proposals_per_positions(self):
        """Adds current proposals to positions per date in dataset.
        Proposals would be in statuses 'Proposed', 'Preselected', 'Booked', 'Onboarding'.

        Parameters
        ----------
        data : pd.DataFrame
            Original dataset to add current proposals.
            The dataset should have at least two fields:
                - position_id (numeric of int)
                - date (str with format like 'YYYY-mm-dd')

        Returns
        -------
        updated dataset : pd.DataFrame
            With all origin columns.
            And 15 additional columns:
                1. booked_external_current_proposals -
                        number of external applicants per date with staffing status 'Booked'
                2. booked_internal_current_proposals -
                        number of internal candidates per date with staffing status 'Booked'
                3. booked_total_current_proposals -
                        total number of external and internal candidates with staffing status 'Booked'
                4. onboarding_external_current_proposals -
                        number of external applicants per date with staffing status 'Onboarding'
                5. onboarding_internal_current_proposals -
                        number of internal candidates per date with staffing status 'Onboarding'
                6. onboarding_total_current_proposals -
                        total number of external and internal candidates with staffing status 'Onboarding'
                7. preselected_external_current_proposals -
                        number of external applicants per date with staffing status 'Preselected'
                8. preselected_internal_current_proposals -
                        number of internal candidates per date with staffing status 'Preselected'
                9. preselected_total_current_proposals -
                        total number of external and internal candidates with staffing status 'Preselected'
                10. proposed_external_current_proposals -
                        number of external applicants per date with staffing status 'Proposed'
                11. proposed_internal_current_proposals -
                        number of internal candidates per date with staffing status 'Proposed'
                12. proposed_total_current_proposals -
                        total number of external and internal candidates with staffing status 'Proposed'
                13. total_external_current_proposals -
                        total number of all active external applicants per date with any staffing status
                14. total_internal_current_proposals -
                        total number of all active internal candidates per date with any staffing status
                15. total_total_current_proposals -
                        total number of external and internal candidates with any staffing status

        """

        logger = logging.getLogger(__name__ + ' : processor_adding_current_proposals_per_positions: {}'.format(self.key))
        logger.info('processor_adding_current_proposals_per_positions: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        columns = self.df.columns

        df_prop = postgres.get_data('select_current_proposals_per_positions.sql')

        self.df.position_id = self.df.position_id.astype(int)
        df_prop.position_id = df_prop.position_id.astype(int)
        df_prop = df_prop[df_prop.position_id.isin(self.df.position_id.unique())]

        self.df = df_proc.datetime_cols(self.df, ['date'])
        df_prop = df_proc.datetime_cols(df_prop, ['date'])

        self.df.position_id = self.df.position_id.astype(int)
        df_prop.position_id = df_prop.position_id.astype(int)

        def adding_proposals(dataset, df_prop, name='total'):

            df = dataset.copy()
            df_proposals = df_prop.copy()

            statuses = {
                'proposed_' + name + '_current_proposals': 'Proposed',
                'booked_' + name + '_current_proposals': 'Booked',
                'onboarding_' + name + '_current_proposals': 'Onboarding',
                'preselected_' + name + '_current_proposals': 'Preselected'
            }

            if (name == 'total'):
                tmp = pd.DataFrame(
                    df_proposals.groupby(['position_id', 'date'])['proposal_id'].count()).reset_index().rename(
                    columns={'proposal_id': 'total_' + name + '_current_proposals'})
                tmp = tmp.set_index(tmp.position_id.astype(str) + tmp.date.astype(str)).drop(
                    ['position_id', 'date'], 1)

                df_proposals = pd.merge(df_proposals, tmp, how='left',
                                        left_on=df_proposals.position_id.astype(str) + df_proposals.date.astype(str),
                                        right_index=True).fillna(0)

            else:
                tmp = pd.DataFrame(df_proposals[
                                       df_proposals.applicant == name
                                       ].groupby(['position_id', 'date'])['proposal_id'].count()).reset_index().rename(
                    columns={'proposal_id': 'total_' + name + '_current_proposals'})
                tmp = tmp.set_index(tmp.position_id.astype(str) + tmp.date.astype(str)).drop(
                    ['position_id', 'date'], 1)

                df_proposals = pd.merge(df_proposals, tmp, how='left',
                                        left_on=df_proposals.position_id.astype(str) + df_proposals.date.astype(str),
                                        right_index=True).fillna(0)

            for col in statuses.keys():

                if (name == 'total'):
                    tmp = pd.DataFrame(df_proposals[
                                           (df_proposals.staffing_status == statuses[col])
                                       ].groupby(['position_id', 'date'])['proposal_id'].count()).reset_index().rename(
                        columns={'proposal_id': col})
                    tmp = tmp.set_index(tmp.position_id.astype(str) + tmp.date.astype(str)).drop(
                        ['position_id', 'date'], 1)

                    df_proposals = pd.merge(df_proposals, tmp, how='left',
                                            left_on=df_proposals.position_id.astype(str) + df_proposals.date.astype(str),
                                            right_index=True).fillna(0)

                else:
                    tmp = pd.DataFrame(df_proposals[
                                           (df_proposals.staffing_status == statuses[col]) &
                                           (df_proposals.applicant == name)
                                           ].groupby(['position_id', 'date'])['proposal_id'].count()).reset_index().rename(
                        columns={'proposal_id': col})
                    tmp = tmp.set_index(tmp.position_id.astype(str) + tmp.date.astype(str)).drop(
                        ['position_id', 'date'], 1)

                    df_proposals = pd.merge(df_proposals, tmp, how='left',
                                            left_on=df_proposals.position_id.astype(str) + df_proposals.date.astype(str),
                                            right_index=True).fillna(0)

            df_proposals['key'] = df_proposals.position_id.astype(str) + df_proposals.date.astype(str)
            df_proposals = df_proposals.sort_values('date', ascending=False)
            df_proposals = df_proposals.drop_duplicates(['key'])

            df_proposals.set_index(df_proposals['key'], inplace=True)

            df_proposals.drop(['proposal_id', 'staffing_status', 'position_id', 'date', 'key', 'applicant'],
                              1, inplace=True)

            df = pd.merge(
                df, df_proposals,
                how='left', left_on=df.position_id.astype(str) + df.date.astype(str), right_index=True
            ).fillna(0)

            for col in list(set(df.columns) - set(dataset.columns)):
                df[col] = df[col].astype(int)

            return df

        self.df = adding_proposals(self.df, df_prop, name='total')
        self.df = adding_proposals(self.df, df_prop, name='external')
        self.df = adding_proposals(self.df, df_prop, name='internal')

        for col in list(set(self.df.columns) - set(columns)):
            assert self.df[col].sum() > 0, ' Feature problem! Current_proposals_per_positions were not joined!!!'

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(columns)))


    def processors_from_first_proposal_booking_onboarding(self):
        logger = logging.getLogger(__name__ + ' : processors_from_first_proposal_booking_onboarding: {}'.format(self.key))
        logger.info('processors_from_first_proposal_booking_onboarding: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        columns = self.df.columns

        self.df = df_proc.datetime_cols(self.df, ['date'])

        first_proposals = postgres.get_data('select_first_proposals.sql')
        first_booking = postgres.get_data('select_first_booking.sql')
        first_onboarding = postgres.get_data('select_first_onboarding.sql')

        first_proposals.set_index('position_id', inplace=True)
        first_booking.set_index('position_id', inplace=True)
        first_onboarding.set_index('position_id', inplace=True)

        self.df = pd.merge(self.df, first_proposals, how='left', left_on='position_id', right_index=True)
        self.df['time_from_first_proposals'] = np.where((self.df['date'] > self.df['time']), self.df.date - self.df.time, 0)
        self.df.time_from_first_proposals = self.df.time_from_first_proposals.dt.days
        self.df = self.df.rename(columns={'time': 'first_proposals_date'})
        # self.df.drop(['time'], 1, inplace=True)

        self.df = pd.merge(self.df, first_booking, how='left', left_on='position_id', right_index=True)
        self.df['time_from_first_booking'] = np.where((self.df['date'] > self.df['time']), self.df.date - self.df.time, 0)
        self.df.time_from_first_booking = self.df.time_from_first_booking.dt.days
        self.df = self.df.rename(columns={'time': 'first_booking_date'})
        # self.df.drop(['time'], 1, inplace=True)

        self.df = pd.merge(self.df, first_onboarding, how='left', left_on='position_id', right_index=True)
        self.df['time_from_first_onboarding'] = np.where((self.df['date'] > self.df['time']), self.df.date - self.df.time, 0)
        self.df.time_from_first_onboarding = self.df.time_from_first_onboarding.dt.days
        self.df = self.df.rename(columns={'time': 'first_onboarding_date'})
        # self.df.drop(['time'], 1, inplace=True)

        # for col in list(set(self.df.columns) - set(columns)):
        #     assert self.df[col].sum() > 0, ' Feature problem! First_proposal_booking_onboarding were not joined!!!'

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(columns)))


    def processors_from_last_proposal_booking_onboarding(self):
        logger = logging.getLogger(__name__ + ' : processors_from_last_proposal_booking_onboarding: {}'.format(self.key))
        logger.info('processors_from_last_proposal_booking_onboarding: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        columns = self.df.columns

        self.df = df_proc.datetime_cols(self.df, ['date'])

        last_proposals = postgres.get_data('select_last_proposals.sql')
        last_booking = postgres.get_data('select_last_booking.sql')
        last_onboarding = postgres.get_data('select_last_onboarding.sql')

        last_proposals.set_index('position_id', inplace=True)
        last_booking.set_index('position_id', inplace=True)
        last_onboarding.set_index('position_id', inplace=True)

        self.df = pd.merge(self.df, last_proposals, how='left', left_on='position_id', right_index=True)
        self.df['time_from_last_proposals'] = np.where((self.df['date'] > self.df['time']), self.df.date - self.df.time, 0)
        self.df.time_from_last_proposals = self.df.time_from_last_proposals.dt.days
        self.df = self.df.rename(columns={'time': 'last_proposals_date'})
        # self.df.drop(['time'], 1, inplace=True)

        self.df = pd.merge(self.df, last_booking, how='left', left_on='position_id', right_index=True)
        self.df['time_from_last_booking'] = np.where((self.df['date'] > self.df['time']), self.df.date - self.df.time, 0)
        self.df.time_from_last_booking = self.df.time_from_last_booking.dt.days
        self.df = self.df.rename(columns={'time': 'last_booking_date'})
        # self.df.drop(['time'], 1, inplace=True)

        self.df = pd.merge(self.df, last_onboarding, how='left', left_on='position_id', right_index=True)
        self.df['time_from_last_onboarding'] = np.where((self.df['date'] > self.df['time']), self.df.date - self.df.time, 0)
        self.df.time_from_last_onboarding = self.df.time_from_last_onboarding.dt.days
        self.df = self.df.rename(columns={'time': 'last_onboarding_date'})
        # self.df.drop(['time'], 1, inplace=True)

        # for col in list(set(self.df.columns) - set(columns)):
        #     assert self.df[col].sum() > 0, ' Feature problem! Last_proposal_booking_onboarding were not joined!!!'

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(columns)))


    def processor_active_positions_per_customers(self):
        logger = logging.getLogger(__name__ + ' : processor_active_positions_per_customers: {}'.format(self.key))
        logger.info('processor_active_positions_per_customers: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        df_positions = postgres.get_data('select_active_positions_per_customers.sql',
                                         tuple(self.df['date'].astype(str).unique()),
                                         param_type='days')
        df_positions = df_positions.rename(
            columns={
                'city_id': 'staffing_locations_city_id',
                'country_id': 'staffing_locations_country_id'
            }
        )

        self.df['customer_id'] = self.df['customer_id'].astype(str)
        df_positions['customer_id'] = df_positions['customer_id'].astype(str)

        df_positions.insert(
            loc=len(df_positions.columns), column='aux_column',
            value=df_positions.shape[0] * ['id'])  # to use method loc_skill_target_processing of df_proc object

        self.df['dates'] = self.df['date']
        self.df['date'] = self.df.date.astype(str) + self.df.customer_id.astype(
            str)  # to use method loc_skill_target_processing of df_proc object
        df_positions['date'] = df_positions.date.astype(str
                                                        ) + df_positions.customer_id.astype(
            str)  # to use method loc_skill_target_processing of df_proc object

        self.df = df_proc.loc_skill_target_processing(
            self.df, df_positions, target_column='aux_column', agregate_column='number_of_positions', prefix='customer')
        df_proc.feature_join_result_check(self.df, df_positions, target_column='aux_column',
                                          agregate_column='number_of_positions', prefix='customer')

        self.df['date'] = self.df['dates']
        self.df.drop(['dates'], 1, inplace=True)

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_active_positions_per_projects(self):
        logger = logging.getLogger(__name__ + ' : processor_active_positions_per_projects: {}'.format(self.key))
        logger.info('processor_active_positions_per_projects: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        df_positions = postgres.get_data('select_active_positions_per_projects.sql', tuple(self.df['date'].astype(str).unique()),
                                         param_type='days')
        df_positions = df_positions.rename(
            columns={
                'city_id': 'staffing_locations_city_id',
                'country_id': 'staffing_locations_country_id'
            }
        )

        df_positions['container_name'] = df_positions['container_name'].map(
            lambda x: "".join(i for i in x if ord(i) < 128)
        )
        self.df['container_name'] = self.df['container_name'].map(
            lambda x: "".join(i for i in x if ord(i) < 128)
        )

        self.df['container_name'] = self.df['container_name'].astype(str)
        df_positions['container_name'] = df_positions['container_name'].astype(str)

        df_positions.insert(
            loc=len(df_positions.columns), column='aux_column',
            value=df_positions.shape[0] * ['name'])  # to use method loc_skill_target_processing of df_proc object

        self.df['dates'] = self.df['date']
        self.df['date'] = self.df.date.astype(str) + self.df.container_name.astype(
            str)  # to use method loc_skill_target_processing of df_proc object
        df_positions['date'] = df_positions.date.astype(str
                                                        ) + df_positions.container_name.astype(
            str)  # to use method loc_skill_target_processing of df_proc object

        self.df = df_proc.loc_skill_target_processing(
            self.df, df_positions, target_column='aux_column', agregate_column='number_of_positions', prefix='container')
        df_proc.feature_join_result_check(self.df, df_positions, target_column='aux_column',
                                          agregate_column='number_of_positions', prefix='container')

        self.df['date'] = self.df['dates']
        self.df.drop(['dates'], 1, inplace=True)

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_unit_ids(self):
        logger = logging.getLogger(__name__ + ' : processor_unit_ids: {}'.format(self.key))
        logger.info('processor_unit_ids: {}'.format(self.key))
        postgres = PostgreSQLBase()

        length = len(self.df)
        cols = self.df.columns

        df_units = postgres.get_data('employees_and_units_per_date.sql', list(self.df['date'].astype(str).unique()),
                                     param_type='days')
        df_units['keys'] = df_units.employee_id.astype(str) + df_units.date.astype(str)
        df_units.drop_duplicates('keys', inplace=True)
        df_units.set_index('keys', inplace=True)

        names = ['account_manager_id', 'supervisor_id', 'project_manager_id', 'coordinator_id', 'project_supervisor_id',
                 'delivery_supervisor_id', 'assignee_id', 'project_sponsor_id', 'creator_id',
                 'delivery_manager_id', 'hiring_manager_id', 'manager_id']

        # TO DO:
        #     multicategorical_names = ['project_coordinators', 'sales_sxecutives', 'sales_managers', 'program_managers',
        #                               'staffing_coordinators', 'supply_owners', 'demand_owners', 'container_staffing_coordinators']

        units = []

        for n in names:
            self.df = pd.merge(self.df, df_units[['unit_id']], how='left', left_on=self.df[n].astype(str) + self.df.date.astype(str),
                          right_index=True)
            self.df = self.df.rename(columns={'unit_id': n[:-3] + '_unit_id'})
            units.append(n[:-3] + '_unit_id')

        self.df[units] = self.df[units].fillna('__UNDEFINED__')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_time_relations_features(self):
        logger = logging.getLogger(__name__ + ' : processor_time_relations_features: {}'.format(self.key))
        logger.info('processor_time_relations_features: {}'.format(self.key))

        cols = self.df.columns

        new_features = []

        try:
            self.df['time_from_first_proposals_div_time_after_creation'
            ] = self.df.time_from_first_proposals * 100 / self.df.time_after_creation
            new_features.append('time_from_first_proposals_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_first_onboarding_div_time_after_creation'
            ] = self.df.time_from_first_onboarding * 100 / self.df.time_after_creation
            new_features.append('time_from_first_onboarding_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_first_booking_div_time_after_creation'
            ] = self.df.time_from_first_booking * 100 / self.df.time_after_creation
            new_features.append('time_from_first_booking_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_to_planned_start_date_div_time_after_creation'
            ] = self.df.time_to_planned_start_date * 100 / self.df.time_after_creation
            new_features.append('time_to_planned_start_date_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_first_proposals_div_planned_staffing_period'
            ] = self.df.time_from_first_proposals * 100 / self.df.planned_staffing_period
            new_features.append('time_from_first_proposals_div_planned_staffing_period')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_first_onboarding_div_planned_staffing_period'
            ] = self.df.time_from_first_onboarding * 100 / self.df.planned_staffing_period
            new_features.append('time_from_first_onboarding_div_planned_staffing_period')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_first_booking_div_planned_staffing_period'
            ] = self.df.time_from_first_booking * 100 / self.df.planned_staffing_period
            new_features.append('time_from_first_booking_div_planned_staffing_period')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_after_creation_div_planned_staffing_period'
            ] = self.df.time_after_creation * 100 / self.df.planned_staffing_period
            new_features.append('time_after_creation_div_planned_staffing_period')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_to_planned_starg_date_div_planned_staffing_period'
            ] = self.df.time_to_planned_start_date * 100 / self.df.planned_staffing_period
            new_features.append('time_to_planned_starg_date_div_planned_staffing_period')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_last_proposals_div_time_after_creation'
            ] = self.df.time_from_last_proposals * 100 / self.df.time_after_creation
            new_features.append('time_from_last_proposals_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_last_onboarding_div_time_after_creation'
            ] = self.df.time_from_last_onboarding * 100 / self.df.time_after_creation
            new_features.append('time_from_last_onboarding_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_last_booking_div_time_after_creation'
            ] = self.df.time_from_last_booking * 100 / self.df.time_after_creation
            new_features.append('time_from_last_booking_div_time_after_creation')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_last_proposals_div_time_from_first_proposals'
            ] = self.df.time_from_last_proposals * 100 / self.df.time_from_first_proposals
            new_features.append('time_from_last_proposals_div_time_from_first_proposals')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_last_onboarding_div_time_from_first_onboarding'
            ] = self.df.time_from_last_onboarding * 100 / self.df.time_from_first_onboarding
            new_features.append('time_from_last_onboarding_div_time_from_first_onboarding')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_from_last_booking_div_time_from_first_booking'
            ] = self.df.time_from_last_booking * 100 / self.df.time_from_first_booking
            new_features.append('time_from_last_booking_div_time_from_first_booking')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['planned_staffing_period_div_proj_duration'
            ] = self.df.planned_staffing_period * 100 / self.df.proj_duration
            new_features.append('planned_staffing_period_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['planned_position_period_div_proj_duration'
            ] = self.df.planned_position_period * 100 / self.df.proj_duration
            new_features.append('planned_position_period_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['proj_after_created_div_proj_duration'
            ] = self.df.proj_after_created * 100 / self.df.proj_duration
            new_features.append('proj_after_created_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['proj_before_created_div_proj_duration'
            ] = self.df.proj_before_created * 100 / self.df.proj_duration
            new_features.append('proj_before_created_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_to_planned_end_date_div_proj_duration'
            ] = self.df.time_to_planned_end_date * 100 / self.df.proj_duration
            new_features.append('time_to_planned_end_date_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['end_pos_in_proj_div_proj_duration'
            ] = self.df.end_pos_in_proj * 100 / self.df.proj_duration
            new_features.append('end_pos_in_proj_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['start_pos_in_proj_div_proj_duration'
            ] = self.df.start_pos_in_proj * 100 / self.df.proj_duration
            new_features.append('start_pos_in_proj_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            self.df['time_of_assignment_div_proj_duration'
            ] = self.df.time_of_assignment * 100 / self.df.proj_duration
            new_features.append('time_of_assignment_div_proj_duration')
        except (AttributeError, ZeroDivisionError) as e:
            pass

        self.df[new_features] = self.df[new_features].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_on_bench_counter(self):
        # function adds on_bench information to the data set.

        logger = logging.getLogger(__name__ + ' : processor_on_bench_counter: {}'.format(self.key))
        logger.info('processor_on_bench_counter: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length = len(self.df)
        cols = self.df.columns

        df_bench = postgres.get_data('select_on_bench_info.sql', tuple(self.df['date'].astype(str).unique()), param_type='days')

        df_bench = df_bench.rename(
            columns={
                'city_id': 'staffing_locations_city_id',
                'country_id': 'staffing_locations_country_id'
            }
        )

        df_bench.insert(loc=len(df_bench.columns), column='on_bench_col',
                        value=df_bench.shape[0] * [
                            'bench'])  # to use method loc_skill_target_processing of df_proc object

        self.df = df_proc.loc_skill_target_processing(
            self.df, df_bench, target_column='on_bench_col', agregate_column='number_of_employees', prefix='on')
        df_proc.feature_join_result_check(self.df, df_bench, target_column='on_bench_col',
                                          agregate_column='number_of_employees', prefix='on')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processing_total_active_positions(self):

        logger = logging.getLogger(__name__ + ' : processing_total_active_positions: {}'.format(self.key))
        logger.info('processing_total_active_positions: {}'.format(self.key))

        cols = self.df.columns

        name_finishing = [
            '_staffing_locations_city_id',
            '_staffing_locations_city_id_primary_skill_category_id',
            '_staffing_locations_city_id_primary_skill_id',
            '_staffing_locations_city_id_skill_prefix',
            '_company',
            '_staffing_locations_country_id',
            '_staffing_locations_country_id_primary_skill_category_id',
            '_staffing_locations_country_id_primary_skill_id',
            '_staffing_locations_country_id_skill_prefix',
            '_primary_skill_category_id',
            '_primary_skill_id',
            '_skill_prefix'
        ]

        for name in name_finishing:

            self.df['total_positions' + name] = 0

            for status in self.df.staffing_status.unique():
                self.df['total_positions' + name] = self.df['total_positions' + name] + self.df[
                    'active_positions_' + status.replace(' ', '_').lower() + name]

        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_supply_vs_demand(self):

        logger = logging.getLogger(__name__ + ' : processor_supply_vs_demand: {}'.format(self.key))
        logger.info('processor_supply_vs_demand: {}'.format(self.key))

        cols = self.df.columns

        name_finishing = [
            '_staffing_locations_city_id',
            '_staffing_locations_city_id_primary_skill_category_id',
            '_staffing_locations_city_id_primary_skill_id',
            '_staffing_locations_city_id_skill_prefix',
            '_company',
            '_staffing_locations_country_id',
            '_staffing_locations_country_id_primary_skill_category_id',
            '_staffing_locations_country_id_primary_skill_id',
            '_staffing_locations_country_id_skill_prefix',
            '_primary_skill_category_id',
            '_primary_skill_id',
            '_skill_prefix'
        ]

        for name in name_finishing:
            self.df['supply_vs_demand' + name] = self.df['on_bench' + name] / self.df['total_positions' + name]
            self.df[['supply_vs_demand' + name]] = self.df[['supply_vs_demand' + name]].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_veiws_per_prev_N_days_div_total_views(self, n=5):
        logger = logging.getLogger(__name__ + ' : processor_veiws_per_prev_5_days_div_total_views: {}'.format(self.key))
        logger.info('processor_veiws_per_prev_5_days_div_total_views: {}'.format(self.key))
        postgres = PostgreSQLBase()

        length = len(self.df)
        cols = self.df.columns

        views = postgres.get_data('select_views_of_positions.sql')

        views = views[views.entity_id.isin(self.df.position_id.astype(str))]
        views.date = pd.to_datetime(views.date)
        self.df.date = pd.to_datetime(self.df.date)

        self.df['prev_date'] = self.df.date.map(lambda x: x - BDay(n))

        total = pd.DataFrame(views.groupby(['entity_id', 'date'])['views'].sum()).reset_index()
        total.sort_values('date', inplace=True)
        total['prev_date'] = total.date.map(lambda x: x - BDay(n))

        days = set(list(total.prev_date.unique()) + list(total.date.unique()) +
                   list(self.df.prev_date.unique()) + list(self.df.date.unique()))

        frames = []
        for d in days:
            tmp = total[total.date <= d]

            tmp = pd.DataFrame(tmp.groupby(['entity_id'])['views'].sum()).reset_index().rename(
                columns={'entity_id': 'position_id', 'views': 'total_prev_views'})

            tmp = pd.merge(tmp.set_index('position_id'),
                           total[total.date == d][['entity_id', 'views']].set_index('entity_id'),
                           how='left', left_index=True, right_index=True).fillna(0)

            tmp.insert(len(tmp.columns), column='date', value=[str(d)[:10]] * len(tmp))
            tmp.insert(len(tmp.columns), column='prev_date', value=datetime.datetime.strptime(
                str(d)[:10], '%Y-%m-%d') - BDay(n))

            frames.append(tmp)

        total = pd.concat(frames)

        total['position_id'] = total.index
        total.views = total.views.astype(int)

        total['ind'] = total.position_id.astype(str) + total.prev_date.astype(str)

        total.set_index('ind', inplace=True)

        tmp = total.set_index(total.position_id.astype(str) + total.date.astype(str))[['total_prev_views']
        ].rename(columns={'total_prev_views': 'total_prev_views_' + str(n) + '_days_before'})

        views = pd.merge(total[['total_prev_views', 'views', 'date', 'prev_date', 'position_id']], tmp,
                         how='left', left_index=True, right_index=True)

        views['total_prev_views_' + str(n) + '_days_before'] = views[
            'total_prev_views_' + str(n) + '_days_before'].fillna(0).astype(int)

        views['last_' + str(n) + '_days_views'] = views['total_prev_views'] - views[
            'total_prev_views_' + str(n) + '_days_before']

        views['ind'] = views.position_id.astype(str) + views.date.astype(str)
        views = views.drop_duplicates('ind')
        views = views.set_index('ind')

        views['last_' + str(n) + '_days_views_div_total_prev_views'] = views['last_' + str(n) + '_days_views'] * 1. / views[
            'total_prev_views']

        views = views[[
            'views', 'total_prev_views', 'last_' + str(n) + '_days_views',
                                         'last_' + str(n) + '_days_views_div_total_prev_views']]

        self.df = pd.merge(self.df.set_index(self.df.position_id.astype(str) + self.df.date.astype(str)), views,
                      how='left', left_index=True, right_index=True)

        self.df.drop(['prev_date', 'views', 'total_prev_views'], 1, inplace=True)

        self.df['last_' + str(n) + '_days_views'] = self.df['last_' + str(n) + '_days_views'].fillna(0).astype(int)

        self.df['last_' + str(n) + '_days_views_div_total_prev_views'] = self.df[
            'last_' + str(n) + '_days_views_div_total_prev_views'].fillna(0)

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_date_of_max_views(self):
        logger = logging.getLogger(__name__ + ' : processor_date_of_max_views: {}'.format(self.key))
        logger.info('processor_date_of_max_views: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length_df = len(self.df)
        cols = self.df.columns

        views = postgres.get_data('select_views_of_positions.sql')

        views = views[views.entity_id.isin(self.df.position_id.astype(str))]
        views.date = pd.to_datetime(views.date)
        self.df.date = pd.to_datetime(self.df.date)

        total = pd.DataFrame(views.groupby(['entity_id', 'date'])['views'].sum()).reset_index()
        total.sort_values('date', inplace=True)

        days = set(list(total.date.unique()) + list(self.df[
                                                        (self.df.date > total.date.min()) &
                                                        (self.df.date < total.date.max())
                                                        ].date.unique()))

        frames = []
        for d in days:
            tmp = total[total.date <= d]

            length = len(tmp.drop_duplicates('entity_id'))

            tmp = pd.merge(
                pd.DataFrame(tmp.groupby(['entity_id'])['views'].sum()),
                tmp.sort_values('views', ascending=False).drop_duplicates(
                    'entity_id', keep='first').set_index('entity_id')[['date']],
                how='left', left_index=True, right_index=True
            )

            assert length == len(tmp), 'Length of dataset was changed after merge operation'

            tmp = tmp.reset_index().rename(
                columns={'entity_id': 'position_id', 'views': 'total_prev_views', 'date': 'date_of_max_views'})

            tmp = pd.merge(tmp.set_index('position_id'),
                           total[total.date == d][['entity_id', 'views']].set_index('entity_id'),
                           how='left', left_index=True, right_index=True).fillna(0)

            tmp['position_id'] = tmp.index

            tmp.insert(len(tmp.columns), column='date', value=[str(d)[:10]] * len(tmp))

            assert len(tmp) == len(tmp.drop_duplicates('position_id'))

            frames.append(tmp)

        total = pd.concat(frames)

        total['daily_views_div_total_prev_views'] = total.views * 100 / total.total_prev_views

        total['ind'] = total.position_id.astype(str) + total.date.astype(str)
        total = total[total.ind.isin(self.df.position_id.astype(str) + self.df.date.astype(str))]

        total.set_index('ind', inplace=True)
        total.views = total.views.astype(int)

        total.drop(['position_id', 'date'], 1, inplace=True)

        self.df = pd.merge(self.df.set_index(self.df.position_id.astype(str) + self.df.date.astype(str)), total,
                      how='left', left_index=True, right_index=True)

        self.df = df_proc.date_column_processing(self.df, date_column='date_of_max_views')

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length_df == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_date_of_starting_views_growth(self):
        logger = logging.getLogger(__name__ + ' : processor_date_of_starting_views_growth: {}'.format(self.key))
        logger.info('processor_date_of_starting_views_growth: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length_df = len(self.df)
        cols = self.df.columns

        views = postgres.get_data('select_views_of_positions.sql')

        views = views[views.entity_id.isin(self.df.position_id.astype(str))]
        views.date = pd.to_datetime(views.date)
        self.df.date = pd.to_datetime(self.df.date)

        total = pd.DataFrame(views.groupby(['entity_id', 'date'])['views'].sum()).reset_index()
        total.sort_values('date', inplace=True)

        total['number_of_week'] = total.date.map(lambda x: str(x.isocalendar()[0])) + total.date.map(
            lambda x: str(x.isocalendar()[1]) if x.isocalendar()[1] >= 10 else '0' + str(x.isocalendar()[1]))

        total['first_date_of_week'] = total.date - pd.to_timedelta(
            total.date.map(lambda x: x.isocalendar()[2]) - [1] * len(total), unit='d')

        weekly = pd.DataFrame(
            total.groupby(['entity_id', 'number_of_week', 'first_date_of_week']).views.sum()).reset_index()

        weekly['shift'] = weekly.groupby('entity_id').views.shift(periods=1)
        weekly.dropna(inplace=True)

        weekly['weekly_ratio'] = weekly.views / weekly['shift']

        days = set(list(total.date.unique()) + list(self.df[
                                                        (self.df.date > total.date.min()) &
                                                        (self.df.date < total.date.max())
                                                        ].date.unique()))

        frames = []
        for d in days:
            tmp = total[total.date <= d][['entity_id', 'views']]

            wk = weekly[weekly.first_date_of_week < d]
            wk = wk.sort_values('weekly_ratio', ascending=False).drop_duplicates(
                'entity_id', keep='first').set_index('entity_id')[['first_date_of_week', 'views', 'weekly_ratio']]
            wk = wk[wk.weekly_ratio >= 1].rename(columns={'views': 'total_prev_views'})

            length = len(tmp.drop_duplicates('entity_id'))

            tmp = pd.merge(
                pd.DataFrame(tmp.groupby(['entity_id'])['views'].sum()),
                wk,
                how='left', left_index=True, right_index=True
            ).rename(columns={'views': 'views_per_week'})

            tmp = pd.merge(tmp,
                           total[total.date == d][['entity_id', 'views']].set_index('entity_id'),
                           how='left', left_index=True, right_index=True).fillna(0)

            tmp['position_id'] = tmp.index

            assert length == len(tmp), 'Length of dataset was changed after merge operation'

            tmp.insert(len(tmp.columns), column='date', value=[str(d)[:10]] * len(tmp))

            assert len(tmp) == len(tmp.drop_duplicates('position_id'))

            frames.append(tmp)

        tmp = pd.concat(frames)
        tmp['ind'] = tmp.position_id.astype(str) + tmp.date.astype(str)
        tmp.set_index('ind', inplace=True)

        tmp.date = pd.to_datetime(tmp.date)
        tmp['first_date_of_week'] = tmp.date - pd.to_timedelta(
            tmp.date.map(lambda x: x.isocalendar()[2]) - [1] * len(tmp), unit='d')

        tmp['weekly_view_div_total_prev_views'] = tmp.views_per_week * 100 / (tmp.total_prev_views + tmp.views_per_week)

        tmp.drop(['position_id', 'date', 'views', 'total_prev_views'], 1, inplace=True)

        self.df = pd.merge(self.df.set_index(self.df.position_id.astype(str) + self.df.date.astype(str)), tmp,
                      how='left', left_index=True, right_index=True)

        self.df = self.df.rename(columns={'first_date_of_week':'date_of_starting_views_growth'})

        self.df = df_proc.date_column_processing(self.df, date_column='date_of_starting_views_growth')

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length_df == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_date_of_starting_version_growth(self):
        logger = logging.getLogger(__name__ + ' : processor_date_of_starting_version_growth: {}'.format(self.key))
        logger.info('processor_date_of_starting_version_growth: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()

        length_df = len(self.df)
        cols = self.df.columns

        vers = postgres.get_data('select_position_version_info.sql')

        self.df = self.df.rename(columns={'version': 'origin_start_version'})

        self.df.date = pd.to_datetime(self.df.date)
        vers.date = pd.to_datetime(vers.date)

        vers['number_of_week'] = vers.date.map(lambda x: str(x.isocalendar()[0])) + vers.date.map(
            lambda x: str(x.isocalendar()[1]) if x.isocalendar()[1] >= 10 else '0' + str(x.isocalendar()[1]))

        vers['first_date_of_week'] = vers.date - pd.to_timedelta(
            vers.date.map(lambda x: x.isocalendar()[2]) - [1] * len(vers), unit='d')

        weekly = pd.DataFrame(vers.groupby(['position_id', 'number_of_week', 'first_date_of_week']
                                           ).version.agg([np.min, np.max])).reset_index()

        weekly['version_growth'] = weekly.amax / weekly.amin

        days = set(list(self.df.date.unique()))

        frames = []
        for d in days:
            wk = weekly[weekly.first_date_of_week < d]
            wk = wk.sort_values('version_growth', ascending=False).drop_duplicates(
                'position_id', keep='first')[['position_id', 'first_date_of_week', 'amin', 'amax', 'version_growth']]
            wk = wk[wk.version_growth >= 1]

            wk = pd.merge(wk.set_index('position_id'),
                          vers[vers.date == d][['position_id', 'version']].set_index('position_id'),
                          how='left', left_index=True, right_index=True).fillna(0)

            wk.insert(len(wk.columns), column='date', value=[str(d)[:10]] * len(wk))

            frames.append(wk)

        tmp = pd.concat(frames)

        tmp['position_id'] = tmp.index
        tmp['ind'] = tmp.position_id.astype(str) + tmp.date.astype(str)
        tmp.set_index('ind', inplace=True)

        tmp.date = pd.to_datetime(tmp.date)
        tmp = tmp[tmp.date.isin(self.df.date)]

        tmp['first_date_of_week'] = tmp.date - pd.to_timedelta(
            tmp.date.map(lambda x: x.isocalendar()[2]) - [1] * len(tmp), unit='d')

        tmp.drop(['position_id', 'date'], 1, inplace=True)

        tmp['max_growth_per_week_div_current_version'] = (tmp.amax - tmp.amin) / tmp.version

        self.df = pd.merge(self.df.set_index(self.df.position_id.astype(str) + self.df.date.astype(str)), tmp,
                      how='left', left_index=True, right_index=True)

        self.df = self.df.rename(columns={'first_date_of_week': 'date_of_starting_version_growth'})

        self.df = df_proc.date_column_processing(self.df, date_column='date_of_starting_version_growth')

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length_df == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_crm_info(self):

        logger = logging.getLogger(__name__ + ' : processor_crm_info: {}'.format(self.key))
        logger.info('processor_crm_info: {}'.format(self.key))
        postgres = PostgreSQLBase()
        crm_proc = aux.CRMDataProcessor()
        df_proc = aux.DataFrameProcessor()

        length_df = len(self.df)
        cols = self.df.columns

        self.df['project_name'] = self.df['container_name']

        #crm data query
        crm_act = postgres.get_data('crm_aggs.sql',
                                    param=(tuple(self.df['date'].astype(str).unique()), 'actual'), param_type='tuple')
        crm_plan = postgres.get_data('crm_aggs.sql',
                                     param=(tuple(self.df['date'].astype(str).unique()), 'planned'), param_type='tuple')

        logger.info('CRM data loaded')

        # assert sorted(crm_plan.date.astype(str).unique()) == sorted(self.df.date.astype(str).unique()), 'Assertion!!! Not all snaps date are in the CRM planned data!'
        # assert sorted(crm_act.date.astype(str).unique()) == sorted(self.df.date.astype(str).unique()), 'Assertion!!! Not all snaps date are in the CRM actual data!'

        crm_plan[crm_proc.month_dict.values()] = crm_plan[crm_proc.month_dict.values()].fillna(0)
        crm_act[crm_proc.month_dict.values()] = crm_act[crm_proc.month_dict.values()].fillna(0)


        #positions data query
        pos = postgres.get_data('position_aggs.sql', tuple(self.df['date'].astype(str).unique()), param_type='days')
        pos = df_proc.datetime_cols(pos, ['workload_start_date'])
        pos['month'] = pos.workload_start_date.map(lambda d: crm_proc.month_dict[d.month])
        pos['year'] = pos.workload_start_date.map(lambda d: d.year)

        pos = pd.pivot_table(pos, values='sum_fte', index=['gbu_id', 'customer_name', 'project_name', 'year', 'date'],
                            columns='month', aggfunc=np.sum, fill_value=0).reset_index()

        for aggs_type in ['previous', 'current', 'next']:
            for aggs_range in ['month', 'quarter', 'half_year', 'year']:
                for base in ['project', 'customer', 'gbu']:
                    for value in ['revenue', 'positions']:
                        for scenario in ['Standard', 'Optimistic', 'Pessimistic', 'Weighted']:

                            logger.info(scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value)

                            if value == 'revenue':

                                if aggs_type != 'previous':
                                    tmp = crm_proc.crm_data_groupper(crm_plan, scenario=scenario, aggs_type=aggs_type, aggs_range=aggs_range, base=base, value=value)

                                else:
                                    tmp = crm_proc.crm_data_groupper(crm_act, scenario=scenario, aggs_type=aggs_type, aggs_range=aggs_range, base=base, value=value)

                            else:
                                tmp = crm_proc.crm_data_groupper(pos, scenario=scenario, aggs_type=aggs_type, aggs_range=aggs_range, base=base, value=value)

                            self.df = pd.merge(
                                self.df, tmp.set_index(tmp[crm_proc.base_dict[base]] + tmp.date.astype(str))[
                                    [scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value]],
                                how='left', left_on=self.df[crm_proc.base_dict[base]] + self.df.date.astype(str), right_index=True
                            )

                            self.df[[scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value]] = self.df[
                                [scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value]].fillna(0)


        for base in ['project', 'customer', 'gbu']:
            for scenario in ['Standard', 'Optimistic', 'Pessimistic', 'Weighted']:
                self.df = crm_proc.previous_actual_revenue_correction(self.df, crm_act=crm_act, crm_plan=crm_plan, scenario=scenario, base=base)

        logger.info('run rates calculations started')

        for aggs_type in ['previous', 'current', 'next']:
            for aggs_range in ['month', 'quarter', 'half_year', 'year']:
                for base in ['project', 'customer', 'gbu']:
                    for scenario in ['Standard', 'Optimistic', 'Pessimistic', 'Weighted']:
                        self.df['avg_run_rate_' + scenario + '_' + aggs_type + '_' + aggs_range + '_' + base] = self.df[scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_revenue'] / \
                                                                                                                self.df[scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_positions']

                        self.df[['avg_run_rate_' + scenario + '_' + aggs_type + '_' + aggs_range + '_' + base]] = self.df[
                            ['avg_run_rate_' + scenario + '_' + aggs_type + '_' + aggs_range + '_' + base]
                        ].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

        logger.info('time dinamics calculations started')

        for aggs_range in ['month', 'quarter', 'half_year', 'year']:
            for base in ['project', 'customer', 'gbu']:
                for value in ['revenue', 'positions']:
                    for scenario in ['Standard', 'Optimistic', 'Pessimistic', 'Weighted']:
                        self.df['ratio_current_previous_' + scenario + '_' + value + '_' + aggs_range + '_' + base] = self.df[
                                                                                              scenario + '_' + 'current_' + aggs_range + '_' + base + '_' + value] / \
                                                                                                                      self.df[
                                                                                              scenario + '_' + 'previous_' + aggs_range + '_' + base + '_' + value]

                        self.df[['ratio_current_previous_' + scenario + '_' + value + '_' + aggs_range + '_' + base]] = self.df[
                            ['ratio_current_previous_' + scenario + '_' + value + '_' + aggs_range + '_' + base]
                        ].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

                        self.df['ratio_next_current_' + scenario + '_' + value + '_' + aggs_range + '_' + base] = self.df[
                                                                                              scenario + '_' + 'next_' + aggs_range + '_' + base + '_' + value] / \
                                                                                                                  self.df[
                                                                                              scenario + '_' + 'current_' + aggs_range + '_' + base + '_' + value]

                        self.df[['ratio_next_current_' + scenario + '_' + value + '_' + aggs_range + '_' + base]] = self.df[
                            ['ratio_next_current_' + scenario + '_' + value + '_' + aggs_range + '_' + base]
                        ].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

        self.df.date = self.df.date.dt.strftime('%Y-%m-%d')

        assert length_df == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))


    def processor_business_days(self):

        logger = logging.getLogger(__name__ + ' : processor_business_days: {}'.format(self.key))
        logger.info('processor_business_days: {}'.format(self.key))
        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()
        loc_ids = aux.City_Country()

        length_df = len(self.df)
        cols = self.df.columns

        countries_dict = loc_ids.id_city_country.drop_duplicates('country_id').set_index('country_id').country.to_dict()

        bad_calendars = {
            'Canada': 'Canada, Calgary',
            'Switzerland': 'Switzerland, Zurich',
            'India': 'India, Hyderabad',
            'Spain': 'Spain, Malaga',
            'Australia': 'Australia, New South Wales',
            'Hong Kong SAR': 'Hong Kong',
            'China- Not Active': 'China'
        }

        day_start = datetime.datetime.strptime('2014-01-01', '%Y-%m-%d')
        day_end = datetime.datetime.today()

        days = pd.date_range(day_start, day_end)
        calendar = pd.DataFrame({'day': days})
        calendar['day'] = calendar['day'].dt.date

        for c in sorted(countries_dict.values()):

            logger.info(c)
            country_id = [k for k, v in countries_dict.items() if v == c][0]

            if c in ['Israel', 'Portugal', 'Viet Nam']:  # no calendar updates for this countries
                c_cal = pd.DataFrame({'day': pd.bdate_range(start=day_start, end=day_end)})
                c_cal[country_id] = 1

            else:

                if c in bad_calendars.keys():
                    cal_id_df = postgres.get_data('get_country_calendar_id.sql', param=bad_calendars[c],
                                                  param_type='day')

                else:
                    cal_id_df = postgres.get_data('get_country_calendar_id.sql', param=c, param_type='day')

                try:
                    c_id = cal_id_df['calendar_id'][0]
                except:
                    logger.debug('problem with country calendar for {}'.format(c))

                c_cal = postgres.get_data('business_days_by_country_id.sql', param=c_id, param_type='day')
                c_cal[country_id] = 1
                c_cal = c_cal.drop_duplicates()

            c_cal['day'] = pd.to_datetime(c_cal['day']).dt.date

            calendar = pd.merge(calendar, c_cal,
                                on='day',
                                how='left')

            calendar = calendar.fillna(0)
            calendar[country_id] = calendar[country_id].astype(int)

        calendar['day'] = calendar['day'].astype(str)
        calendar = calendar.set_index('day')

        def bd_range_processing(x, start_date_field, end_date_field, calendar, bd_type='min'):

            if bd_type == 'none':
                return len(len(pd.bdate_range(start=x[start_date_field], end=x[end_date_field])))

            else:

                if bd_type == 'min':
                    return np.min(
                        calendar.loc[x[start_date_field]:x[end_date_field], x['staffing_locations_country_ids']].sum(
                            axis=0))
                elif bd_type == 'max':
                    return np.max(
                        calendar.loc[x[start_date_field]:x[end_date_field], x['staffing_locations_country_ids']].sum(
                            axis=0))
                elif bd_type == 'mean':
                    return np.mean(
                        calendar.loc[x[start_date_field]:x[end_date_field], x['staffing_locations_country_ids']].sum(
                            axis=0))

        self.df = df_proc.datetime_cols(self.df, ['date', 'timesheet_date'])

        self.df['eom_date_cur_month'] = self.df.date.map(
            lambda d: datetime.datetime.strftime(d + relativedelta(day=31), '%Y-%m-%d'))
        self.df['eom_date_next_1_month'] = self.df.date.map(
            lambda d: datetime.datetime.strftime(d + relativedelta(months=1) + relativedelta(day=31), '%Y-%m-%d'))
        self.df['eom_date_next_2_month'] = self.df.date.map(
            lambda d: datetime.datetime.strftime(d + relativedelta(months=2) + relativedelta(day=31), '%Y-%m-%d'))

        self.df[['date', 'created', 'planned_start_date', 'planned_end_date', 'workload_start_date', 'last_assigned_date',
            'proj_start_date', 'proj_end_date', 'timesheet_date', 'first_proposals_date', 'first_booking_date',
            'first_onboarding_date', 'last_proposals_date', 'last_booking_date', 'last_onboarding_date',
            'eom_date_cur_month', 'eom_date_next_1_month', 'eom_date_next_2_month']
        ] = self.df[
            ['date', 'created', 'planned_start_date', 'planned_end_date', 'workload_start_date', 'last_assigned_date',
             'proj_start_date', 'proj_end_date', 'timesheet_date', 'first_proposals_date', 'first_booking_date',
             'first_onboarding_date', 'last_proposals_date', 'last_booking_date', 'last_onboarding_date',
             'eom_date_cur_month', 'eom_date_next_1_month', 'eom_date_next_2_month']].astype(str)

        self.df['distance_bd_between_snap_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                               bd_range_processing(x,
                                                                                                   start_date_field='date',
                                                                                                   end_date_field='workload_start_date',
                                                                                                   calendar=calendar,
                                                                                                   bd_type='min'),
                                                                               axis=1
                                                                               )
        logger.debug('{} column processed'.format('distance_bd_between_snap_date_and_workload_start_date'))

        self.df['distance_bd_between_snap_date_and_eom_date_cur_month'] = self.df.apply(lambda x:
                                                                              bd_range_processing(x,
                                                                                                  start_date_field='date',
                                                                                                  end_date_field='eom_date_cur_month',
                                                                                                  calendar=calendar,
                                                                                                  bd_type='min'), axis=1
                                                                              )
        logger.debug('{} column processed'.format('distance_bd_between_snap_date_and_eom_date_cur_month'))

        self.df['distance_bd_between_snap_date_and_eom_date_next_1_month'] = self.df.apply(lambda x:
                                                                                 bd_range_processing(x,
                                                                                                     start_date_field='date',
                                                                                                     end_date_field='eom_date_next_1_month',
                                                                                                     calendar=calendar,
                                                                                                     bd_type='min'),
                                                                                 axis=1
                                                                                 )
        logger.debug('{} column processed'.format('distance_bd_between_snap_date_and_eom_date_next_1_month'))

        self.df['distance_bd_between_snap_date_and_eom_date_next_2_month'] = self.df.apply(lambda x:
                                                                                 bd_range_processing(x,
                                                                                                     start_date_field='date',
                                                                                                     end_date_field='eom_date_next_2_month',
                                                                                                     calendar=calendar,
                                                                                                     bd_type='min'),
                                                                                 axis=1
                                                                                 )
        logger.debug('{} column processed'.format('distance_bd_between_snap_date_and_eom_date_next_2_month'))

        self.df['distance_bd_between_snap_date_and_planned_start_date'] = self.df.apply(lambda x:
                                                                              bd_range_processing(x,
                                                                                                  start_date_field='date',
                                                                                                  end_date_field='planned_start_date',
                                                                                                  calendar=calendar,
                                                                                                  bd_type='min'), axis=1
                                                                              )
        logger.debug('{} column processed'.format('distance_bd_between_snap_date_and_planned_start_date'))

        self.df['distance_bd_between_created_and_snap_date'] = self.df.apply(lambda x:
                                                                   bd_range_processing(x, start_date_field='created',
                                                                                       end_date_field='date',
                                                                                       calendar=calendar,
                                                                                       bd_type='min'), axis=1
                                                                   )
        logger.debug('{} column processed'.format('distance_bd_between_created_and_snap_date'))

        self.df['distance_bd_between_created_and_workload_start_date'] = self.df.apply(lambda x:
                                                                             bd_range_processing(x,
                                                                                                 start_date_field='created',
                                                                                                 end_date_field='workload_start_date',
                                                                                                 calendar=calendar,
                                                                                                 bd_type='min'), axis=1
                                                                             )
        logger.debug('{} column processed'.format('distance_bd_between_created_and_workload_start_date'))

        self.df['distance_bd_between_workload_start_date_and_planned_end_date'] = self.df.apply(lambda x:
                                                                                      bd_range_processing(x,
                                                                                                          start_date_field='workload_start_date',
                                                                                                          end_date_field='planned_end_date',
                                                                                                          calendar=calendar,
                                                                                                          bd_type='min'),
                                                                                      axis=1
                                                                                      )
        logger.debug('{} column processed'.format('distance_bd_between_workload_start_date_and_planned_end_date'))

        self.df['distance_bd_between_created_and_planned_start_date'] = self.df.apply(lambda x:
                                                                            bd_range_processing(x,
                                                                                                start_date_field='created',
                                                                                                end_date_field='planned_start_date',
                                                                                                calendar=calendar,
                                                                                                bd_type='min'), axis=1
                                                                            )
        logger.debug('{} column processed'.format('distance_bd_between_created_and_planned_start_date'))

        self.df['distance_bd_between_planned_start_date_and_planned_end_date'] = self.df.apply(lambda x:
                                                                                     bd_range_processing(x,
                                                                                                         start_date_field='planned_start_date',
                                                                                                         end_date_field='planned_end_date',
                                                                                                         calendar=calendar,
                                                                                                         bd_type='min'),
                                                                                     axis=1
                                                                                     )
        logger.debug('{} column processed'.format('distance_bd_between_planned_start_date_and_planned_end_date'))

        self.df['distance_bd_between_snap_date_and_planned_end_date'] = self.df.apply(lambda x:
                                                                            bd_range_processing(x,
                                                                                                start_date_field='date',
                                                                                                end_date_field='planned_end_date',
                                                                                                calendar=calendar,
                                                                                                bd_type='min'), axis=1
                                                                            )
        logger.debug('{} column processed'.format('distance_bd_between_snap_date_and_planned_end_date'))

        self.df['distance_bd_between_proj_start_date_and_proj_end_date'] = self.df.apply(lambda x:
                                                                               bd_range_processing(x,
                                                                                                   start_date_field='proj_start_date',
                                                                                                   end_date_field='proj_end_date',
                                                                                                   calendar=calendar,
                                                                                                   bd_type='min'),
                                                                               axis=1
                                                                               )
        logger.debug('{} column processed'.format('distance_bd_between_proj_start_date_and_proj_end_date'))

        self.df['distance_bd_between_proj_start_date_and_created'] = self.df.apply(lambda x:
                                                                         bd_range_processing(x,
                                                                                             start_date_field='proj_start_date',
                                                                                             end_date_field='created',
                                                                                             calendar=calendar,
                                                                                             bd_type='min'), axis=1
                                                                         )
        logger.debug('{} column processed'.format('distance_bd_between_proj_start_date_and_created'))

        self.df['distance_bd_between_created_and_proj_end_date'] = self.df.apply(lambda x:
                                                                       bd_range_processing(x,
                                                                                           start_date_field='created',
                                                                                           end_date_field='proj_end_date',
                                                                                           calendar=calendar,
                                                                                           bd_type='min'), axis=1
                                                                       )
        logger.debug('{} column processed'.format('distance_bd_between_created_and_proj_end_date'))

        self.df['distance_bd_between_proj_start_date_and_planned_start_date'] = self.df.apply(lambda x:
                                                                                    bd_range_processing(x,
                                                                                                        start_date_field='proj_start_date',
                                                                                                        end_date_field='planned_start_date',
                                                                                                        calendar=calendar,
                                                                                                        bd_type='min'),
                                                                                    axis=1
                                                                                    )
        logger.debug('{} column processed'.format('distance_bd_between_proj_start_date_and_planned_start_date'))

        self.df['distance_bd_between_planned_start_date_and_proj_end_date'] = self.df.apply(lambda x:
                                                                                  bd_range_processing(x,
                                                                                                      start_date_field='planned_start_date',
                                                                                                      end_date_field='proj_end_date',
                                                                                                      calendar=calendar,
                                                                                                      bd_type='min'),
                                                                                  axis=1
                                                                                  )
        logger.debug('{} column processed'.format('distance_bd_between_planned_start_date_and_proj_end_date'))

        self.df['distance_bd_between_first_proposals_date_and_date'] = self.df.apply(lambda x:
                                                                           bd_range_processing(x,
                                                                                               start_date_field='first_proposals_date',
                                                                                               end_date_field='date',
                                                                                               calendar=calendar,
                                                                                               bd_type='min'), axis=1
                                                                           )
        logger.debug('{} column processed'.format('distance_bd_between_first_proposals_date_and_date'))

        self.df['distance_bd_between_first_booking_date_and_date'] = self.df.apply(lambda x:
                                                                         bd_range_processing(x,
                                                                                             start_date_field='first_booking_date',
                                                                                             end_date_field='date',
                                                                                             calendar=calendar,
                                                                                             bd_type='min'), axis=1
                                                                         )
        logger.debug('{} column processed'.format('distance_bd_between_first_booking_date_and_date'))

        self.df['distance_bd_between_first_onboarding_date_and_date'] = self.df.apply(lambda x:
                                                                            bd_range_processing(x,
                                                                                                start_date_field='first_onboarding_date',
                                                                                                end_date_field='date',
                                                                                                calendar=calendar,
                                                                                                bd_type='min'), axis=1
                                                                            )
        logger.debug('{} column processed'.format('distance_bd_between_first_onboarding_date_and_date'))

        self.df['distance_bd_between_last_proposals_date_and_date'] = self.df.apply(lambda x:
                                                                          bd_range_processing(x,
                                                                                              start_date_field='last_proposals_date',
                                                                                              end_date_field='date',
                                                                                              calendar=calendar,
                                                                                              bd_type='min'), axis=1
                                                                          )
        logger.debug('{} column processed'.format('distance_bd_between_last_proposals_date_and_date'))

        self.df['distance_bd_between_last_booking_date_and_date'] = self.df.apply(lambda x:
                                                                        bd_range_processing(x,
                                                                                            start_date_field='last_booking_date',
                                                                                            end_date_field='date',
                                                                                            calendar=calendar,
                                                                                            bd_type='min'), axis=1
                                                                        )
        logger.debug('{} column processed'.format('distance_bd_between_last_booking_date_and_date'))

        self.df['distance_bd_between_last_onboarding_date_and_date'] = self.df.apply(lambda x:
                                                                           bd_range_processing(x,
                                                                                               start_date_field='last_onboarding_date',
                                                                                               end_date_field='date',
                                                                                               calendar=calendar,
                                                                                               bd_type='min'), axis=1
                                                                           )
        logger.debug('{} column processed'.format('distance_bd_between_last_onboarding_date_and_date'))

        self.df['distance_bd_between_last_onboarding_date_and_date'] = self.df.apply(lambda x:
                                                                           bd_range_processing(x,
                                                                                               start_date_field='last_onboarding_date',
                                                                                               end_date_field='date',
                                                                                               calendar=calendar,
                                                                                               bd_type='min'), axis=1
                                                                           )
        logger.debug('{} column processed'.format('distance_bd_between_last_onboarding_date_and_date'))

        self.df['distance_bd_between_first_proposals_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                          bd_range_processing(x,
                                                                                                              start_date_field='first_proposals_date',
                                                                                                              end_date_field='workload_start_date',
                                                                                                              calendar=calendar,
                                                                                                              bd_type='min'),
                                                                                          axis=1
                                                                                          )
        logger.debug('{} column processed'.format('distance_bd_between_first_proposals_date_and_workload_start_date'))

        self.df['distance_bd_between_first_booking_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                        bd_range_processing(x,
                                                                                                            start_date_field='first_booking_date',
                                                                                                            end_date_field='workload_start_date',
                                                                                                            calendar=calendar,
                                                                                                            bd_type='min'),
                                                                                        axis=1
                                                                                        )
        logger.debug('{} column processed'.format('distance_bd_between_first_booking_date_and_workload_start_date'))

        self.df['distance_bd_between_first_onboarding_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                           bd_range_processing(x,
                                                                                                               start_date_field='first_onboarding_date',
                                                                                                               end_date_field='workload_start_date',
                                                                                                               calendar=calendar,
                                                                                                               bd_type='min'),
                                                                                           axis=1
                                                                                           )
        logger.debug('{} column processed'.format('distance_bd_between_first_onboarding_date_and_workload_start_date'))

        self.df['distance_bd_between_last_proposals_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                         bd_range_processing(x,
                                                                                                             start_date_field='last_proposals_date',
                                                                                                             end_date_field='workload_start_date',
                                                                                                             calendar=calendar,
                                                                                                             bd_type='min'),
                                                                                         axis=1
                                                                                         )
        logger.debug('{} column processed'.format('distance_bd_between_last_proposals_date_and_workload_start_date'))

        self.df['distance_bd_between_last_booking_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                       bd_range_processing(x,
                                                                                                           start_date_field='last_booking_date',
                                                                                                           end_date_field='workload_start_date',
                                                                                                           calendar=calendar,
                                                                                                           bd_type='min'),
                                                                                       axis=1
                                                                                       )
        logger.debug('{} column processed'.format('distance_bd_between_last_booking_date_and_workload_start_date'))

        self.df['distance_bd_between_last_onboarding_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                          bd_range_processing(x,
                                                                                                              start_date_field='last_onboarding_date',
                                                                                                              end_date_field='workload_start_date',
                                                                                                              calendar=calendar,
                                                                                                              bd_type='min'),
                                                                                          axis=1
                                                                                          )
        logger.debug('{} column processed'.format('distance_bd_between_last_onboarding_date_and_workload_start_date'))

        self.df['distance_bd_between_last_onboarding_date_and_workload_start_date'] = self.df.apply(lambda x:
                                                                                          bd_range_processing(x,
                                                                                                              start_date_field='last_onboarding_date',
                                                                                                              end_date_field='workload_start_date',
                                                                                                              calendar=calendar,
                                                                                                              bd_type='min'),
                                                                                          axis=1
                                                                                          )
        logger.debug('{} column processed'.format('distance_bd_between_last_onboarding_date_and_workload_start_date'))

        self.df['distance_bd_between_workload_start_date_and_eom_date_cur_month'] = self.df.apply(lambda x:
                                                                                        bd_range_processing(x,
                                                                                                            start_date_field='workload_start_date',
                                                                                                            end_date_field='eom_date_cur_month',
                                                                                                            calendar=calendar,
                                                                                                            bd_type='min'),
                                                                                        axis=1
                                                                                        )
        logger.debug('{} column processed'.format('distance_bd_between_workload_start_date_and_eom_date_cur_month'))

        self.df['distance_bd_between_workload_start_date_and_eom_date_next_1_month'] = self.df.apply(lambda x:
                                                                                           bd_range_processing(x,
                                                                                                               start_date_field='workload_start_date',
                                                                                                               end_date_field='eom_date_next_1_month',
                                                                                                               calendar=calendar,
                                                                                                               bd_type='min'),
                                                                                           axis=1
                                                                                           )
        logger.debug('{} column processed'.format('distance_bd_between_workload_start_date_and_eom_date_next_1_month'))

        self.df['distance_bd_between_workload_start_date_and_eom_date_next_2_month'] = self.df.apply(lambda x:
                                                                                           bd_range_processing(x,
                                                                                                               start_date_field='workload_start_date',
                                                                                                               end_date_field='eom_date_next_2_month',
                                                                                                               calendar=calendar,
                                                                                                               bd_type='min'),
                                                                                           axis=1
                                                                                           )
        logger.debug('{} column processed'.format('distance_bd_between_workload_start_date_and_eom_date_next_2_month'))

        self.df = self.df.drop(['eom_date_cur_month', 'eom_date_next_1_month', 'eom_date_next_2_month'], 1)

        assert length_df == len(self.df), 'Assertion!!! The dataframe length was changed after merge operation!'
        logger.info(sorted(set(self.df.columns) - set(cols)))