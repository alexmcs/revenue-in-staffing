import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import sys
sys.path.append('../src')
from datahub import PostgreSQLBase, ToDatahubWriter
import logging

import auxiliary as aux
reload (aux)

logger = logging.getLogger('Revenue_from_utlz')

class RevFromUtlz():

    def __init__(self, date = 'none'):

        if date == 'none':
            date = datetime.date.today()
        else:
            date = datetime.datetime.strptime(date, '%Y-%m-%d')

        cur_period = date.replace(day=1)
        next_period = cur_period + relativedelta(months=1)
        next_period = next_period.replace(day=1)

        self.date = datetime.datetime.strftime(date, '%Y-%m-%d')
        self.cur_period = datetime.datetime.strftime(cur_period, '%Y-%m-%d')
        self.next_period = datetime.datetime.strftime(next_period, '%Y-%m-%d')


    def calculations(self, period):

        logger.info('calculations for date={} and period={} were started'.format(self.date, period))

        postgres = PostgreSQLBase()
        df_proc = aux.DataFrameProcessor()
        to_datahub = ToDatahubWriter('revenue_from_utilization')


        params = (self.date, period)  # (run_date, period_of_predictions)
        df = postgres.get_data('utilization_per_employees_forecast.sql', param=params, param_type='tuple')
        df.employee_id = df.employee_id.astype(str)

        # df1 - employees with assigned Billable positions
        # df2 - employees with NB:Obligation positions
        # df3 - employees without assigned positions and don't have an external workload

        df1 = df[
            df.position_id.notnull()
        ]

        df2 = df[
            (df.position_id.isnull()) &
            (df.ext_wl != 0)
        ]

        df3 = df[
            (df.position_id.isnull()) &
            (df.ext_wl.fillna(0) == 0)
        ]

        params = (period, tuple(df2.employee_id.astype(str).unique()))
        tmp = postgres.get_data('rev_from_utlz_NB_positions.sql', param=params, param_type='tuple')

        df2 = pd.merge(
            df2[['employee_id', 'predicted_hours', 'ext_wl']], tmp.set_index('assignee_id'), how='inner',
            left_on=df2['employee_id'].astype(str), right_index=True
        )[['employee_id', 'predicted_hours', 'ext_wl', 'position_id', 'monthly_workload']]


        logger.info('active positions importance sorting')

        df = postgres.get_data('positions_importance_sorting.sql', param=period, param_type='day')

        df.insert(loc=len(df.columns), column='date', value=df.shape[0] * [self.date])

        df = df_proc.datetime_cols(df, ['created', 'planned_start_date', 'planned_end_date', 'date'])
        df = df_proc.time_distance(df, 'planned_start_date', 'planned_end_date', 'duration')
        df = df_proc.time_distance(df, 'created', 'date', 'distance_from_created')
        df = df_proc.time_distance(df, 'date', 'planned_start_date', 'time_to_planned_start_date')

        def discounting(duration, time_to_planned_start_date, alpha=0.12, alpha_overdue=0.24):

            discounted_sum = 0

            if time_to_planned_start_date >= 0:
                for i in xrange(time_to_planned_start_date, time_to_planned_start_date + duration):
                    discounted_sum += 1 / ((1 + (alpha / 365)) ** i)

            else:
                for i in xrange(time_to_planned_start_date, 0):
                    discounted_sum += 1 / ((1 + (alpha_overdue / 365)) ** i)

                for i in xrange(0, time_to_planned_start_date + duration):
                    discounted_sum += 1 / ((1 + (alpha / 365)) ** i)

            return discounted_sum

        def coeffitioning(discounted, number_of_positions_per_container, number_of_positions_per_customer,
                          distance_from_created):

            def coef_cont(number_of_positions_per_container):
                if (number_of_positions_per_container <= 20):
                    return 0.8
                elif (number_of_positions_per_container <= 50):
                    return 1.
                elif (number_of_positions_per_container <= 100):
                    return 1.2
                elif (number_of_positions_per_container <= 200):
                    return 1.5
                else:
                    return 2.

            def coef_cust(number_of_positions_per_customer):
                if (number_of_positions_per_customer <= 50):
                    return 0.8
                elif (number_of_positions_per_customer <= 100):
                    return 1.
                elif (number_of_positions_per_customer <= 250):
                    return 1.2
                elif (number_of_positions_per_customer <= 500):
                    return 1.5
                elif (number_of_positions_per_customer <= 1000):
                    return 2.
                else:
                    return 3

            def coef_created_dist(distance_from_created):
                if (distance_from_created <= 50):
                    return 2.
                elif (distance_from_created <= 100):
                    return 1.5
                elif (distance_from_created <= 150):
                    return 1
                elif (distance_from_created <= 365):
                    return 0.5
                else:
                    return 0.2

            score = discounted \
                    * coef_cont(number_of_positions_per_container) \
                    * coef_cust(number_of_positions_per_customer) \
                    * coef_created_dist(distance_from_created)

            return score

        df['score'] = df.apply(
            lambda row: coeffitioning(discounting(row['duration'], row['time_to_planned_start_date']),
                                      row['number_of_positions_per_container'],
                                      row['number_of_positions_per_customer'],
                                      row['distance_from_created']), axis=1)

        urg = postgres.get_data('urgency.sql', param=self.date, param_type='day')

        df.position_id = df.position_id.astype(str)
        urg.position_id = urg.position_id.astype(str)

        df = pd.merge(df, urg.set_index('position_id'), how='left', left_on='position_id', right_index=True)

        df.weight_adjusted = df.weight_adjusted.fillna(1)
        df['position_importance'] = df.score * df.weight_adjusted

        match = postgres.get_data('employees_and_positions_matching.sql', tuple(df3.employee_id.unique()),
                                  param_type='day')

        match.position_id = match.position_id.astype(str)

        df = pd.merge(
            match, df.set_index('position_id')[['position_importance']], how='left',
            left_on='position_id', right_index=True
        )

        df = df[df.position_importance.notnull()]

        df.employee_id = df.employee_id.astype(str)
        df.position_id = df.position_id.astype(str)

        logger.info('matching script started')

        result = []
        iteration = True
        i = 0
        while iteration:

            df = df.sort_values(['position_importance', 'score'], ascending=False)

            result.append(pd.DataFrame(df.iloc[0]).T)

            df = df[
                (df.position_id != df.iloc[0]['position_id']) &
                (df.employee_id != df.iloc[0]['employee_id'])
                ]

            if len(df.position_id.unique()) == 0:
                iteration = False

            i = i + 1
            logger.debug(i)

        res = pd.concat(result)

        df3 = pd.merge(
            df3[['employee_id', 'predicted_hours', 'ext_wl', 'monthly_workload']],
            res.set_index('employee_id')[['position_id']],
            how='left', left_on='employee_id', right_index=True
        )

        df = pd.concat([df1, df2, df3])

        logger.info('starting write results for date = {} and period = {} into datalake'.format(self.date, period))

        df.insert(loc=len(df.columns), column='date', value=df.shape[0] * [self.date])
        df.insert(loc=len(df.columns), column='period', value=df.shape[0] * [period])

        df = df[df.position_id.notnull()]

        df = df.set_index('position_id')

        to_datahub.write_info(df)

        logger.info('predictions for date = {} and period = {} are written into datalake successfully'.format(self.date, period))


    def make_predictions(self):

        self.calculations(self.cur_period)
        self.calculations(self.next_period)

        logger.info('all predictions were made successfully')

