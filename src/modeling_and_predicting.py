import logging
from datahub import PostgreSQLBase
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from os import listdir
import os
import sys
sys.path.append('../src')
import features
reload(features)
import variable_selection
reload(variable_selection)
import data_splitting
reload(data_splitting)
import pipeline_classes
reload(pipeline_classes)
import model_metrics_prod
reload(model_metrics_prod)
import revenue
reload (revenue)
import termination
reload(termination)


def is_billable_retrospective_models_training(min_date = '2017-12-31', key = 'is_billable'):

    logger = logging.getLogger(__name__ + ' : is_billable_retrospective_models_training')
    logger.info('is_billable_retrospective_models_training')

    min_date_dt = datetime.datetime.strptime(min_date, '%Y-%m-%d')

    days = []
    while min_date_dt < datetime.datetime.today():

        while min_date_dt.month == (min_date_dt + relativedelta(days=1)).month:
            min_date_dt = min_date_dt + relativedelta(days=1)

        days.append(datetime.datetime.strftime(min_date_dt, '%Y-%m-%d'))
        min_date_dt = min_date_dt + relativedelta(months=1)

    models = []
    for f in listdir('../data/'):
        if '{}_model_'.format(key) in f:
            models.append(f)

    df = pd.DataFrame(models, columns=['files'])
    df['date'] = df['files'].map(lambda x: x.split('.')[0][-10:]).astype(str)

    missing_models = []
    for d in days:
        if d not in list(df.date):
            missing_models.append(d)

    for d in missing_models:

        logger = logging.getLogger(__name__ + ' : is_billable_model_training')
        logger.info('{}_model_training per date={}'.format(key, d))

        tm = termination.Termination(job='model_training', date=d)
        tm.model_training()

        logger.info('{} is billable models per date {} trained successfully and dumped to pickle file'.format(key, d))



def revenue_in_staffing_retrospective_models_training(min_date = '2017-12-31', key = 'revenue'):

    logger = logging.getLogger(__name__ + ' : revenue_in_staffing_retrospective_models_training')
    logger.info('revenue_in_staffing_retrospective_models_training')

    min_date_dt = datetime.datetime.strptime(min_date, '%Y-%m-%d')

    days = []
    while min_date_dt < datetime.datetime.today():

        while min_date_dt.month == (min_date_dt + relativedelta(days=1)).month:
            min_date_dt = min_date_dt + relativedelta(days=1)

        days.append(datetime.datetime.strftime(min_date_dt, '%Y-%m-%d'))
        min_date_dt = min_date_dt + relativedelta(months=1)

    models = []
    for f in listdir('../data/'):
        if '{}_model_'.format(key) in f:
            models.append(f)

    df = pd.DataFrame(models, columns=['files'])
    df['date'] = df['files'].map(lambda x: x.split('.')[0][-10:]).astype(str)

    missing_models = []
    for d in days:
        if d not in list(df.date):
            missing_models.append(d)

    for d in missing_models:

        logger = logging.getLogger(__name__ + ' : revenue_in_staffing_model_training')
        logger.info('{}_in_staffing_model_training per date={}'.format(key, d))

        rv = revenue.RevenueInStaffing(job='model_training', date=d)
        rv.model_training()

        logger.info('{} in staffing model per date {} trained successfully and dumped to pickle file'.format(key, d))


def revenue_in_staffing_retrospective_predictions(key = 'revenue'):

    logger = logging.getLogger(__name__ + ' : revenue_in_staffing_retrospective_predictions')
    logger.info('revenue_in_staffing_retrospective_predictions')
    postgres = PostgreSQLBase()

    def list_of_dates(x):
        lst = []
        d = datetime.datetime.strptime(x['date'], '%Y-%m-%d')

        while d < datetime.datetime.strptime(x['next_date'], '%Y-%m-%d'):
            lst.append(datetime.datetime.strftime(d, '%Y-%m-%d'))
            d = d + relativedelta(days=1)

        return lst

    def query_list_of_dates(x):

        try:
            params = (key, x['date'], x['next_date'])
            df = postgres.get_data('days_list_for_predictions.sql', param=params, param_type='tuple')

            return list(df.date)

        except AttributeError:
            return []

    def dates_without_predictions(x):

        return list(set(x['list_of_dates']) - set(x['dates_with_predictions']))

    files = []

    for f in listdir('../data/'):
        if '{}_model_'.format(key) in f:
            files.append(f)

    df = pd.DataFrame(files, columns=['files'])
    df['date'] = df['files'].map(lambda x: x.split('.')[0][-10:]).astype(str)

    df = df.sort_values('date')
    df['next_date'] = df.date.shift(-1).fillna(datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d'))
    df['list_of_dates'] = df.apply(list_of_dates, axis=1)
    df['dates_with_predictions'] = df.apply(query_list_of_dates, axis=1)
    df['dates_without_predictions'] = df.apply(dates_without_predictions, axis=1)
    df['list_len'] = df['dates_without_predictions'].map(lambda x: len(x))
    df = df[df['list_len'] != 0]
    df = df.drop(['date', 'next_date', 'list_of_dates', 'dates_with_predictions'], 1)

    for ind in df[df.list_len > 1].index:

        model_file = df.at[ind, 'files']
        dates = df.at[ind, 'dates_without_predictions']

        rv = revenue.RevenueInStaffing(job='making_predictions', date=dates)
        rv.making_predictions(model_file=model_file)

    if df[df.list_len == 1].shape[0] >=1:

        for ind in df[df.list_len == 1].index:

            model_file = df.at[ind, 'files']
            dates = df.at[ind, 'dates_without_predictions']
            dates.append(
                datetime.datetime.strftime(
                    datetime.datetime.strptime(dates[0], '%Y-%m-%d') - relativedelta(days=1), '%Y-%m-%d')
            )

            rv = revenue.RevenueInStaffing(job='making_predictions', date=dates)
            rv.making_predictions(model_file=model_file, dates_type='single_day')

    logger.info('revenue in staffing making predictions script successfully finished')