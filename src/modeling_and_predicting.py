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
import revenue_from_utlz
reload(revenue_from_utlz)


def revenue_from_utilization_script():

    rev = revenue_from_utlz.RevFromUtlz()
    rev.make_predictions()


def is_billable_retrospective_models_training(min_date = '2019-07-31', key = 'is_billable'):

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



def revenue_in_staffing_retrospective_models_training(min_date = '2019-09-01', key = 'revenue'):

    logger = logging.getLogger(__name__ + ' : revenue_in_staffing_retrospective_models_training')
    logger.info('revenue_in_staffing_retrospective_models_training')

    min_date_dt = datetime.datetime.strptime(min_date, '%Y-%m-%d').date()

    days = []

    while min_date_dt < datetime.date.today():
        days.append(datetime.datetime.strftime(min_date_dt, '%Y-%m-%d'))

        while (min_date_dt.month == (min_date_dt + relativedelta(days=7)).month) & \
                (min_date_dt < datetime.date.today() - relativedelta(days=7)):
            min_date_dt = min_date_dt + relativedelta(days=7)
            days.append(datetime.datetime.strftime(min_date_dt, '%Y-%m-%d'))

        min_date_dt = min_date_dt - relativedelta(days=7)
        min_date_dt = min_date_dt + relativedelta(months=1)
        min_date_dt = min_date_dt.replace(day=1)

    models = []
    for scope in ['no_staffing_required', 'assigned', 'in_staffing']:
        for f in listdir('../data/'):
            if '{}_{}_model_'.format(key, scope) in f:
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

        while d <= datetime.datetime.strptime(x['next_date'], '%Y-%m-%d'):
            lst.append(datetime.datetime.strftime(d, '%Y-%m-%d'))
            d = d + relativedelta(days=1)

        return lst

    def query_list_of_dates(x):

        try:
            params = ('revenue_probability', x['date'], x['next_date'])
            df = postgres.get_data('days_list_for_predictions.sql', param=params, param_type='tuple')

            return list(df.date)

        except AttributeError:
            return []

    def dates_without_predictions(x):

        return list(set(x['list_of_dates']) - set(x['dates_with_predictions']))

    def models_list(x):

        return [x['assigned'], x['in_staffing'], x['no_staffing_required']]

    files = []

    for scope in ['no_staffing_required', 'assigned', 'in_staffing']:

        for f in listdir('../data/'):
            if '{}_{}_model_'.format(key, scope) in f:
                files.append(f)

    df = pd.DataFrame(files, columns=['files'])
    df['date'] = df['files'].map(lambda x: x.split('.')[0][-10:]).astype(str)

    df['scope'] = df.files.map(lambda f: f[8:][:-21])

    tmp = pd.DataFrame()
    tmp['date'] = df.date.unique()
    tmp = tmp.sort_values('date')
    tmp['next_date'] = tmp.date.shift(-1).fillna(datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d'))

    for scope in ['no_staffing_required', 'assigned', 'in_staffing']:
        tmp[scope] = tmp.date.map(lambda d: list(df[(df.date == d) & (df.scope == scope)].files))
        tmp[scope] = tmp[scope].map(lambda a: a[0] if a else None)
        tmp[scope] = tmp[scope].fillna(method='ffill')

    tmp['models'] = tmp.apply(models_list, axis=1)
    tmp = tmp.drop(['no_staffing_required', 'assigned', 'in_staffing'], 1)

    tmp['list_of_dates'] = tmp.apply(list_of_dates, axis=1)
    tmp['dates_with_predictions'] = tmp.apply(query_list_of_dates, axis=1)
    tmp['dates_without_predictions'] = tmp.apply(dates_without_predictions, axis=1)
    tmp['list_len'] = tmp['dates_without_predictions'].map(lambda x: len(x))
    tmp = tmp[tmp['list_len'] != 0]
    tmp = tmp.drop(['date', 'next_date', 'list_of_dates', 'dates_with_predictions'], 1)

    for ind in tmp[tmp.list_len > 1].index:

        model_files = tmp.at[ind, 'models']
        dates = tmp.at[ind, 'dates_without_predictions']

        rv = revenue.RevenueInStaffing(job='making_predictions', date=dates)
        rv.making_predictions(model_files=model_files)

    if tmp[tmp.list_len == 1].shape[0] >=1:

        for ind in tmp[tmp.list_len == 1].index:

            model_files = tmp.at[ind, 'models']
            dates = tmp.at[ind, 'dates_without_predictions']
            dates.append(
                datetime.datetime.strftime(
                    datetime.datetime.strptime(dates[0], '%Y-%m-%d') - relativedelta(days=1), '%Y-%m-%d')
            )

            rv = revenue.RevenueInStaffing(job='making_predictions', date=dates)
            rv.making_predictions(model_files=model_files, dates_type='single_day')

    logger.info('revenue in staffing making predictions script successfully finished')


def is_billable_retrospective_predictions(key = 'is_billable'):

    logger = logging.getLogger(__name__ + ' : is_billable_retrospective_predictions')
    logger.info('is_billable_retrospective_predictions')
    postgres = PostgreSQLBase()

    def list_of_dates(x):
        lst = []
        d = datetime.datetime.strptime(x['date'], '%Y-%m-%d')

        while d <= datetime.datetime.strptime(x['next_date'], '%Y-%m-%d'):
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

    def models_list(x):

        return [x['assigned'], x['in_staffing'], x['no_staffing_required']]

    files = []

    for scope in ['no_staffing_required', 'assigned', 'in_staffing']:

        for f in listdir('../data/'):
            if '{}_{}_model_'.format(key, scope) in f:
                files.append(f)

    df = pd.DataFrame(files, columns=['files'])
    df['date'] = df['files'].map(lambda x: x.split('.')[0][-10:]).astype(str)

    df['scope'] = df.files.map(lambda f: f[12:][:-21])

    tmp = pd.DataFrame()
    tmp['date'] = df.date.unique()
    tmp = tmp.sort_values('date')
    tmp['next_date'] = tmp.date.shift(-1).fillna(datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d'))

    for scope in ['no_staffing_required', 'assigned', 'in_staffing']:
        tmp[scope] = tmp.date.map(lambda d: list(df[(df.date == d) & (df.scope == scope)].files))
        tmp[scope] = tmp[scope].map(lambda a: a[0] if a else None)
        tmp[scope] = tmp[scope].fillna(method='ffill')

    tmp['models'] = tmp.apply(models_list, axis=1)
    tmp = tmp.drop(['no_staffing_required', 'assigned', 'in_staffing'], 1)

    tmp['list_of_dates'] = tmp.apply(list_of_dates, axis=1)
    tmp['dates_with_predictions'] = tmp.apply(query_list_of_dates, axis=1)
    tmp['dates_without_predictions'] = tmp.apply(dates_without_predictions, axis=1)
    tmp['list_len'] = tmp['dates_without_predictions'].map(lambda x: len(x))
    tmp = tmp[tmp['list_len'] != 0]
    tmp = tmp.drop(['date', 'next_date', 'list_of_dates', 'dates_with_predictions'], 1)

    for ind in tmp[tmp.list_len > 1].index:

        model_files = tmp.at[ind, 'models']
        dates = tmp.at[ind, 'dates_without_predictions']

        tm = termination.Termination(job='making_predictions', date=dates)
        tm.making_predictions(model_files=model_files)

    if tmp[tmp.list_len == 1].shape[0] >=1:

        for ind in tmp[tmp.list_len == 1].index:

            model_files = tmp.at[ind, 'models']
            dates = tmp.at[ind, 'dates_without_predictions']
            dates.append(
                datetime.datetime.strftime(
                    datetime.datetime.strptime(dates[0], '%Y-%m-%d') - relativedelta(days=1), '%Y-%m-%d')
            )

            tm = termination.Termination(job='making_predictions', date=dates)
            tm.making_predictions(model_files=model_files, dates_type='single_day')

    logger.info('is_billable making predictions script successfully finished')