import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta


def splitting(data, sorting='max_date', date='2018-01-01'):
    # train / test splitting the data

    df = data.copy()
    df['date'] = pd.to_datetime(df.date)
    df[sorting] = pd.to_datetime(df[sorting])
    d = datetime.datetime.strptime(date, '%Y-%m-%d')

    df.sort_values(sorting, inplace=True)

    train = df[df[sorting] < d]
    test = df[(df.date >= d) & (df[sorting] > d)]

    X_train = train.drop(['created', 'actual_start_date', 'target', 'max_date', 'last_assigned_date',
                          'last_staffing_status_update_date', 'date'], 1)
    X_test = test.drop(['created', 'actual_start_date', 'target', 'max_date', 'last_assigned_date',
                        'last_staffing_status_update_date', 'date'], 1)

    y_train = train['target']
    y_test = test['target']

    return X_train, X_test, y_train, y_test


def train_val_splitting(start_date='2016-07-01', end_date='2018-01-31', n=3, alf=0.25):

    start_d = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_d = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    delt = 1 / alf - 1

    t = round((end_d - start_d).days / (n + delt))

    start_tr = []
    end_tr = []
    start_val = []
    end_val = []

    for i in range(n):

        start_tr_d = start_d + relativedelta(days=t * i)
        end_tr_d = start_d + relativedelta(days=t * (i + delt), microseconds=-1)
        start_val_d = start_d + relativedelta(days=t * (i + delt))
        if i == (n - 1):
            end_val_d = end_d + relativedelta(days=1, microseconds=-1)
        else:
            end_val_d = start_d + relativedelta(days=t * (i + delt + 1), microseconds=-1)

        start_tr.append(start_tr_d)
        end_tr.append(end_tr_d)
        start_val.append(start_val_d)
        end_val.append(end_val_d)


    return start_tr, end_tr, start_val, end_val


def short_train_val_splitting(start_date='2016-07-01', end_date='2018-01-31', n=3, train_len=3, test_len=1):
    # train_len and test_len are set in months
    # if n is smaller than available folds number, n would be taken from the end of the time period

    start_d = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_d = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    def available_folds_number(start_d, end_d, train_len, test_len):

        months_num = 0
        date = start_d

        while (date <= end_d + relativedelta(days=1)):
            months_num = months_num + 1
            date = date + relativedelta(months=1)

        months_num = months_num - 1

        return (months_num - train_len) / test_len

    max_num = available_folds_number(start_d, end_d, train_len, test_len)

    assert n <= max_num, 'number of folds could not be larger than: ' + max_num

    start_tr = []
    end_tr = []
    start_val = []
    end_val = []

    for i in range(n):
        end_val.append(end_d - relativedelta(months=i))
        start_val.append(end_d - relativedelta(months=test_len + i) + relativedelta(days=1))
        end_tr.append(end_d - relativedelta(months=test_len + i))
        start_tr.append(end_d - relativedelta(months=(test_len + train_len + i)) + relativedelta(days=1))

    return start_tr, end_tr, start_val, end_val


def X_y_split(data):
    df = data.copy()

    X = df.drop(['target', 'created', 'actual_start_date', 'max_date',
                 'last_assigned_date', 'last_staffing_status_update_date'], 1)
    y = df['target']

    return X, y