from datahub import PostgreSQLBase
import logging
import json
import string
import pandas as pd
import spacy
import random
from dateutil.relativedelta import relativedelta
import numpy as np
import datetime
import credentials
reload(credentials)

logger = logging.getLogger('model.auxiliary')
nlp = spacy.load('en')

class City_Country:
    """Class to process cities and countries from position 'locations' field

    Parameters
    ----------
    Use 'from_datahub' parameter, to get origin list with cities & countries from query.
    Another way, it should be loaded from local .pkl-file.

    After initialization the instance will have dataframe format variable 'id_city_country'
    with the following fields:

        - city_id
        - city
        - country_id
        - country

    """

    def __init__(self, source='from_datahub'):

        if source == 'from_datahub':

            postgres = PostgreSQLBase()
            df_location = postgres.get_data('location_data.sql')

            id_city_country = df_location[df_location.location_type_id == 6][[
                'location_id', 'location_name', 'parent_location_id_clean']].rename(columns={'location_name': 'city'})

            id_city_country = pd.merge(id_city_country,
                                       df_location.drop_duplicates('location_id').set_index('location_id')[
                                           ['location_name']].rename(columns={'location_name': 'country'}),
                                       how='left', left_on='parent_location_id_clean', right_index=True)

            id_city_country = id_city_country.rename(columns={'location_id': 'city_id',
                                                              'parent_location_id_clean': 'country_id'})

            id_city_country.city_id = id_city_country.city_id.astype(str)
            id_city_country.country_id = id_city_country.country_id.astype(str)

            id_city_country.to_pickle('../data/id_city_country.pkl')

            self.id_city_country = id_city_country

        else:

            id_city_country = pd.read_pickle('../data/id_city_country.pkl')

            self.id_city_country = id_city_country

    def city_country_processor(self, data, column_name='staffing_locations'):

        """
        This method processes dataframe with origin JSON fiels 'staffing location' from Datahub.
        The result is same dataframe object without 'staffing location' field having two new columns:
            - cities;
            - countries.

        The value in each cell of new columns is a list with all possible cities / countries for the particular position.

        """

        id_city_country = self.id_city_country
        df = data.copy()

        df[column_name] = df[column_name].astype(str)
        df[column_name+'_ids'] = df[column_name].map(
            lambda x:
            [loc.split('", "name": "')[0] for loc in x.split('"id": "')[1:]]
        )

        df['is_any_location'] = df[column_name+'_ids'].map(lambda x: x[0] if x else '')  # id for 'Any Location' = -1

        any_loc = df[df.is_any_location == '-1']

        any_loc[column_name+'_cities'] = [list(id_city_country.city.unique())] * len(any_loc)
        any_loc[column_name+'_countries'] = [list(id_city_country.country.unique())] * len(any_loc)

        df = df[df.is_any_location != '-1']

        df[column_name+'_cities'] = df[column_name+'_ids'].map(
            lambda x:
            str([id_city_country.set_index('city_id').get_value(str(i), 'city')
                 if str(i) in list(id_city_country.city_id)
                 else list(city for city in list(id_city_country[id_city_country.country_id == str(i)].city.unique()))
                 for i in x]).translate(None, "[]'").split(', ')
        )

        df[column_name+'_countries'] = df[column_name+'_ids'].map(
            lambda x:
            list(set(
                str([id_city_country.drop_duplicates('country_id').set_index('country_id').get_value(str(i), 'country')
                     if str(i) in list(id_city_country.country_id)
                     else id_city_country.set_index('city_id').get_value(str(i), 'country')
                     for i in x]).translate(None, "[]'").split(', ')))
        )

        df = pd.concat([any_loc, df])

        df = df.drop(['is_any_location', column_name+'_ids', column_name], 1)

        assert len(data) == len(df), 'Feature problem! The dataframe length was changed after the merge operation!!!'

        return df

    def city_country_ids_processor(self, data, column_name='staffing_locations'):

        """
        This method processes dataframe with origin JSON fiels 'staffing location' from Datahub.
        The result is same dataframe object without 'staffing location' field having two new columns:
            - city_ids;
            - country_ids.

        The value in each cell of new columns is a list with all possible ids of cities and countries
        for the particular position.

        """

        id_city_country = self.id_city_country
        df = data.copy()

        df[column_name] = df[column_name].astype(str)
        df[column_name+'_ids'] = df[column_name].map(
            lambda x:
            [loc.split('", "name": "')[0] for loc in x.split('"id": "')[1:]]
        )

        df['is_any_location'] = df[column_name+'_ids'].map(lambda x: x[0] if x else '')  # id for 'Any Location' = -1

        any_loc = df[df.is_any_location == '-1']

        any_loc[column_name+'_city_ids'] = [list(id_city_country.city_id.unique())] * len(any_loc)
        any_loc[column_name+'_country_ids'] = [list(id_city_country.country_id.unique())] * len(any_loc)

        df = df[df.is_any_location != '-1']

        df[column_name+'_city_ids'] = df[column_name+'_ids'].map(
            lambda x:
            str([str(i)
                 if str(i) in list(id_city_country.city_id)
                 else list(
                city for city in list(id_city_country[id_city_country.country_id == str(i)].city_id.unique()))
                 for i in x]).translate(None, "[]'").split(', ')
        )

        df[column_name+'_country_ids'] = df[column_name+'_ids'].map(
            lambda x:
            list(set(str([str(i)
                          if str(i) in list(id_city_country.country_id)
                          else id_city_country.set_index('city_id').get_value(str(i), 'country_id')
                          for i in x]).translate(None, "[]'").split(', ')))
        )

        df = pd.concat([any_loc, df])

        df = df.drop(['is_any_location', column_name+'_ids', column_name], 1)

        assert len(data) == len(df), 'Feature problem! The dataframe length was changed after the merge operation!!!'

        return df

    def straighten_cities_by_ids(self, data):

        """
        This method could be used only after usage of city_country_ids_processor method!!!

        This method processes dataframe with two columns:
            - city_ids;
            - country_ids.

        The value in each cell of this columns should be a list with all possible ids of cities and countries
        for the particular position.

        After applying this method you'll get new dataframe. But instead of list of city_ids in a cell for position
        there'll be a set of rows with only one city_id and country_id for every position.
        New columns are:
            - city_id;
            - country_id.

        For example:

            instead of old dataframe:

                    id | position_id | ... |  city_ids           | country_ids
                    -----------------------------------------------------------------------
                    1  | 12345678    | ... | [Minsk_id, Kiev_id] | [Belarus_id, Ukraine_id]

            you'll get the new one:

                    id | position_id | ... | city_id  | country_id
                    -----------------------------------------------
                    1  | 12345678    | ... | Minsk_id | Belarus_id
                    -----------------------------------------------
                    2  | 12345678    | ... | Kiev_id  | Ukraine_id

        To get city_names and country_namse from ids you could use the instance variable 'id_city_country'.

        """

        # works only wit lists of ids for cities and for countries, after usage of city_country_ids_processor method

        id_city_country = self.id_city_country
        df = data.copy()

        frames = []

        df['num_locs'] = df['staffing_locations_city_ids'].map(lambda x: len(x))

        single_loc = df[df['num_locs'] == 1]
        single_loc['staffing_locations_city_id'] = single_loc.staffing_locations_city_ids.map(lambda x: x[0])
        single_loc['staffing_locations_country_id'] = single_loc.staffing_locations_country_ids.map(lambda x: x[0])
        single_loc.drop(['staffing_locations_city_ids', 'staffing_locations_country_ids', 'num_locs'], 1, inplace=True)

        df = df[df['num_locs'] > 1]

        frames.append(single_loc)

        for ind in df.index:

            row = pd.DataFrame(df.iloc[df.index.get_loc(ind)]).T
            n = len(row.get_value(ind, 'staffing_locations_city_ids'))

            tmp = row.copy()

            while len(tmp) < n:
                tmp = tmp.append(row)

            tmp = pd.concat([
                tmp.reset_index().drop(['index'], 1),
                pd.DataFrame(row.get_value(ind, 'staffing_locations_city_ids'), columns=['staffing_locations_city_id']).reset_index().drop(['index'], 1)
            ], axis=1)

            tmp = pd.merge(
                tmp,
                id_city_country[['city_id', 'country_id']].set_index('city_id').rename(
                    columns={'country_id':'staffing_locations_country_id'}),
                how='left', left_on='staffing_locations_city_id', right_index=True
            )

            tmp.drop(['staffing_locations_city_ids', 'staffing_locations_country_ids', 'num_locs'], 1, inplace=True)

            frames.append(tmp)

        result = pd.concat(frames)

        result.drop(['staffing_locations_country_id'], 1, inplace=True)

        result = self.vlookuper(result)

        return result

    def vlookuper(self, data, from_field='city_id', to_field='country_id', column_name='staffing_locations'):

        """
        This method adds new column 'to_field', by the key = 'from_field'
        """

        df = data.copy()

        if column_name == 'none':

            df = pd.merge(
                df,
                self.id_city_country.drop_duplicates(from_field).set_index(from_field)[[to_field]],
                how='left', left_on=from_field, right_index=True
            )

        else:

            df = pd.merge(
                df,
                self.id_city_country.drop_duplicates(from_field).set_index(from_field)[[to_field]].rename(
                    columns={'country_id':column_name+'_'+to_field}),
                how='left', left_on=column_name+'_'+from_field, right_index=True
            )

        assert len(data) == len(df), 'Feature problem! The dataframe length was changed after the merge operation!!!'

        return df


class DataFrameProcessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ' : DataFrameProcessor')

    def multicategorical_cols(self, df, column_names):

        logger.debug('multicategorical columns process started')
        for col in column_names:
            df[col] = df[col].fillna('__UNDEFINED__')
            df[col] = df[col].astype(str)
            df[col] = df[col].map(
                lambda x:
                [loc.split('", "_class": "')[0] for loc in x.split('", "name": "')[1:]]
            )
            logger.debug(col)

        return df

    def datetime_cols(self, df, column_names):

        logger.debug('datetime columns process started')
        for col in column_names:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.debug(col)

        return df

    def text_cols(self, df, column_names):

        logger.debug('text columns process started')
        for col in column_names:
            df[col] = df[col].fillna('')
            df[col] = df[col].map(lambda x:
                                  "".join(
                                      [c for c in json.dumps(x, ensure_ascii=False) if
                                       c in string.printable]).decode(
                                      'utf-8', 'ignore')
                                  ).astype(unicode)
            df[col] = df[col].map(
                lambda desc:
                ' '.join([token.lemma_ for token in nlp(desc)])
            )

            logger.debug(col)

        return df

    def time_distance(self, df, from_col, to_col, new_col_name):

        logger.debug('time_distance process started')

        df[new_col_name] = df[to_col] - df[from_col]
        df[new_col_name] = df[new_col_name].dt.days.fillna(0)
        df[new_col_name] = df[new_col_name].astype(int)

        return df

    def loc_skill_target_processing(self, df, df_feature, target_column, agregate_column, prefix, res_type=int):

        length = len(df)

        loc_ids = City_Country()
        skills = Skills()

        tmp = df[['date', 'position_id', 'primary_skill_id', 'primary_skill_category_id',
                  'staffing_locations_city_ids', 'staffing_locations_country_ids']]
        tmp = loc_ids.straighten_cities_by_ids(tmp)
        tmp = skills.vlookuper(tmp, from_field='primary_skill_category_id', to_fields=['skill_prefix'])

        df_feature = skills.vlookuper(df_feature, to_fields=['primary_skill_category_id', 'skill_prefix'])
        df_feature[['primary_skill_category_id', 'skill_prefix']] = df_feature[
            ['primary_skill_category_id', 'skill_prefix']].fillna('__UNDEFINED__')

        loc_group = ['staffing_locations_city_id', 'staffing_locations_country_id', 'total']
        skill_group = ['primary_skill_id', 'primary_skill_category_id', 'skill_prefix', 'total']
        target_group = df_feature[target_column].unique()

        for loc in loc_group:
            for skill in skill_group:
                for target in target_group:

                    if (loc == 'total' and skill != 'total'):
                        df_targ_loc_skill = pd.DataFrame(df_feature[df_feature[target_column] == target]
                                                         .groupby(['date', skill])[agregate_column].sum()).reset_index()

                        col = prefix + '_' + target.replace(' ', '_').lower() + '_' + skill

                        tmp = pd.merge(
                            tmp.set_index(tmp.date.astype(str) + tmp[skill].astype(str)),
                            df_targ_loc_skill.set_index(df_targ_loc_skill.date.astype(str) +
                                                        df_targ_loc_skill[skill].astype(str)),
                            how='left', left_index=True, right_index=True
                        ).drop(['date_y', skill + '_y'], 1
                               ).rename(columns={'date_x': 'date', skill + '_x': skill, agregate_column: col}).fillna(0)

                        tmp[col] = tmp[col].astype(res_type)
                        t = pd.DataFrame(tmp.groupby(['position_id', 'date'])[col].mean()).reset_index()

                    elif (skill == 'total' and loc != 'total'):
                        df_targ_loc_skill = pd.DataFrame(df_feature[df_feature[target_column] == target]
                                                         .groupby(['date', loc])[agregate_column].sum()).reset_index()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_' + loc

                        tmp = pd.merge(
                            tmp.set_index(tmp.date.astype(str) + tmp[loc].astype(str)),
                            df_targ_loc_skill.set_index(
                                df_targ_loc_skill.date.astype(str) + df_targ_loc_skill[loc].astype(str)),
                            how='left', left_index=True, right_index=True
                        ).drop(['date_y', loc + '_y'], 1
                               ).rename(columns={'date_x': 'date', loc + '_x': loc, agregate_column: col}).fillna(0)

                        tmp[col] = tmp[col].astype(res_type)
                        tmp['key'] = tmp.position_id.astype(str)+tmp.date.astype(str)+tmp[loc].astype(str)
                        t = pd.DataFrame(tmp.drop_duplicates('key').groupby(['position_id', 'date'])[col].sum()).reset_index()

                    elif (loc == 'total' and skill == 'total'):
                        df_targ_loc_skill = pd.DataFrame(df_feature[df_feature[target_column] == target]
                                                         .groupby(['date'])[agregate_column].sum()).reset_index()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_company'

                        tmp = pd.merge(
                            tmp.set_index(tmp.date.astype(str)),
                            df_targ_loc_skill.set_index(df_targ_loc_skill.date.astype(str)),
                            how='left', left_index=True, right_index=True
                        ).drop(['date_y'], 1
                               ).rename(columns={'date_x': 'date', agregate_column: col}).fillna(0)

                        tmp[col] = tmp[col].astype(res_type)
                        t = pd.DataFrame(tmp.groupby(['position_id', 'date'])[col].mean()).reset_index()

                    else:
                        df_targ_loc_skill = pd.DataFrame(df_feature[df_feature[target_column] == target]
                                                         .groupby(['date', loc, skill])[
                                                             agregate_column].sum()).reset_index()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_' + loc + '_' + skill

                        tmp = pd.merge(
                            tmp.set_index(tmp.date.astype(str) + tmp[loc].astype(str) + tmp[skill].astype(str)),
                            df_targ_loc_skill.set_index(df_targ_loc_skill.date.astype(str) +
                                                        df_targ_loc_skill[loc].astype(str) + df_targ_loc_skill[
                                                            skill].astype(str)),
                            how='left', left_index=True, right_index=True
                        ).drop(['date_y', loc + '_y', skill + '_y'], 1
                               ).rename(
                            columns={'date_x': 'date', loc + '_x': loc, skill + '_x': skill, agregate_column: col
                                     }).fillna(0)

                        tmp[col] = tmp[col].astype(res_type)
                        tmp['key'] = tmp.position_id.astype(str) + tmp.date.astype(str) + tmp[loc].astype(str) + tmp[skill].astype(str)
                        t = pd.DataFrame(tmp.drop_duplicates('key').groupby(['position_id', 'date'])[col].sum()).reset_index()

                    df = pd.merge(
                        df,
                        t.set_index(t.date.astype(str) + t.position_id.astype(str))[[col]],
                        how='left', left_on=df.date.astype(str) + df.position_id.astype(str), right_index=True
                    )

                    # assert df[col].sum() > 0, 'Feature problem! Column were not joined!!!'
        assert length == len(df), 'Feature problem! The dataframe length was changed after the merge operation!!!'

        return df

    def feature_join_result_check(self, df, df_feature, target_column, agregate_column, prefix):

        loc_group = ['staffing_locations_city_id', 'staffing_locations_country_id', 'total']
        skill_group = ['primary_skill_id', 'total']
        target_group = list(df_feature[target_column].unique())

        for loc in loc_group:
            for skill in skill_group:
                for target in target_group:

                    if (loc == 'total' and skill != 'total'):

                        d = random.choice(tuple(df['date'].astype(str).unique()))
                        s = random.choice(tuple(df[
                                                    (df.date.astype(str) == d)
                                                ][skill].astype(str).unique()))

                        sum_feat = df_feature[
                            (df_feature.date.astype(str) == d) &
                            (df_feature[skill] == s) &
                            (df_feature[target_column] == target)
                            ][agregate_column].sum()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_' + skill

                        sum_df = df[
                            (df.date.astype(str) == d) &
                            (df[skill] == s)
                            ][col].mean()

                        mes = 'Assertion! For column: ' + col + ', skill_group: ' + skill + ', skill = ' + s + ', date = ' + d + \
                              ' and location = total for Company'

                        assert int(sum_feat) == int(sum_df), mes

                    elif (skill == 'total' and loc != 'total'):

                        d = random.choice(tuple(df['date'].astype(str).unique()))
                        l = random.choice(tuple(df[
                                                    (df.date.astype(str) == d)
                                                ][loc + 's'].astype(str).unique()))

                        sum_feat = df_feature[
                            (df_feature.date.astype(str) == d) &
                            (df_feature[loc].isin(eval(l)))&
                            (df_feature[target_column] == target)
                            ][agregate_column].sum()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_' + loc

                        sum_df = df[
                            (df.date.astype(str) == d) &
                            (df[loc + 's'].astype(str) == l)
                            ][col].mean()

                        mes = 'Assertion! For column: ' + col + ', location_group: ' + loc + ', loc = ' + l + ', date = ' + d + \
                              ' and skill = total for Company'

                        assert int(sum_feat) == int(sum_df), mes

                    elif (loc == 'total' and skill == 'total'):

                        d = random.choice(tuple(df['date'].astype(str).unique()))

                        sum_feat = df_feature[
                            (df_feature.date.astype(str) == d)&
                            (df_feature[target_column] == target)
                        ][agregate_column].sum()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_company'

                        sum_df = df[
                            (df.date.astype(str) == d)
                        ][col].mean()

                        mes = 'Assertion! For column: ' + col + ', date = ' + d + \
                              ', location and skill = total for Company'

                        assert int(sum_feat) == int(sum_df), mes

                    else:

                        d = random.choice(tuple(df['date'].astype(str).unique()))
                        s = random.choice(tuple(df[
                                                    (df.date.astype(str) == d)
                                                ][skill].unique()))
                        l = random.choice(tuple(df[
                                                    (df.date.astype(str) == d) &
                                                    (df[skill] == s)
                                                    ][loc + 's'].astype(str).unique()))

                        sum_feat = df_feature[
                            (df_feature.date.astype(str) == d) &
                            (df_feature[skill] == s) &
                            (df_feature[loc].isin(eval(l)))&
                            (df_feature[target_column] == target)
                            ][agregate_column].sum()

                        col = prefix + '_' +  target.replace(' ', '_').lower() + '_' + loc + '_' + skill

                        sum_df = df[
                            (df.date.astype(str) == d) &
                            (df[skill] == s) &
                            (df[loc + 's'].astype(str) == l)
                            ][col].mean()

                        mes = 'Assertion! For column: ' + col + ', location_group: ' + loc + ', loc = ' + l + ', date = ' + d + \
                              ', skill_group: ' + skill + ', skill = ' + s

                        assert int(sum_feat) == int(sum_df), mes

    def date_column_processing(self, data, date_column):

        df = data.copy()

        df.date = pd.to_datetime(df.date)
        df.planned_start_date = pd.to_datetime(df.planned_start_date)

        new_col = 'distance_from_the_date_to_' + date_column

        df[new_col] = np.where(
            (df['date'] >= df[date_column]), df.date - df[date_column], 0
        )
        df[new_col] = df[new_col].dt.days

        try:
            df[new_col + '_div_time_after_creation'] = df[new_col] * 100 / df.time_after_creation
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_planned_staffing_period'] = df[new_col] * 100 / df.planned_staffing_period
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_to_planned_start_date'] = df[new_col] * 100 / df.time_to_planned_start_date
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_from_first_proposals'] = df[new_col] * 100 / df.time_from_first_proposals
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_from_first_onboarding'] = df[new_col] * 100 / df.time_from_first_onboarding
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_from_first_booking'] = df[new_col] * 100 / df.time_from_first_booking
        except (AttributeError, ZeroDivisionError) as e:
            pass

        new_col = 'distance_from_the_planned_start_date_to_' + date_column

        df[new_col] = np.where(
            (df['date'] >= df[date_column]), df.planned_start_date - df[date_column], 0
        )
        df[new_col] = df[new_col].dt.days

        try:
            df[new_col + '_div_time_after_creation'] = df[new_col] * 100 / df.time_after_creation
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_planned_staffing_period'] = df[new_col] * 100 / df.planned_staffing_period
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_to_planned_start_date'] = df[new_col] * 100 / df.time_to_planned_start_date
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_from_first_proposals'] = df[new_col] * 100 / df.time_from_first_proposals
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_from_first_onboarding'] = df[new_col] * 100 / df.time_from_first_onboarding
        except (AttributeError, ZeroDivisionError) as e:
            pass

        try:
            df[new_col + '_div_time_from_first_booking'] = df[new_col] * 100 / df.time_from_first_booking
        except (AttributeError, ZeroDivisionError) as e:
            pass

        df = df.drop([date_column], 1)

        new_columns = list(set(df.columns) - set(data.columns))
        df[new_columns] = df[new_columns].fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

        return df

    def start_processing(self, df):
        loc_ids = City_Country()

        df['workload_start_date'] = df['workload_start_date'].astype(str)
        df['wkl_report_month'] = df.workload_start_date.map(lambda d: d[:7] if d else '')

        df['planned_end_date'] = df['planned_end_date'].fillna(0)
        df['planned_start_date'] = df['planned_start_date'].fillna(0)
        df['workload_start_date'] = df['workload_start_date'].fillna(0)
        df['created'] = df['created'].fillna(0)
        df['dates'] = df['date']

        df = self.datetime_cols(df, ['last_assigned_date', 'created', 'dates', 'planned_end_date', 'planned_start_date',
                                     'workload_start_date'])
        df = self.time_distance(df, 'created', 'workload_start_date', 'distance_between_creation_and_workload_start_date')
        df = self.time_distance(df, 'dates', 'workload_start_date',
                                'distance_between_snap_date_and_workload_start_date')
        df = self.time_distance(df, 'workload_start_date', 'planned_end_date',
                                'distance_between_workload_start_date_and_planned_end_date')

        df = self.time_distance(df, 'created', 'planned_start_date', 'planned_staffing_period')
        df = self.time_distance(df, 'planned_start_date', 'planned_end_date', 'planned_position_period')

        df = self.time_distance(df, 'created', 'dates', 'time_after_creation')
        df = self.time_distance(df, 'dates', 'planned_start_date', 'time_to_planned_start_date')
        df = self.time_distance(df, 'dates', 'planned_end_date', 'time_to_planned_end_date')

        df['wkl_cur_month'] = df.dates.map(lambda d: datetime.datetime.strftime(d, '%Y-%m-%d')[:7])
        df['wkl_next_1_month'] = df.dates.map(
            lambda d: datetime.datetime.strftime(d + relativedelta(months=1), '%Y-%m-%d')[:7])
        df['wkl_next_2_month'] = df.dates.map(
            lambda d: datetime.datetime.strftime(d + relativedelta(months=2), '%Y-%m-%d')[:7])

        df['wkl_next_plus_months'] = np.where(
            df.wkl_report_month == df.wkl_cur_month, 0, np.where(
                df.wkl_report_month == df.wkl_next_1_month, 0, np.where(
                    df.wkl_report_month == df.wkl_next_2_month, 0, 1
                )
            )
        )
        df['wkl_cur_month'] = np.where(df.wkl_cur_month == df.wkl_report_month, 1, 0)
        df['wkl_next_1_month'] = np.where(df.wkl_next_1_month == df.wkl_report_month, 1, 0)
        df['wkl_next_2_month'] = np.where(df.wkl_next_2_month == df.wkl_report_month, 1, 0)

        df['wkl_target'] = np.where(
            df.wkl_cur_month == 1, 0, np.where(
                df.wkl_next_1_month == 1, 1, np.where(
                    df.wkl_next_2_month == 1, 2, np.where(
                        df.wkl_next_plus_months == 1, 3, 4
                    )
                )
            )
        )

        df['creation_month'] = df['created'].map(lambda x: x.month)

        df['overdue_planned'] = np.where((df['dates'] > df['planned_start_date']) & (
                df['planned_start_date'] > datetime.datetime.strptime("01 Jan 1970", "%d %b %Y")), 1, 0)

        df['overdue_actual'] = np.where((df['dates'] > df['last_assigned_date']) & (
                df['last_assigned_date'] > datetime.datetime.strptime("01 Jan 1970", "%d %b %Y")), 1, 0)

        df['two_weeks_to_planned_start_date'] = df.time_to_planned_start_date.map(
            lambda x: 'yes' if (x <= 14) & (x > 0) else 'no')

        df.comment_id = df.comment_id.fillna(0)
        df.version = df.version.fillna(0)
        df = df.fillna('__UNDEFINED__')

        df = self.text_cols(df, ['description', 'position_name'])

        df['len_description'] = df.description.map(lambda x: len(x))

        df = self.multicategorical_cols(df, ['staffing_channels', 'project_coordinators', 'sales_sxecutives',
                                                'supply_owners', 'seniority_level', 'demand_owners',
                                                'container_staffing_coordinators',
                                                'sales_managers', 'program_managers', 'staffing_coordinators',
                                                'pmc_roles'])

        # df['staffing_channels_aggregation'] = df['staffing_channels'].map(lambda x: x[0] if x else '')
        # df = df[df['staffing_channels_aggregation'] != 'No staffing required']
        df = df.drop(['bt', 'dates'], 1)

        df['project_prefix'] = df['container_name'].str.split('-').str.get(0)

        df = loc_ids.city_country_ids_processor(df)
        df = loc_ids.city_country_ids_processor(df, column_name='container_locations')

        return df

    def first_processing_wkl_dataset(self, df):
        loc_ids = City_Country()

        df['planned_end_date'] = df['planned_end_date'].fillna(0)
        df['planned_start_date'] = df['planned_start_date'].fillna(0)
        df['dates'] = df['date']

        df = self.datetime_cols(df, ['last_assigned_date', 'created', 'dates', 'planned_end_date', 'planned_start_date'])

        df = self.time_distance(df, 'last_assigned_date', 'dates', 'time_of_assignment')
        df = self.time_distance(df, 'created', 'planned_start_date', 'planned_staffing_period')
        df = self.time_distance(df, 'planned_start_date', 'planned_end_date', 'planned_position_period')
        df = self.time_distance(df, 'dates', 'planned_end_date', 'time_to_planned_end_date')
        df = self.time_distance(df, 'created', 'dates', 'time_after_creation')
        df = self.time_distance(df, 'planned_start_date', 'dates', 'time_after_planned_start_date')

        df['creation_month'] = df['created'].map(lambda x: x.month)

        df['ped_in_the_end_of_Dec'] = df.planned_end_date.map(lambda x: 1 if (x.month == 12 and x.day > 15) else 0)
        df['ped_in_the_end_of_Jun'] = df.planned_end_date.map(lambda x: 1 if (x.month == 6 and x.day > 15) else 0)
        df['ped_at_the_end_of_month'] = df.planned_end_date.map(lambda x: 1 if (x.day > 24) else 0)
        df['ped_at_the_begining_of_month'] = df.planned_end_date.map(lambda x: 1 if (x.day < 5) else 0)

        df['ped_is_last_day_of_month'] = df.planned_end_date.map(
            lambda x: 1 if (
                    (x.month in [1, 3, 5, 7, 8, 10, 12] and x.day == 31) or (x.month in [4, 6, 9, 11] and x.day == 30
                                                                             ) or (
                            x.month == 2 and x.day in [28, 29])) else 0
        )

        df['ped_is_first_day_of_month'] = df.planned_end_date.map(lambda x: 1 if (x.day == 1) else 0)

        df['middle_of_the_year'] = df.planned_end_date.map(lambda x: datetime.datetime(year=x.year, month=7, day=1))
        df['end_of_the_year'] = df.planned_end_date.map(lambda x: datetime.datetime(year=x.year, month=12, day=31))

        df = self.time_distance(df, 'planned_end_date', 'middle_of_the_year',
                                   'time_distance_between_ped_end_middle_of_the_year')
        df = self.time_distance(df, 'planned_end_date', 'end_of_the_year',
                                   'time_distance_between_ped_end_end_of_the_year')

        df = df.drop(['middle_of_the_year', 'end_of_the_year'], 1)

        df.comment_id = df.comment_id.fillna(0)
        df.version = df.version.fillna(0)

        df = df.fillna('__UNDEFINED__')

        df = self.text_cols(df, ['description', 'position_name'])

        df['len_description'] = df.description.map(lambda x: len(x))

        df = self.multicategorical_cols(df, ['staffing_channels', 'project_coordinators', 'sales_sxecutives',
                                                'supply_owners', 'seniority_level', 'demand_owners',
                                                'container_staffing_coordinators',
                                                'sales_managers', 'program_managers', 'staffing_coordinators',
                                                'pmc_roles'])

        df['project_prefix'] = df['container_name'].str.split('-').str.get(0)

        df = df.drop(['bt', 'actual_end_date', 'dates'], 1)

        df = loc_ids.city_country_ids_processor(df)
        df = loc_ids.city_country_ids_processor(df, column_name='container_locations')

        return df

    def processor_time_relations_features_rampdown(self, df):
        logger = logging.getLogger(__name__ + ' : processor_time_relations_features_rampdown')
        logger.info('processor_time_relations_features_rampdown')

        new_features = []

        df['start_pos_in_proj_div_proj_duration'
        ] = df.start_pos_in_proj * 100 / df.proj_duration
        new_features.append('start_pos_in_proj_div_proj_duration')

        df['end_pos_in_proj_div_proj_duration'
        ] = df.end_pos_in_proj * 100 / df.proj_duration
        new_features.append('end_pos_in_proj_div_proj_duration')

        df['time_of_assignment_div_proj_duration'
        ] = df.time_of_assignment * 100 / df.proj_duration
        new_features.append('time_of_assignment_div_proj_duration')

        df['time_after_planned_start_date_div_proj_duration'
        ] = df.time_after_planned_start_date * 100 / df.proj_duration
        new_features.append('time_after_planned_start_date_div_proj_duration')

        df['time_to_planned_end_date_div_proj_duration'
        ] = df.time_to_planned_end_date * 100 / df.proj_duration
        new_features.append('time_to_planned_end_date_div_proj_duration')

        df['proj_before_created_div_proj_duration'
        ] = df.proj_before_created * 100 / df.proj_duration
        new_features.append('proj_before_created_div_proj_duration')

        df['proj_after_created_div_proj_duration'
        ] = df.proj_after_created * 100 / df.proj_duration
        new_features.append('proj_after_created_div_proj_duration')

        df['planned_position_period_div_proj_duration'
        ] = df.planned_position_period * 100 / df.proj_duration
        new_features.append('planned_position_period_div_proj_duration')

        df['planned_staffing_period_div_proj_duration'
        ] = df.planned_staffing_period * 100 / df.proj_duration
        new_features.append('planned_staffing_period_div_proj_duration')

        df[new_features] = df[new_features].fillna(0)

        return df



class Skills:
    """Class to process primary_skills and skill categories from position 'primary_skill_id' field

    Parameters
    ----------
    Use 'from_datahub' parameter, to get origin list with cities & countries from query.
    Another way, it should be loaded from local .pkl-file.

    After initialization the instance will have dataframe format variable 'skill_ids'
    with the following fields:

        - primary_skill_id
        - primary_skill_name
        - primary_skill_category_id
        - primary_skill_category_name
        - skill_prefix

    """

    def __init__(self, source='from_datahub'):

        if source == 'from_datahub':

            postgres = PostgreSQLBase()
            skill_ids = postgres.get_data('skill_data.sql')
            skill_ids['skill_prefix'] = skill_ids['primary_skill_category_name'].str.split('.').str.get(0)

            skill_ids.to_pickle('../data/skill_ids.pkl')

            self.skill_ids = skill_ids

        else:

            skill_ids = pd.read_pickle('../data/skill_ids.pkl')

            self.skill_ids = skill_ids


    def vlookuper(self, data, from_field='primary_skill_id', to_fields = [
        'primary_skill_name', 'primary_skill_category_id', 'primary_skill_category_name', 'skill_prefix'
    ]):

        """
        This method adds new columns from list to_fields, by the key = 'from_field'.
        """

        df = data.copy()

        df = pd.merge(
            df,
            self.skill_ids.drop_duplicates(from_field).set_index(from_field)[to_fields],
            how='left', left_on=from_field, right_index=True
        )

        assert len(data) == len(df), 'Feature problem! The dataframe length was changed after the merge operation!!!'

        return df


class Matcher_by_ids:
    """Class to process customer names from 'customer_id'

    Parameters
    ----------
    Use 'from_datahub' parameter, to get origin list with cities & countries from query.
    Another way, it should be loaded from local .pkl-file.

    After initialization the instance will have dataframe format variable 'customer_ids'
    with the following fields:

        - customer_id
        - customer_name

    """

    def __init__(self, source='from_datahub'):

        if source == 'from_datahub':

            postgres = PostgreSQLBase()
            customer_ids = postgres.get_data('customer_data.sql')

            customer_ids.to_pickle('../data/customer_ids.pkl')

            self.customer_ids = customer_ids

        else:

            customer_ids = pd.read_pickle('../data/customer_ids.pkl')

            self.customer_ids = customer_ids


    def vlookuper(self, data, from_field='customer_id', to_field = 'customer_name', table='customer'):

        """
        This method adds new column from to_field, by the key = 'table' + '_' + 'from_field'.
        """

        df = data.copy()

        df = pd.merge(
            df,
            self.customer_ids.drop_duplicates(from_field).set_index(from_field)[[to_field]],
            how='left', left_on=from_field, right_index=True
        )

        assert len(data) == len(df), 'Feature problem! The dataframe length was changed after the merge operation!!!'

        return df


class CRMDataProcessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ' : CRMDataProcessor')

        self.month_dict = {
            1 : 'jan',
            2 : 'feb',
            3 : 'mar',
            4 : 'apr',
            5 : 'may',
            6 : 'jun',
            7 : 'jul',
            8 : 'aug',
            9 : 'sep',
            10 : 'oct',
            11 : 'nov',
            12 : 'dec'
        }

        self.quarter_dict = {
            'q1' : ['jan', 'feb', 'mar'],
            'q2' : ['apr', 'may', 'jun'],
            'q3' : ['jul', 'aug', 'sep'],
            'q4' : ['oct', 'nov', 'dec']
        }

        self.half_year_dict = {
            'h1' : ['jan', 'feb', 'mar', 'apr', 'may', 'jun'],
            'h2' : ['jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        }


        self.base_dict = {
            'project': 'project_name',
            'customer': 'customer_name',
            'gbu': 'gbu_id'
        }

        self.probability_dict = {
            '0% - Non-qualified': 0,
            '10% - Qualified': 0.1,
            '20% - Proposal submitted': 0.2,
            '40% - Solution accepted / EPAM shortlisted': 0.4,
            '60% - Verbal win notification': 0.6,
            '80% - Written win notification': 0.8,
            '90% - Contract Negotiation': 0.9,
            '100% - Contract Signed': 1.0
        }


    def range_groupper(self, data, aggs_range='month'):
        """
        This method groups numerical data within dataset from monthly ranges into one of:
                - quarterly,
                - 6 monthly,
                - yearly.

            Parameters
            --------------
            data : pd.DataFrame
                Original dataset to make data processing.
                The dataset should have revenue informations with following columns:
                    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'

            aggs_range : str
                One of possible values:
                    - 'month' - return dataset itself,
                    - 'quarter' - groups data within year quarters,
                    - 'half_year' - groups data within year halfs,
                    - 'year' - groups data yearly

            Returns
            -------------
            Updated dataset without original monthly columns

        """
        df = data.copy()

        if aggs_range == 'month':
            return df

        elif aggs_range == 'quarter':

            for k in self.quarter_dict.keys():
                df[k] = df[self.quarter_dict[k][0]] + df[self.quarter_dict[k][1]] + df[self.quarter_dict[k][2]]
                df = df.drop(self.quarter_dict[k], 1)
            return df

        elif aggs_range == 'half_year':

            for k in self.half_year_dict.keys():
                df[k] = df[self.half_year_dict[k][0]] + df[self.half_year_dict[k][1]] + df[self.half_year_dict[k][2]] + \
                        df[self.half_year_dict[k][3]] + df[self.half_year_dict[k][4]] + df[self.half_year_dict[k][5]]
                df = df.drop(self.half_year_dict[k], 1)
            return df

        elif aggs_range == 'year':

            df['year_value_sum'] = df[self.month_dict[1]] + df[self.month_dict[2]] + df[self.month_dict[3]] + \
                            df[self.month_dict[4]] + df[self.month_dict[5]] + df[self.month_dict[6]] + \
                            df[self.month_dict[7]] + df[self.month_dict[8]] + df[self.month_dict[9]] + \
                            df[self.month_dict[10]] + df[self.month_dict[11]] + df[self.month_dict[12]]
            df = df.drop(self.month_dict.values(), 1)
            return df


    def ranges_types_combiner(self, data, aggs_type='current', aggs_range='month'):

        """
        This method defines current / next / previous time interval with taking into account different kinds of ranges:
        month / quarter / half_year / year

            Parameters
            --------------
            data : pd.DataFrame
                Original dataset to make data processing.
                The dataset should have column = 'date' with str in formate 'YYYY-mm-dd'
            aggs_ramge : str
                One of possible values:
                - 'month' - returns month name,
                - 'quarter' - returns quarter name,
                - 'half_year' - returns 'h1' for first half of a year or 'h2' for second,
                - 'year' - return int for year
            aggs_type : str
                One of possible values:
                - 'current' for current time interval,
                - 'previous' for previous time interval,
                - 'next' for the next time interval.

            Returns
            -------------
            Updated dataset with adding one new column, in order of processing timestamp in 'date' column.
            Examples:
            aggs_range='month', aggs_type='current'  - new column 'current_month'
            aggs_range='quarter', aggs_type='next'  - new column 'next_quarter'

        """

        df_proc = DataFrameProcessor()

        df = data.copy()

        df = df_proc.datetime_cols(df, ['date'])


        if aggs_type == 'current':

            if aggs_range == 'month':
                df['current_month'] = df.date.map(lambda d: self.month_dict[d.month])
                return df

            elif aggs_range == 'quarter':
                df['current_quarter'] = df.date.map(
                lambda d: [q for q, r in self.quarter_dict.items() if self.month_dict[d.month] in r][0])
                return df

            elif aggs_range == 'half_year':
                df['current_half_year'] = df.date.map(
                    lambda d: [h for h, r in self.half_year_dict.items() if self.month_dict[d.month] in r][0])
                return df

            elif aggs_range == 'year':
                df['current_year'] = df.date.map(lambda d: d.year)
                return df

        elif aggs_type == 'next':

            if aggs_range == 'month':
                df['next_month'] = df.date.map(lambda d: self.month_dict[d.month+1] if d.month < 12 else 'jan')
                return df

            elif aggs_range == 'quarter':
                df['next_quarter'] = df.date.map(lambda d: [q for q, r in self.quarter_dict.items() if (
                    self.month_dict[d.month+3] if (d.month+3) <= 12 else self.month_dict[d.month+3-12]
                ) in r][0])
                return df

            elif aggs_range == 'half_year':
                df['next_half_year'] = df.date.map(
                    lambda d: [h for h, r in self.half_year_dict.items() if self.month_dict[d.month] not in r][0])
                return df

            elif aggs_range == 'year':
                df['next_year'] = df.date.map(lambda d: d.year + 1)
                return df

        elif aggs_type == 'previous':

            if aggs_range == 'month':
                df['previous_month'] = df.date.map(lambda d: self.month_dict[d.month-1] if d.month > 1 else 'dec')
                return df

            elif aggs_range == 'quarter':
                df['previous_quarter'] = df.date.map(lambda d: [q for q, r in self.quarter_dict.items() if (
                    self.month_dict[d.month-3] if (d.month-3) >= 1 else self.month_dict[d.month-3+12]
                ) in r][0])
                return df

            elif aggs_range == 'half_year':
                df['previous_half_year'] = df.date.map(
                    lambda d: [h for h, r in self.half_year_dict.items() if self.month_dict[d.month] not in r][0])
                return df

            elif aggs_range == 'year':
                df['previous_year'] = df.date.map(lambda d: d.year - 1)
                return df


    def crm_data_groupper(self, data, scenario = 'Standard', aggs_type = 'current', aggs_range = 'month', base = 'project', value = 'revenue'):

        """
        This method groups crm revenue (Forecast or Actual) from input dataset 'data'
        by the grouping base = 'base'. And positions counts (for value = 'positions')

        :param data: pd.DataFrame
        CRM dataset with monthly revenue, column of the snapshot 'date'. And columns of possible grouping bases.

        :param aggs_type: str
        The time period description: 'previous', 'current' or 'next'.

        :param aggs_range: str
        The time period duration: 'month', 'quarter', 'half_year' or 'year'.

        :param base: str
        The grouping base description: 'project', 'customer' or 'bgu'.

        :param value: str
        The type of aggregated value: 'revenue' or 'positions'.

        :return: pd.DataFrame
        The dataset which is ready to be joined with the main dataset. Contains the following columns:
            - date
            - 'base_column' with the appropriate name ('project_name', 'customer_id' or 'gbu_id')
            - grouped revenue column with the appropriate name. For example: 'previous_month_project_revenue' or
                'next_year_gbu_revenue'.
        """

        ranges = []
        switch_list = []

        if value == 'revenue':
            df = self.crm_scenarios_applier(data, scenario=scenario)

        else:
            df = data.copy()

        df = df.groupby(['year', self.base_dict[base], 'date'])[
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'].sum().reset_index()

        df = self.range_groupper(df, aggs_range=aggs_range)

        df = self.ranges_types_combiner(df, aggs_type=aggs_type, aggs_range=aggs_range)

        if aggs_range != 'year':
            df = self.ranges_types_combiner(df, aggs_type=aggs_type, aggs_range='year')

            if aggs_type != 'current':
                df = self.ranges_types_combiner(df, aggs_type='current', aggs_range='year')

        if aggs_range == 'month':
            exec ('ranges = self.' + aggs_range + '_dict.values()')

        elif aggs_range in ['quarter', 'half_year']:
            exec ('ranges = self.' + aggs_range + '_dict.keys()')

        if aggs_type != 'current':

            next_list = ['jan', 'q1', 'h1']
            previous_list = ['dec', 'q4', 'h2']
            exec ('switch_list = ' + aggs_type + '_list')

            if aggs_range != 'year':

                new_df = []
                for r in ranges:

                    if r not in switch_list:
                        tmp = df[
                            (df[aggs_type + '_' + aggs_range] == r) &
                            (df.year == df.current_year)
                            ]

                    else:
                        tmp = df[
                            (df[aggs_type + '_' + aggs_range] == r) &
                            (df.year == df[aggs_type + '_year'])
                            ]

                    tmp[scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value] = tmp[r]
                    new_df.append(tmp)

                df = pd.concat(new_df)

            else:
                df = df[df.year == df[aggs_type + '_year']].rename(columns = {
                    'year_value_sum' : scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value
                })

        else:
            if aggs_range != 'year':

                new_df = []
                for r in ranges:
                    tmp = df[
                        (df[aggs_type + '_' + aggs_range] == r) &
                        (df.year == df.current_year)
                        ]

                    tmp[scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value] = tmp[r]
                    new_df.append(tmp)

                df = pd.concat(new_df)

            else:
                df = df[df.year == df.current_year].rename(columns = {
                    'year_value_sum' : scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value
                })

        df = df[[self.base_dict[base], 'date', scenario + '_' + aggs_type + '_' + aggs_range + '_' + base + '_' + value]]

        return df


    def previous_actual_revenue_correction(self, data, crm_act, crm_plan, scenario = 'Standard', base = 'project'):

        """
        This method corrects previous actual revenue in cases, when the actual revenue per previous month doesn't exist.
        Only planned revenue. The method adds the planned revenue per previous month to the:
            - previous month revenue (if exist),
            - previous quarter revenue (if exist),
            - previous half_year revenue (if exist),
            - previous year revenue (if exist).
        :param data: pd.DataFrame
        :param crm_act: pd.DataFrame
        The CRM actual revenue dataset.
        :param crm_plan: pd.DataFrame
        The CRM planned revenue dataset.
        :param base: str
        The grouping base description: 'project', 'customer' or 'bgu'.
        :return: pd.DataFrame
        The dataset with corrected previous periods revenue.
        """

        logger.info('previous_actual_revenue_correction: {0} scenario: revenue grouped by {1}: started'.format(*(scenario, base)))

        df = data.copy()

        tmp_act = self.crm_data_groupper(crm_act, scenario='Optimistic', aggs_type='previous', aggs_range='month', base=base)
        tmp_plan = self.crm_data_groupper(crm_plan, scenario=scenario, aggs_type='previous', aggs_range='month', base=base)

        tmp = tmp_act[tmp_act['Optimistic_previous_month_' + base + '_revenue'] == 0]

        tmp['day'] = tmp.date.map(lambda d: d.day)
        tmp = tmp[tmp['day'] <= 15]

        tmp = pd.merge(
            tmp,
            tmp_plan.set_index(tmp_plan[self.base_dict[base]] + tmp_plan.date.astype(str))[[scenario + '_previous_month_' + base + '_revenue']
            ].rename(columns={scenario + '_previous_month_' + base + '_revenue': scenario + '_plan_previous_month_' + base + '_revenue'}),
            how='left', left_on=tmp[self.base_dict[base]] + tmp.date.astype(str), right_index=True
        )

        tmp = tmp[tmp[scenario + '_plan_previous_month_' + base + '_revenue'] > 0]

        df = pd.merge(
            df,
            tmp.set_index(tmp[self.base_dict[base]] + tmp.date.astype(str))[
                [scenario + '_plan_previous_month_' + base + '_revenue']],
            how='left', left_on=df[self.base_dict[base]] + df.date.astype(str), right_index=True
        )

        df = self.ranges_types_combiner(df, aggs_type='previous', aggs_range='month')

        if scenario+'_previous_month_'+base+'_revenue' in df.columns:

            df[scenario+'_previous_month_'+base+'_revenue'] = np.where(
                df[scenario+'_plan_previous_month_' + base + '_revenue'].notnull(),
                df[scenario+'_previous_month_' + base + '_revenue'] + df[scenario+'_plan_previous_month_' + base + '_revenue'],
                df[scenario+'_previous_month_' + base + '_revenue']
            )

        if scenario+'_previous_quarter_'+base+'_revenue' in df.columns:

            df[scenario+'_previous_quarter_'+base+'_revenue'] = np.where(
                (df[scenario+'_plan_previous_month_' + base + '_revenue'].notnull())&
                (df.previous_month.isin(['mar', 'jun', 'sep', 'dec'])),
                df[scenario+'_previous_quarter_' + base + '_revenue'] + df[scenario+'_plan_previous_month_' + base + '_revenue'],
                df[scenario+'_previous_quarter_' + base + '_revenue']
            )

        if scenario+'_previous_half_year_'+base+'_revenue' in df.columns:

            df[scenario+'_previous_half_year_'+base+'_revenue'] = np.where(
                (df[scenario+'_plan_previous_month_' + base + '_revenue'].notnull())&
                (df.previous_month.isin(['jun', 'dec'])),
                df[scenario+'_previous_half_year_' + base + '_revenue'] + df[scenario+'_plan_previous_month_' + base + '_revenue'],
                df[scenario+'_previous_half_year_' + base + '_revenue']
            )

        if scenario+'_previous_year_'+base+'_revenue' in df.columns:

            df[scenario+'_previous_year_'+base+'_revenue'] = np.where(
                (df[scenario+'_plan_previous_month_' + base + '_revenue'].notnull())&
                (df.previous_month == 'dec'),
                df[scenario+'_previous_year_' + base + '_revenue'] + df[scenario+'_plan_previous_month_' + base + '_revenue'],
                df[scenario+'_previous_year_' + base + '_revenue']
            )

        df = df.drop(['previous_month', scenario+'_plan_previous_month_' + base + '_revenue'], 1)

        return df


    def crm_scenarios_applier(self, data, scenario = 'Standard'):

        """
        This methods applies different scenarios to crm data.
        The possible srenarios are:

            - Standard.
                - For past quarters - 80%+ opportunities revenue
                - For current quarter - 80%+ opportunities revenue,
                - For next quarter - opportunities 60%+ revenue,
                - For quarter after next - opportunities 40%+ revenue
                - For future quarters - 0%+ revenue.
            - Optimistic. Revenue shown all opportunities.
            - Pessimistic. Revenue shown for won opportunities only.
            - Weighted. Revenue amounts are multiplied by opportunity probability.

        :param data: pd.DataFrame
        :param scenario: str
        'Standard', 'Optimistic', 'Pessimistic', Weighted.
        :return: pd.DataFrame
        The dataframe with selected data.
        """

        df = data.copy()

        past_quarters = {
            'q1': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q2': ['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q3': ['jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q4': ['oct', 'nov', 'dec']
        }

        current_quarters = {
            'q1': ['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q2': ['jan', 'feb', 'mar', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q3': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'oct', 'nov', 'dec'],
            'q4': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep']
        }

        next_quarters = {
            'q1': ['jan', 'feb', 'mar', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q2': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'oct', 'nov', 'dec'],
            'q3': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep'],
            'q4': ['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        }

        quarter_after_next = {
            'q1': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'oct', 'nov', 'dec'],
            'q2': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep'],
            'q3': ['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            'q4': ['jan', 'feb', 'mar', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        }

        future_quarters = {
            'q1': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep'],  # cancel current year except q4
            'q2': [],  # get all next year
            'q3': ['jan', 'feb', 'mar'],  # cancel q1 of next year
            'q4': ['jan', 'feb', 'mar', 'apr', 'may', 'jun']  # cancel q1 and q2 of next year
        }


        if scenario == 'Optimistic':
            return df

        elif scenario == 'Pessimistic':

            df = df[
                (df.probability == '100% - Contract Signed')
            ]

            return df

        elif scenario == 'Weighted':

            df['prob'] = df.probability.map(lambda p: self.probability_dict[p])

            for m in self.month_dict.values():
                df[m] = df[m] * df['prob']

            df = df.drop(['prob'], 1)

            return df

        elif scenario == 'Standard':

            df = self.ranges_types_combiner(df, aggs_range='quarter')
            df = self.ranges_types_combiner(df, aggs_range='year')

            # past years

            part1 = df[
                (df.year < df.current_year) &
                (df.probability.isin(
                    ['80% - Written win notification', '90% - Contract Negotiation', '100% - Contract Signed']))
                ]

            # current year, past quarters

            tmp_df = []
            for q in past_quarters.keys():

                tmp = df[
                    (df.year == df.current_year) &
                    (df.current_quarter == q) &
                    (df.probability.isin(
                        ['80% - Written win notification', '90% - Contract Negotiation', '100% - Contract Signed']))
                    ]

                for m in past_quarters[q]:
                    tmp[m] = 0

                tmp_df.append(tmp)
            part2 = pd.concat(tmp_df)

            # current quarter

            tmp_df = []
            for q in current_quarters.keys():

                tmp = df[
                    (df.year == df.current_year) &
                    (df.current_quarter == q) &
                    (df.probability.isin(
                        ['80% - Written win notification', '90% - Contract Negotiation', '100% - Contract Signed']))
                    ]

                for m in current_quarters[q]:
                    tmp[m] = 0

                tmp_df.append(tmp)
            part3 = pd.concat(tmp_df)

            # next quarter

            tmp_df = []
            for q in next_quarters.keys():

                if q != 'q4':

                    tmp = df[
                        (df.year == df.current_year) &
                        (df.current_quarter == q) &
                        (df.probability.isin(['60% - Verbal win notification', '80% - Written win notification',
                                              '90% - Contract Negotiation', '100% - Contract Signed']))
                        ]

                    for m in next_quarters[q]:
                        tmp[m] = 0

                    tmp_df.append(tmp)

                else:
                    tmp = df[
                        (df.year == df.current_year + 1) &
                        (df.current_quarter == q) &
                        (df.probability.isin(['60% - Verbal win notification', '80% - Written win notification',
                                              '90% - Contract Negotiation', '100% - Contract Signed']))
                        ]

                    for m in next_quarters[q]:
                        tmp[m] = 0

                    tmp_df.append(tmp)

            part4 = pd.concat(tmp_df)

            # quarter after next

            tmp_df = []
            for q in quarter_after_next.keys():

                if q in ['q1', 'q2']:

                    tmp = df[
                        (df.year == df.current_year) &
                        (df.current_quarter == q) &
                        (df.probability.isin(['40% - Solution accepted / EPAM shortlisted',
                                              '60% - Verbal win notification', '80% - Written win notification',
                                              '90% - Contract Negotiation', '100% - Contract Signed']))
                        ]

                    for m in quarter_after_next[q]:
                        tmp[m] = 0

                    tmp_df.append(tmp)

                else:
                    tmp = df[
                        (df.year == df.current_year + 1) &
                        (df.current_quarter == q) &
                        (df.probability.isin(['40% - Solution accepted / EPAM shortlisted',
                                              '60% - Verbal win notification', '80% - Written win notification',
                                              '90% - Contract Negotiation', '100% - Contract Signed']))
                        ]

                    for m in quarter_after_next[q]:
                        tmp[m] = 0

                    tmp_df.append(tmp)

            part5 = pd.concat(tmp_df)

            # future quarters

            tmp_df = []
            for q in future_quarters.keys():

                if q == 'q1':

                    tmp = df[
                        (df.year == df.current_year) &
                        (df.current_quarter == q)
                        ]

                    for m in future_quarters[q]:
                        tmp[m] = 0

                    tmp_df.append(tmp)

                    tmp = df[
                        (df.year > df.current_year) &
                        (df.current_quarter == q)
                        ]

                    tmp_df.append(tmp)

                elif q == 'q2':

                    tmp = df[
                        (df.year > df.current_year) &
                        (df.current_quarter == q)
                        ]

                    tmp_df.append(tmp)

                elif q in ['q3', 'q4']:

                    tmp = df[
                        (df.year == df.current_year + 1) &
                        (df.current_quarter == q)
                        ]

                    for m in future_quarters[q]:
                        tmp[m] = 0

                    tmp_df.append(tmp)

                    tmp = df[
                        (df.year > df.current_year + 1) &
                        (df.current_quarter == q)
                        ]

                    tmp_df.append(tmp)

            part6 = pd.concat(tmp_df)

            df = pd.concat([part1, part2, part3, part4, part5, part6])

            df = df.drop(['current_quarter', 'current_year'], 1)

            return df