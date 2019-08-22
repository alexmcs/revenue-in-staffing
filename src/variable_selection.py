
import pandas as pd
import numpy as np


def variable_selection(data):
    """
    for classification problem task value should be 'clf'; for regression problem: 'reg'
    :param data:
    :param task:
    :return:
    """

    df = data.copy()
    #df.set_index('position_id', inplace=True)
    text_cols = ['description', 'position_name', 'proj_description', 'comments']  #

    cat_cols = ['account_manager_name', 'billing_type_name', 'calc_utilization_group_name',
                'container_project_billing_type_name', 'container_project_category_name',
                'customer_is_new', 'gbu_name', 'ibu_name',
                'primary_skill_category_name', 'primary_skill_name', 'priority_name',
                'probability_name', 'remote_work_allowed', 'role_name',
                'city', 'country', 'overdue_planned', 'overdue_actual',
                'supervisor_name', 'project_manager_name', 'coordinator_name', 'project_supervisor_name',
                'delivery_supervisor_name', 'assignee_name',
                'project_sponsor_name',
                'admin_project_agg',
                'container_type', 'skill_prefix', 'project_prefix', 'seniority', 'proj_is_billable',
                'is_using_template', 'creation_month', 'position_status',
                'staffing_status',
                'creator_name', 'container_name',
                'customer_name', 'delivery_manager_name', 'manager_name',
                'account_manager_unit_name',
                'supervisor_unit_name',
                'project_manager_unit_name',
                'coordinator_unit_name',
                'project_supervisor_unit_name',
                'delivery_supervisor_unit_name',
                'assignee_unit_name',
                'project_sponsor_unit_name',
                'creator_unit_name',
                'delivery_manager_unit_name',
                'manager_unit_name', 'two_weeks_to_planned_start_date',
                'account_manager_id', 'billing_type_id', 'calc_utilization_group_id',
                'container_project_billing_type_id', 'container_project_category_id',
                'gbu_id', 'ibu_id',
                'primary_skill_category_id', 'primary_skill_id', 'priority_id',
                'probability_id', 'role_id',
                'supervisor_id', 'project_manager_id', 'coordinator_id', 'project_supervisor_id',
                'delivery_supervisor_id', 'assignee_id',
                'project_sponsor_id',
                'creator_id',
                'customer_id', 'delivery_manager_id', 'manager_id',
                'account_manager_unit_id',
                'supervisor_unit_id',
                'project_manager_unit_id',
                'coordinator_unit_id',
                'project_supervisor_unit_id',
                'delivery_supervisor_unit_id',
                'assignee_unit_id',
                'project_sponsor_unit_id', 'country_id', 'city_id',
                'creator_unit_id', 'hiring_manager_id', 'solution_competency_id',
                'delivery_manager_unit_id', 'hiring_manager_unit_id',
                'manager_unit_id',
                'planned_end_date', 'ped_in_the_end_of_Dec', 'ped_in_the_end_of_Jun', 'ped_at_the_end_of_month',
                'ped_at_the_begining_of_month', 'ped_is_last_day_of_month', 'ped_is_first_day_of_month',
                'job_function_id', 'location_id', 'emp_manager_id', 'employment_status_id', 'org_category_id',
                'emp_primary_skill_category_id', 'emp_primary_skill_id', 'primary_workload_type', 'production_category_id',
                'unit_id', 'unit_manager_id',
                'cur_month', 'cur_month_on_bench', 'cur_month_bench_bucket', 'next_month', 'next_month_on_bench',
                'next_month_bench_bucket',
                'third_month', 'third_month_on_bench', 'third_month_bench_bucket', 'pos_primary_skill_category_id',
                'pos_primary_skill_id', 'pos_manager_id', 'location_name', 'unit_name', 'bench_month', 'project_name',
                'workload_month', 'target_by_workload',
                'wkl_cur_month', 'wkl_next_1_month', 'wkl_next_2_month', 'wkl_next_plus_months', 'day', 'month', 'year',
                'wkl_target'
                                ]

    multicat_cols = ['staffing_locations', 'staffing_channels', 'project_coordinators',
                     'sales_sxecutives', 'container_locations_city_ids', 'container_locations_country_ids',
                     'staffing_locations_city_ids', 'staffing_locations_country_ids',
                     'container_locations_city', 'container_locations_country',
                     'staffing_locations_city', 'staffing_locations_country', 'seniority_level',
                     'sales_managers', 'program_managers', 'container_locations', 'container_staffing_coordinators',
                     'staffing_coordinators', 'supply_owners', 'demand_owners', 'pmc_roles', 'key_skills', 'languages']

    to_del = ['created', 'target', 'actual_start_date', 'max_date', 'date', 'last_assigned_date', 'employee_id',
              'position_id', 'last_staffing_status_update_date', 'planned_end_date', 'comment_id',
              'planned_start_date', 'timesheet_date', 'timesheet_date_min', 'timesheet_date_max', 'workload_start_date',
              'wkl_report_month', 'actual_end_date', 'proj_start_date', 'proj_end_date',
              'first_proposals_date', 'first_booking_date', 'first_onboarding_date', 'last_proposals_date',
              'last_booking_date', 'last_onboarding_date', 'eom_date_cur_month', 'eom_date_next_1_month',
              'eom_date_next_2_month'
              ]

    text_data_columns = []
    categorical_data_columns = []
    multicategorical_data_columns = []
    columns_to_del = []

    for col in text_cols:
        if col in list(df.columns):
            text_data_columns.append(col)

    for col in cat_cols:
        if col in list(df.columns):
            categorical_data_columns.append(col)

    for col in multicat_cols:
        if col in list(df.columns):
            multicategorical_data_columns.append(col)

    for col in to_del:
        if col in list(df.columns):
            columns_to_del.append(col)


    numeric_data_columns = list(df.columns.drop(np.concatenate((text_data_columns, categorical_data_columns,
                                            multicategorical_data_columns, columns_to_del))))

    return text_data_columns, categorical_data_columns, multicategorical_data_columns, numeric_data_columns


def cut_categoric_var(data, categorical_columns, multicategorical_columns, n=400):
    df = data.copy()
    #df.set_index('position_id', inplace=True)

    def cut_multicat(series):
        massive = []
        for lists in series:
            for element in lists:
                massive.append(element)
        df_cut = pd.DataFrame(massive)
        df_cut['index'] = df_cut.index
        df_cut = df_cut.groupby([0])['index'].agg(['count']).sort_values(by=['count'], ascending=False)
        df_cut['cut'] = df_cut['count'].map(lambda x: x >= n)
        df_cut = df_cut[df_cut['cut'] == True].index.values

        return df_cut

    for col in multicategorical_columns:
        cut_col = cut_multicat(df[col])
        df[col] = df[col].map(lambda x:
                              [el for el in x if (el in cut_col)]
                              )

    for col in categorical_columns:
        cut_col = df.groupby(col)['created'].agg(['count']).sort_values(
            by=['count'], ascending=False)
        cut_col['cut'] = cut_col['count'].map(lambda x: x >= n)
        cut_col = cut_col[cut_col['cut'] == True].index.values
        df[col] = df[col].map(lambda x: x if (x in cut_col) else '')

    return df