with dates as (
    select * from unnest(array{}) date
)
select d.date::date, p.*, t.timesheet_date
from dates d
left join
        (select bt, position_id, assignee_id, account_manager_id, actual_start_date, actual_end_date,
        created, creator_id, last_assigned_date,
        calc_utilization_group_id, hiring_manager_id, delivery_supervisor_id, comment_id,
        billing_type_id, container_project_billing_type_id, container_project_category_id, customer_id,
        customer_is_new, description, position_name, planned_start_date, planned_end_date,
        primary_skill_category_id, primary_skill_id, priority_id, probability_id, project_manager_id,
        remote_work_allowed, role_id, solution_competency_id, coordinator_id, staffing_locations,
        position_status, manager_id, staffing_channels, project_coordinators,
        sales_sxecutives, sales_managers, program_managers, staffing_coordinators, version,
        delivery_manager_id, supervisor_id, project_supervisor_id, staffing_status, project_sponsor_id,
        admin_project_agg, container_locations, container_name, container_type, container_staffing_coordinators,
        last_staffing_status_update_date, supply_owners, seniority_level, demand_owners, pmc_roles, gbu_id, customer_name
        from wpm_all.epm_staf_m_positions pos2
        where pos2.created >= date'2014-01-01' and staffing_status not in ('Assigned', 'Terminated',
        'On Hold', 'Closed') and billing_type_id = '1' and customer_id != '314427597' -- not EPAM Inc
        ) p on p.bt @> d.date::timestamp
left join
        (
        select min(timesheet_date) as timesheet_date, position_id
        from
            (
            select timesheet_date, bind_position_id as position_id
            from wpm_time_head.epm_time_worklog
            where billable = true
            ) temp
        group by position_id
        ) t on p.position_id::text = t.position_id