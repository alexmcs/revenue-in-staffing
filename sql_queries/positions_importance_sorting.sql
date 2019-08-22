select pos.position_id, pos.billing_type_id, pos.customer_id, pos.created, pos.planned_start_date, pos.planned_end_date,
pos.staffing_channels, pos.staffing_status, pos.container_name, pos.seniority_level,
cust.sum as number_of_positions_per_customer, cont.sum as number_of_positions_per_container
from wpm_head.epm_staf_m_positions pos
left join (
	select customer_id, sum(number_of_positions)
	from wpm_anlt.epm_wpm_ds_staff_active_positions_per_customer
	where run_id = (
		select max(run_id)
		from wpm_anlt.epm_wpm_ds_staff_active_positions_per_customer
	)
	group by customer_id
) cust on cust.customer_id = pos.customer_id
left join (
	select container_name, sum(number_of_positions)
	from wpm_anlt.epm_wpm_ds_staff_active_positions_per_projects
	where run_id = (
		select max(run_id)
		from wpm_anlt.epm_wpm_ds_staff_active_positions_per_projects
	)
	group by container_name
) cont on cont.container_name = pos.container_name
left join
	(select *
	from wpm_head.epm_staf_m_monthly_workload
	where workload_start_date = '{0}') w on pos.position_id = w.position_id
where pos.staffing_status not in ('Assigned', 'Terminated',
        'On Hold', 'Closed') and pos.billing_type_id = '1' and pos.customer_id != '314427597' -- not EPAM Inc