select p.position_id::text, p.assignee_id, w.monthly_workload
from (
	select assignee_id, position_id
	from wpm_head.epm_staf_m_positions
	where assignee_id in {1}
	)p
left join wpm_head.epm_staf_m_monthly_workload w
	on p.position_id = w.position_id
where workload_start_date <= '{0}'::date
and workload_end_date >= '{0}'::date
and monthly_workload > 0