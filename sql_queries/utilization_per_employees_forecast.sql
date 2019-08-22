select employee_id, (reported+predicted) as predicted_hours, ext_wl, pos.position_id, pos.monthly_workload
from wpm_anlt.epm_wpm_utlz_prediction_user u
	left join
		(
		select p.position_id, p.assignee_id, w.monthly_workload
		from wpm_head.epm_staf_m_positions p
			left join
			(select *
			from wpm_head.epm_staf_m_monthly_workload
			where workload_start_date = '{1}') w on p.position_id = w.position_id
		where billing_type_id = '1' and w.monthly_workload_id is not null and w.monthly_workload > 0
		) pos on u.employee_id::text = pos.assignee_id
where model = 'ext' and run_date::date = '{0}' and period='{1}' --and ext_wl = 0 and p.position_id is null