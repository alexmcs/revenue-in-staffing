select position_id, date, workload_start_date
from wpm_anlt.epm_wpm_ds_staff_wkl_start_date
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_wkl_start_date
            where date in {}
        ) q
    group by date
)