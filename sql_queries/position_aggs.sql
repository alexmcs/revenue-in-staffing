select workload_start_date, gbu_id, container_name as project_name, customer_name, date, sum_fte
from wpm_anlt.epm_wpm_ds_staff_fte_wkl_counting
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_fte_wkl_counting
            where date in {0}
        ) q
    group by date
)