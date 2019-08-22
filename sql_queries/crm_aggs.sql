select *
from wpm_crm_anlt.epm_wpm_ds_staff_crm_{1}_rev_daily
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_crm_anlt.epm_wpm_ds_staff_crm_{1}_rev_daily
            where date in {0}
        ) q
    group by date
)