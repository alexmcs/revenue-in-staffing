select position_id, version, date
from wpm_anlt.epm_wpm_ds_staff_position_version_info
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_position_version_info
        ) q
    group by date
)