select primary_skill_id, location_id, total_hours, in_trip_hours, billable_hours, date
from wpm_anlt.epm_wpm_ds_staff_time_reporting
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_time_reporting
            where date in {}
        ) q
    group by date
)