select primary_skill_id, city_id, country_id, staffing_status, date, number_of_positions
from wpm_anlt.epm_wpm_ds_staff_active_pos_counting
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_active_pos_counting
            where date in {}
        ) q
    group by date
)