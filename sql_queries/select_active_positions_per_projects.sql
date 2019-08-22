select primary_skill_id, city_id, country_id, container_name, date, number_of_positions
from wpm_anlt.epm_wpm_ds_staff_active_positions_per_projects
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_active_positions_per_projects
            where date in {}
        ) q
    group by date
)