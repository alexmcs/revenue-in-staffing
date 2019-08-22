select primary_skill_id, city_id, country_id, staffing_status, number_of_proposals, date
from wpm_anlt.epm_wpm_ds_staff_proposals_counting
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_proposals_counting
            where date in {}
        ) q
    group by date
)