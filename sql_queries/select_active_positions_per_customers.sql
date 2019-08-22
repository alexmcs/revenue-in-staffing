select primary_skill_id, city_id, country_id, customer_id, date, number_of_positions
from wpm_anlt.epm_wpm_ds_staff_active_positions_per_customer
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_active_positions_per_customer
            where date in {}
        ) q
    group by date
)