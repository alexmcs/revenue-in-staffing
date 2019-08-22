select primary_skill_id, city_id, country_id, number_of_employees, date
from wpm_anlt.epm_wpm_ds_staff_on_bench_counter
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_on_bench_counter
            where date in {}
        ) q
    group by date
)