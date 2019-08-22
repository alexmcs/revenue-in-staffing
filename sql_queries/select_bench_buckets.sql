select primary_skill_id, location_id, bench_buckets, date, number_of_employees
from wpm_anlt.epm_wpm_ds_staff_bench_buckets
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_bench_buckets
            where date in {}
        ) q
    group by date
)