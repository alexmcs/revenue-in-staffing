with dates as (
    select * from unnest(array{0}) date
)
select sum(p.fte) as total_workload, d.date::date, p.position_id
from dates d
left join
        (
        select workload_start_date, position_id, fte, bt
        from wpm_all.epm_staf_m_monthly_workload
        where monthly_workload > 0 and position_id in {1}
        ) p on p.bt @> d.date::timestamp and d.date::date <= p.workload_start_date
group by date, position_id