with dates as (
    select * from unnest(array{}) date
)
select d.date::date, e.employee_id, e.unit_id
from dates d
left join
        (select bt, employee_id, unit_id
        from wpm_all.epm_staf_m_employees
        where unit_id is not null
        ) e on e.bt @> d.date::timestamp