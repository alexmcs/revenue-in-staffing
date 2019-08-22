with dates as (
    select * from unnest(array{2}) date
)
select d.date::date, t.*
from dates d
left join
    (select
        project_uid_projection,
        pmc_user_id,
        duration,
        on_site,
        billable,
        changed,
        timesheet_date,
        activity_name,
        id,
        pmc_id,
        bind_position_id,
        bt
    from wpm_time_all.epm_time_worklog
    where
        timesheet_date::date >= date'{0}' and
        timesheet_date::date < date'{1}'
    ) t on t.bt @> d.date::timestamp