select date, position_id, staffing_date_prediction
from wpm_anlt.epm_wpm_ds_staff_staffing_date_predictions
where run_id in (
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_staffing_date_predictions
        ) q
    group by date
) and date in (
    select *
    from unnest(array{}) dates
)