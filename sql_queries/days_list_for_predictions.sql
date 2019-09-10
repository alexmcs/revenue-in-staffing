select distinct(date)
from wpm_anlt.epm_wpm_ds_staff_{0}_probability_predictions
where date::date >= '{1}'::date and date::date <= '{2}'::date
order by date