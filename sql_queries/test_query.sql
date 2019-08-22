select run_date, period, employee_id, region, country, city, primary_skill, skill_practice,
solution_practice, predicted, capacity, model
from wpm_anlt.epm_wpm_utlz_prediction_user
where country in ('USA', 'Canada')
and run_date >= '2019-07-01'::date
and period in ('2019-07-01', '2019-08-01')
