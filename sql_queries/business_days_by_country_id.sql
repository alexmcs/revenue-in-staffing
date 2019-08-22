select distinct calendar_date as "day"
from wpm_head.epm_pmc_m_calendar_day
where calendar_id = '{}'
and norm_hours > 0
order by calendar_date