select DISTINCT ON (position_id) position_id, time
from wpm_all.epm_staf_m_proposals
where staffing_status = 'Onboarding'
order by position_id, time desc