select project_id, description as proj_description, end_date as proj_end_date,
is_billable as proj_is_billable, is_using_template,
start_date as proj_start_date, project_code
from wpm_head.epm_pmc_m_project