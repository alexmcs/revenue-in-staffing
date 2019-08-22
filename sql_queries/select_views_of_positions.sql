select count(*) as views, rv.time::date as date, entity_id, meta
from wpm_all.epm_staf_m_recent_view_item rv
where entity_type = 'FtsPosition'
group by date, entity_id, meta