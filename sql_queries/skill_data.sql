select primary_skill_id, primary_skill_name, category_id as primary_skill_category_id, category_name as primary_skill_category_name
from wpm_head.epm_staf_ec_upsa_primary_skill
where is_primary = true and is_active = true