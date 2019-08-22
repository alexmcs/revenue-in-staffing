select position_id, weight_adjusted, urgency
from wpm_all.wpm_staffing_attention_position_weight_online
where _bt$@>'{}'::timestamp