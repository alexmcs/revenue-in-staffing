select position_id, billing_type_name
from wpm_head.epm_staf_m_positions
where position_id in (
    select *
    from unnest(array{}) p
	)