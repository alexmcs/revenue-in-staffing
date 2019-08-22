select external_id, code as container, customer_code as account
from wpm_head.epm_staf_m_container
where external_id in (
    select *
    from unnest(array{}) p
	)