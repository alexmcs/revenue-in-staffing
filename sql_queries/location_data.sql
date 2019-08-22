SELECT *, COALESCE(parent_location_id, '-999') as parent_location_id_clean
FROM wpm_head.epm_pmc_m_geographical_location