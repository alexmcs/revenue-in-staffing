select proposal_id, position_id, staffing_status, applicant, date
from wpm_anlt.epm_wpm_ds_staff_current_proposals_per_positions
where run_id in
(
    select max(q.run_id) as run_id
    from
        (
            select *
            from wpm_anlt.epm_wpm_ds_staff_current_proposals_per_positions
        ) q
    group by date
)