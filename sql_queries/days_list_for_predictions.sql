select date
from (
	select date, count(scope)
	from (
		select distinct date, scope
		from wpm_anlt.epm_wpm_ds_staff_rev_in_staf_{0}
        where date::date >= '{1}'::date
        and date::date <= '{2}'::date
		) d
	group by date
	) c
where count = 3
order by date