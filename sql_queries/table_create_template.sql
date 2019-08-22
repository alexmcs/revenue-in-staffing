create table wpm_anlt.epm_wpm_ds_staff_rampdown_probability_predictions (
	run_id bigint not null,
	employee_id varchar(50) not null,
	rampdown_probability_prediction float not null,
	date varchar(10) not null,
	date_of_processing timestamp not null
)



GRANT ALL PRIVILEGES ON wpm_anlt.epm_wpm_ds_staff_rampdown_probability_predictions TO "Auto_EPM-WPM_DS_Staff@epam.com"
--or SELECT instead of ALL PRIVILEGES


ALTER TABLE wpm_anlt.epm_wpm_ds_staff_rampdown_probability_predictions
ADD COLUMN rampdown_probability_prediction_30_days float not null;

UPDATE wpm_anlt.epm_wpm_ds_staff_workload_extension_predictions
SET date_of_processing = (TIMESTAMP WITH TIME ZONE 'epoch' + run_id * INTERVAL '1 second')


ALTER TABLE wpm_anlt.epm_wpm_ds_staff_rampdown_probability_predictions
DROP COLUMN rampdown_probability_prediction


DELETE FROM wpm_anlt.epm_wpm_ds_staff_rampdown_probability_predictions
WHERE condition;