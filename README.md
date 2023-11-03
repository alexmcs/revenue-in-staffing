Project goals and objectives
================
The main goal of this project is to predict billable hours that will be reported at the end of the calendar month and accepted by clients.
Based on hours and rates, invoices are issued, which are converted into revenue.
Within the project, predictions are made for three calendar months (horizons): the current month, the next one and the following one.
Within the project, two independent approaches were used.
The main approach analyzed staffing data and predicted the number of hours for each open or active position that would be transformed into revenue (from positions).
The second approach used the data of the current utilization model, which predicted reportable hours tied to each employee. Then, through heuristics, it determined the positions to which these hours would be reported.

Business logic
-----------------
The following operations are performed within this project:

- Raw data from transactional systems are processed daily. The processing results are used as features for the models. Datasets assembly for training and inference take place based on daily snapshots.
- Predictions are made daily for four horizons: the current month, next 1 month, next 2 months, and next +. 
- A separate classifier is trained for each horizon. In the final predictions, the available monthly hours for each position are aggregated for positive classifier predictions. 
- To improve the quality, the following business cases are separated:
  - Open positions for which candidates are proposed within the staffing process.
  - Positions with an assigned employee. It is necessary to predict whether the client will accept billable hours.
  - Positions marked as 'no staffing required'. Predictions are needed for when billable hours will be submitted on them, which the client will accept.
- All models are retrained three times a month: at the beginning of the month, in the middle of the month, and at the end of the month. This approach ensures sufficient prediction quality.
- After each model retraining, hyperparameter tuning, validation, and quality metric verification are performed for the classifier. The main quality metric used is the Mean Absolute Error of aggregated hours, relative to the actual reported hours.

Script launch
---------------
The script_data_set_element_processing.py module is used to run scripts for feature processing, with parameters corresponding to a specific feature passed into it.
For running scripts related to model training and predictions, the script_modeling_and_predicting.py module is used.

To be done
--------------

- Add SHAP for explainability and analyzing issues with low-quality predictions if problem areas are identified.
- Switch from XGBoost to LightGBM. This will speed up model training, reduce memory consumption during data processing, improve handling of categorical variables, and obtain additional benefits due to the leaf-wise tree growth strategy.
- For tracking model versions and quality metrics, switch to using the MLFlow framework. This will enhance experiment tracking, model reproducibility, model packaging and deployment, and model versioning.
- Incorporate additional data into the model:
  - Position quality score: a metric that determines how well the required position attributes are filled, and how accurately and consistently the requirements are formulated.
  - Position skills complexity: for understanding whether the candidate requirements are standard, rare, or unique (unicorn-like).
  - Positions reason for opening (ramp-up, replacement, new business, technical, etc.)
