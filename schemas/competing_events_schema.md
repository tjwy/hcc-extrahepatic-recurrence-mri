# Adjudicated competing-event input

Provide exactly one row per analysis patient, in a CSV with:

- `MRI_id`: deidentified join key, matching the analysis dataset exactly.
- `analysis_time_months`: positive time from surgery to the applicable event or last disease assessment.
- `event_type`: 0 for censoring, 1 for first extrahepatic recurrence, 2 for death before EHR.

Use the frozen, adjudicated study dates. Do not infer a competing death merely from a later all-cause death after recurrence. The primary evaluator does not reconstruct event dates from unrelated recurrence endpoints.

The prediction dataset contains `Set` (`Train_CV` or `External_Test`) and `Model_CRM`, `Model_IFM`, `Clinical_only_Risk`, and `BCLC_Risk`. Development scores must be out of fold. These are continuous scores, not probabilities.

Risk groups use the development-score 86th percentile independently for each model, with the same numerical cutoff applied externally. This is exploratory post hoc stratification, not a validated surveillance interval or treatment rule.
