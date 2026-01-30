# Dataset Summary

This document provides a summary of all datasets available in the configs folder, including the number of features and any actionability constraints defined in their configuration files.

## Overview

- **Total Datasets**: 14
- **Datasets with Actionability Constraints**: 1
- **Datasets without Constraints**: 13

## Dataset Details

| Dataset Name | Number of Features | Actionability Constraints |
|--------------|-------------------|--------------------------|
| abalone | 11 | None |
| diabetes | 9 | Age: non_decreasing; Pregnancies: non_decreasing; DiabetesPedigreeFunction: no_change; BMI: actionable; Glucose: actionable; BloodPressure: actionable |
| ecoli | 8 | None |
| education_dataset | 4 | None |
| energy_dataset | 4 | None |
| fraud_detection_dataset | 5 | None |
| isolet | 618 | None |
| libras_move | 91 | None |
| mammography | 7 | None |
| satimage | 37 | None |
| scene | 295 | None |
| us_crime | 101 | None |
| yeast_ml8 | 104 | None |

## Datasets with Actionability Constraints

### diabetes (9 features)
- **Age**: non_decreasing
- **Pregnancies**: non_decreasing
- **DiabetesPedigreeFunction**: no_change
- **BMI**: actionable
- **Glucose**: actionable
- **BloodPressure**: actionable

---

**Note**: iris dataset has a config.yaml file but no dataset.csv file, so it's not included in the table. If a dataset.csv is available for iris, it would also be part of this analysis.

## Constraint Types Explanation

Actionability constraints define how features can be modified when generating counterfactual examples:

- **actionable**: The feature can be freely modified to any value
- **non_decreasing**: The feature value can only increase or stay the same
- **non_increasing**: The feature value can only decrease or stay the same
- **no_change**: The feature value must remain unchanged

## Feature Count Statistics

| Metric | Value |
|--------|-------|
| Minimum features | 4 (education_dataset, energy_dataset) |
| Maximum features | 618 (isolet) |
| Average features | ~91 |
| Median features | ~49 |

## Notes

- All feature counts include the target column
- Actionability constraints are defined in the `methods._default.actionability` section of each config.yaml file
- The **iris** dataset was requested but does not have a dataset.csv file (only a config.yaml file exists), so it's not included in the summary table
