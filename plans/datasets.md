# Dataset Summary

This document provides a summary of all datasets available in the configs folder, including the number of features and any actionability constraints defined in their configuration files.

## Overview

- **Total Datasets**: 14
- **Datasets with Actionability Constraints**: 1
- **Datasets without Constraints**: 13

## Dataset Details

| Dataset Name | Number of Features | Number of Samples | Number of Classes | Actionability Constraints |
|--------------|-------------------|------------------|------------------|--------------------------|
| abalone | 11 | 4,177 | 2 | None |
| diabetes | 8 | 768 | 2 | Age: non_decreasing; Pregnancies: non_decreasing; DiabetesPedigreeFunction: no_change; BMI: actionable; Glucose: actionable; BloodPressure: actionable |
| ecoli | 8 | 336 | 2 | None |
| education_dataset | 4 | 1,000 | 2 | None |
| energy_dataset | 4 | 1,000 | 2 | None |
| fraud_detection_dataset | 5 | 1,000 | 2 | None |
| iris | 5 | 150 | 3 | None |
| isolet | 618 | 7,797 | 2 | None |
| libras_move | 91 | 360 | 2 | None |
| mammography | 7 | 11,183 | 2 | None |
| satimage | 37 | 6,435 | 2 | None |
| scene | 295 | 2,407 | 2 | None |
| us_crime | 101 | 1,994 | 2 | None |
| yeast_ml8 | 103 | 2,417 | 2 | None |

## Datasets with Actionability Constraints

### diabetes (8 features)
- **Age**: non_decreasing
- **Pregnancies**: non_decreasing
- **DiabetesPedigreeFunction**: no_change
- **BMI**: actionable
- **Glucose**: actionable
- **BloodPressure**: actionable

---

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
- The **iris** dataset information (150 samples, 4 features, 3 classes) is from scikit-learn's built-in iris dataset
