# Dataset Summary

This document provides a summary of all datasets available in the configs folder, including the number of features and any actionability constraints defined in their configuration files.

## Overview

- **Total Datasets**: 18
- **Datasets with Actionability Constraints**: 6
- **Datasets without Constraints**: 12

## Dataset Details

| Dataset Name | Number of Features | Number of Samples | Number of Classes | Actionability Constraints |
|--------------|-------------------:|------------------:|------------------:|--------------------------|
| abalone | 10 | 4,177 | 2 | None |
| bank_marketing | 20 | 41,188 | 2 | Yes (see details below) |
| customer_personality_analysis | 28 | 2,240 | 2 | Yes (see details below) |
| diabetes | 8 | 768 | 2 | Yes (see details below) |
| ecoli | 7 | 336 | 2 | None |
| education_dataset | 3 | 1,000 | 2 | None |
| energy_dataset | 3 | 1,000 | 2 | None |
| healthcare_dataset_stroke | 11 | 5,110 | 2 | Yes (see details below) |
| iris | 4 | 150 | 3 | None |
| isolet | 617 | 7,797 | 2 | None |
| libras_move | 90 | 360 | 2 | None |
| mammography | 6 | 11,183 | 2 | None |
| obesity_levels | 16 | 2,111 | 7 | Yes (see details below) |
| satimage | 36 | 6,435 | 2 | None |
| scene | 294 | 2,407 | 2 | None |
| telco_customer_churn | 20 | 7,043 | 2 | Yes (see details below) |
| us_crime | 100 | 1,994 | 2 | None |
| yeast_ml8 | 103 | 2,417 | 2 | None |

## Datasets with Actionability Constraints

### bank_marketing (20 features)
- **age**: non_decreasing
- **job**: actionable
- **marital**: actionable
- **education**: actionable
- **default**: no_change
- **housing**: actionable
- **loan**: actionable
- **contact**: actionable
- **month**: actionable
- **day_of_week**: actionable
- **duration**: non_decreasing
- **campaign**: non_decreasing
- **pdays**: non_decreasing
- **previous**: non_decreasing
- **poutcome**: actionable
- **emp.var.rate**: actionable
- **cons.price.idx**: actionable
- **cons.conf.idx**: actionable
- **euribor3m**: actionable
- **nr.employed**: actionable

### customer_personality_analysis (28 features)
- **ID**: no_change
- **Year_Birth**: no_change
- **Education**: actionable
- **Marital_Status**: actionable
- **Income**: actionable
- **Kidhome**: non_decreasing
- **Teenhome**: non_decreasing
- **Dt_Customer**: no_change
- **Recency**: non_decreasing
- **MntWines**: actionable
- **MntFruits**: actionable
- **MntMeatProducts**: actionable
- **MntFishProducts**: actionable
- **MntSweetProducts**: actionable
- **MntGoldProds**: actionable
- **NumDealsPurchases**: actionable
- **NumWebPurchases**: actionable
- **NumCatalogPurchases**: actionable
- **NumStorePurchases**: actionable
- **NumWebVisitsMonth**: actionable
- **AcceptedCmp3**: no_change
- **AcceptedCmp4**: no_change
- **AcceptedCmp5**: no_change
- **AcceptedCmp1**: no_change
- **AcceptedCmp2**: no_change
- **Complain**: no_change
- **Z_CostContact**: no_change
- **Z_Revenue**: no_change
- **Response**: no_change

### diabetes (8 features)
- **Age**: non_decreasing
- **Pregnancies**: non_decreasing
- **DiabetesPedigreeFunction**: no_change
- **BMI**: actionable
- **Glucose**: actionable
- **BloodPressure**: actionable

### healthcare_dataset_stroke (11 features)
- **age**: non_decreasing
- **gender**: no_change
- **hypertension**: no_change
- **heart_disease**: no_change
- **ever_married**: no_change
- **work_type**: actionable
- **Residence_type**: actionable
- **avg_glucose_level**: actionable
- **bmi**: actionable
- **smoking_status**: actionable

### obesity_levels (16 features)
- **Age**: non_decreasing
- **Gender**: no_change
- **Height**: no_change
- **Weight**: actionable
- **CALC**: actionable
- **FAVC**: actionable
- **FCVC**: actionable
- **NCP**: actionable
- **SCC**: actionable
- **SMOKE**: actionable
- **CH2O**: actionable
- **family_history_with_overweight**: no_change
- **FAF**: actionable
- **TUE**: actionable
- **CAEC**: actionable
- **MTRANS**: actionable

### telco_customer_churn (20 features)
- **customerID**: no_change
- **gender**: no_change
- **SeniorCitizen**: no_change
- **Partner**: actionable
- **Dependents**: actionable
- **tenure**: non_decreasing
- **PhoneService**: actionable
- **MultipleLines**: actionable
- **InternetService**: actionable
- **OnlineSecurity**: actionable
- **OnlineBackup**: actionable
- **DeviceProtection**: actionable
- **TechSupport**: actionable
- **StreamingTV**: actionable
- **StreamingMovies**: actionable
- **Contract**: actionable
- **PaperlessBilling**: actionable
- **PaymentMethod**: actionable
- **MonthlyCharges**: actionable
- **TotalCharges**: actionable

---

## Constraint Types Explanation

Actionability constraints define how features can be modified when generating counterfactual examples:

- **actionable**: The feature can be freely modified to any value
- **non_decreasing**: The feature value can only increase or stay the same
- **non_increasing**: The feature value can only decrease or stay the same
- **no_change**: The feature value must remain unchanged

## Feature Count Statistics

| Metric | Value |
|--------|-------:|
| Minimum features | 3 (education_dataset, energy_dataset) |
| Maximum features | 617 (isolet) |
| Average features | ~75.78 |
| Median features | 16 |

## Notes

- Feature counts exclude the target column (the target/label is not considered a feature)
- Actionability constraints are defined in the `methods._default.actionability` section of each config.yaml file
- The **iris** dataset information (150 samples, 4 features, 3 classes) is from scikit-learn's built-in iris dataset
