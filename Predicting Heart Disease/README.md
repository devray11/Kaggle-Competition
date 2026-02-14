# Heart Diseases Detection – Playground Series S6E2

Kaggle Competition Link:  
https://www.kaggle.com/competitions/playground-series-s6e2

---

## Leaderboard Performance

<p align="center">
  <img src="Leaderboard-Ranking.png" alt="Heart Disease Leaderboard Ranking" width="1000"/>
</p>

**Public Leaderboard Rank:** 2 / 2,491  
**Top Percentage:** 0.08%  
**Public Score (Accuracy):** 0.95400  
**Total Submissions:** —  

> Ranking corresponds to the public leaderboard at the time of submission.

---

## Problem Overview

The Heart Disease competition is a binary classification task:

- **Objective:** Predict presence of heart disease  
  (0 = Absence, 1 = Presence)
- **Evaluation Metric:** Accuracy
- **Training Data:** Structured clinical tabular dataset
- **Test Data:** Unlabeled evaluation set

---

## Project Architecture

The solution follows a performance-driven boosting ensemble workflow:

1. Heavy clinical feature engineering  
2. Stratified 5-Fold Cross-Validation  
3. Multi-model gradient boosting ensemble  
4. Out-of-fold (OOF) prediction tracking  
5. Weighted probability blending  
6. Full test-set prediction averaging  
7. Final Kaggle submission export  

---

## Feature Engineering

Advanced medical interaction features were engineered to increase model depth.

### Clinical Ratio Features
- `Chol_to_Age`
- `BP_to_MaxHR`
- `ST_Dep_Product`

### High-Risk Boolean Flags
- `HighRisk_Thal`
- `HighRisk_Vessels`
- `Silent_Killer`

### Interaction Features
- `Age_Vessel_Interaction`
- `HR_Thal_Interaction`

### Distribution Stabilization
- `Log_Chol`
- `Log_BP`

These transformations introduce nonlinear signals without manual feature selection.

---

## Model Strategy

The solution uses a high-capacity gradient boosting ensemble:

- **XGBoost** (Deep trees, 3000 estimators)
- **LightGBM** (High leaf complexity)
- **CatBoost** (Native categorical handling)
- **HistGradientBoosting** (Sklearn histogram-based boosting)

All models trained with:
- Early stopping
- Stratified 5-Fold CV
- High estimator counts
- Low learning rates

---

## Ensembling Strategy

### Weighted Blending per Fold

Dynamic weighted blend:

- XGBoost → 30%
- LightGBM → 20%
- CatBoost → 30%
- HistGradientBoosting → 20%

Blending is performed:
- On validation fold (OOF tracking)
- On test predictions (averaged across folds)

---

## Cross-Validation Framework

- **Validation Method:** Stratified 5-Fold CV
- **OOF Accuracy:** ~0.95400
- **Evaluation Metric:** Accuracy

This ensures:
- Stable generalization
- Low variance
- No leakage

---

## Final Submission Strategy

- Averaged predictions across folds
- Probability threshold at 0.5
- Label mapping restored to competition format
- Exported file: `submission.csv`

---

## Folder Structure / Project Structure



---

## Technical Highlights

- Multi-boosting ensemble (XGB + LGB + CAT + HGB)
- Heavy clinical interaction engineering
- Stratified K-Fold cross-validation
- Early stopping for overfitting control
- Out-of-fold accuracy tracking
- Weighted probability blending
- Fully reproducible training pipeline

---

## Notes

- All validation scores are computed using Stratified 5-Fold CV.
- Public leaderboard position reflects time of submission.
- Ranking screenshot (`Leaderboard-Ranking.png`) should be placed in this directory.

---

## License

MIT License
