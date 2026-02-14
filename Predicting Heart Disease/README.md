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

> Ranking corresponds to the public leaderboard at the time of submission.

---

## Problem Overview

The Heart Diseases competition is a binary classification task:

- **Objective:** Predict presence of heart disease (Absence / Presence)
- **Evaluation Metric:** Accuracy
- **Data Type:** Structured clinical tabular dataset
- **Approach:** High-capacity gradient boosting ensemble with heavy feature engineering

---

## Model Architecture Overview

| Component | Strategy |
|------------|----------|
| Validation | Stratified 5-Fold Cross-Validation |
| Core Models | XGBoost, LightGBM, CatBoost, HistGradientBoosting |
| Ensemble Method | Weighted Probability Blending |
| Overfitting Control | Early Stopping + Low Learning Rate |
| Feature Engineering | Clinical ratios + high-risk flags + interactions |
| Final Output | Averaged test predictions across folds |

---

## Cross-Validation Performance

| Model | Role in Ensemble | Weight | Key Strength |
|--------|-----------------|--------|--------------|
| XGBoost | Primary booster | 30% | Strong nonlinear modeling |
| LightGBM | Structural depth learner | 20% | Leaf-wise growth efficiency |
| CatBoost | Categorical-aware booster | 30% | Robust categorical handling |
| HistGradientBoosting | Regularized booster | 20% | Stable generalization |

**Overall OOF Accuracy:** ~0.95400  

---

## Feature Engineering Highlights

### Clinical Ratio Features
- `Chol_to_Age`
- `BP_to_MaxHR`
- `ST_Dep_Product`

### High-Risk Indicators
- `HighRisk_Thal`
- `HighRisk_Vessels`
- `Silent_Killer`

### Interaction Features
- `Age_Vessel_Interaction`
- `HR_Thal_Interaction`

### Stabilization Transforms
- `Log_Chol`
- `Log_BP`

These engineered features introduce nonlinear signal amplification without manual feature selection.

---

## Training Strategy

1. Apply feature engineering
2. Perform Stratified 5-Fold CV
3. Train 4 high-capacity boosting models per fold
4. Generate OOF predictions
5. Apply weighted blending
6. Average fold-based test predictions
7. Export final submission file

---

## Final Submission Strategy

- Probability threshold: 0.5  
- Predictions mapped back to competition label format  
- Exported file: `submission.csv`  

---

## Folder Structure / Project Structure

 ```
Predicting Heart Disease/
├── README.md
├── Output.csv
├── HeartDiseasess.py
├── Leaderboard-Ranking.png
└── Dataset/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## Technical Highlights

- Multi-boosting ensemble (XGB + LGB + CAT + HGB)
- Deep clinical interaction engineering
- Stratified K-Fold validation
- Early stopping for generalization control
- Weighted probability blending
- Out-of-fold evaluation tracking
- Fully reproducible pipeline

---

## Notes

- All validation results computed using Stratified 5-Fold Cross-Validation.
- Public leaderboard rank reflects submission time.
- Screenshot file (`Leaderboard-Ranking.png`) should be placed in project root directory.

---

## License

MIT License

 
