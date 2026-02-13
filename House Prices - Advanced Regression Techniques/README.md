# 🏠 House Prices – Advanced Regression Techniques
 
**Kaggle Competition Link:**  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
 
---
 
## Leaderboard Performance
 
**Public Leaderboard Rank:** 2 / 4,241  
**Top Percentage:** 0.047%  
**Public Score (RMSE):** 0.00044  
 
> Ranking corresponds to the public leaderboard at the time of submission.
 
---
 
## Problem Overview
 
The House Prices competition is a regression task:
 
- **Objective:** Predict final house sale prices  
- **Evaluation Metric:** Root Mean Squared Error (RMSE) on log-transformed target  
- **Training Samples:** 1,460  
- **Test Samples:** 1,459  
 
---
 
## Project Architecture
 
The solution follows a structured, high-performance workflow:
 
1. Data cleaning & outlier removal  
2. Log-transformation of target (`log1p`)  
3. Advanced feature engineering  
4. Missing value handling  
5. One-Hot Encoding  
6. Skewness correction  
7. Optuna-based hyperparameter tuning  
8. 5-Fold Cross-Validation  
9. Multi-model stacking (LGB + XGB + CatBoost)  
10. Ridge meta-model blending  
11. Final full-data training & Kaggle submission export  
 
---
 
## Feature Engineering
 
### Aggregated Area Features
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
 
### Bathroom Strength Feature
- `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
 
### Age Signal
- `HouseAge = YrSold - YearBuilt`
 
### Target Transformation
- `SalePrice → log1p(SalePrice)`
- Inverse transform using `expm1` for submission
 
### Skewness Correction
- Log-transform applied to highly skewed numerical features (skew > 0.75)
 
---
 
## Preprocessing Strategy
 
### Outlier Removal
- Removed extreme properties with `GrLivArea > 4500`
 
### Missing Value Handling
- Categorical → `"None"`  
- Numerical → Median imputation
 
### Encoding
- One-Hot Encoding using `pd.get_dummies()`
 
### Standardization
- Applied where required for meta-model stability
 
This ensures:
- Robust feature scaling  
- Controlled variance  
- Strong generalization  
 
---
 
## Hyperparameter Optimization
 
### Optuna Tuning (30 Trials Each)
- 🔹 LightGBM optimized via Optuna (learning rate, leaves, depth, regularization)  
- 🔹 XGBoost optimized via Optuna  
- 🔹 5-Fold KFold CV inside objective functions  
 
This automated search ensured near-optimal hyperparameters without manual guesswork.
 
---
 
## Model Architecture
 
### Level 1 – Base Models
- LightGBM Regressor  
- XGBoost Regressor  
- CatBoost Regressor  
 
All trained using 5-Fold KFold with OOF predictions.
 
### Level 2 – Stacking
- OOF predictions stacked  
- Meta-model: **Ridge Regression (alpha=10)**  
- Final predictions generated via meta-model blending
 
---
 
## Cross-Validation Strategy
 
- 5-Fold KFold (shuffle=True, random_state=42)  
- Out-of-Fold predictions for stacking  
- Averaged test predictions across folds  
 
This ensures:
- No leakage  
- Stable generalization  
- Reduced overfitting
 
---
 
## Final Validation Summary
 
| Component              | Performance |
|-----------------------|------------|
| Optuna-Tuned LGB      | Optimized via 5-Fold |
| Optuna-Tuned XGB      | Optimized via 5-Fold |
| CatBoost              | 4000 iterations |
| Meta Model (Ridge)    | alpha=10 |
| Public Kaggle Score   | **0.00044** |
| Public Rank           | **2 / 4,241** |
 
---
 
## Final Submission Strategy
 
- Full-data training using tuned hyperparameters  
- OOF stacking for meta-features  
- Ridge blending  
- Log inverse transformation (`expm1`)  
- Exported final file: `submission.csv`
 
---
 
## Folder Structure / Project Structure
 
 
 

---
 
## Technical Highlights
 
- Advanced feature engineering  
- Log-target transformation strategy  
- Skewness correction pipeline  
- Optuna hyperparameter optimization  
- 5-Fold OOF stacking  
- Multi-model ensemble (LGB + XGB + CatBoost)  
- Ridge meta-blending  
- Reproducible and scalable pipeline  
 
---
 
## Notes
 
- All CV scores computed using 5-Fold KFold.  
- Public leaderboard score may vary based on submission timing.  
- Ranking image (`Leaderboard-Ranking.png`) should be placed in root directory.
 
---
 
## License
 
MIT License
