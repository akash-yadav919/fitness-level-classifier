# 🏋️ Gym Fitness Level Predictor
 
A machine learning pipeline that predicts a gym member's **fitness tier** (Beginner / Intermediate / Advanced) using personal stats, workout habits, and nutrition data — with a bonus regression model that estimates the raw fitness score.
 
---
 
## 📁 Project Structure
 
```
├── gym_fitness_model.py       # Main ML pipeline
├── Gym_Fitness_Data.csv       # Input dataset (required)
├── fitness_model_dashboard.png  # Output: evaluation charts
└── README.md
```
 
---
 
## 🚀 Features
 
- **Classification** — Predicts fitness tier using a Random Forest Classifier
- **Regression** — Estimates raw fitness score using a Random Forest Regressor
- **Feature Engineering** — Extracts workout type, computes BMI, calculates membership seniority
- **Evaluation Dashboard** — Confusion matrix, feature importances, and actual vs predicted scatter plot
- **Prediction Function** — Predict fitness tier for any new gym member with a single function call
 
---
 
## 📦 Requirements
 
Install dependencies via pip:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
 
| Library | Version (recommended) |
|---|---|
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |
| scikit-learn | ≥ 1.2 |
 
---
 
## 📊 Dataset
 
The script expects a file named `Gym_Fitness_Data.csv` in the same directory with the following columns:
 
| Column | Type | Description |
|---|---|---|
| `member_id` | string | Member ID with workout type (e.g. `ID_777 Powerlifting`) |
| `weight_kg` | float | Body weight in kilograms |
| `height_cm` | float | Height in centimetres |
| `calories_intake` | string | Daily calorie intake (e.g. `3200 kcal`) |
| `protein_g` | float | Daily protein intake in grams |
| `workout_days_week` | int | Number of workout days per week |
| `experience_level` | string | One of: `Beginner`, `Intermediate`, `Advanced`, `Elite` |
| `joining_year` | int | Year the member joined |
| `fat_percentage` | float | Body fat percentage |
| `fitness_score` | float | Numeric fitness score (used as regression target) |
 
> Missing values in numeric columns are filled with the column median automatically.
 
---
 
## ⚙️ How It Works
 
### 1. Data Cleaning
- Strips unit strings from `calories_intake` (e.g. removes `kcal`)
- Fills missing values with column medians
 
### 2. Feature Engineering
| Engineered Feature | Logic |
|---|---|
| `workout_type` | Extracted from the second word of `member_id` |
| `experience_level` | Mapped to integers: Beginner=0, Intermediate=1, Advanced=2, Elite=3 |
| `member_seniority` | `2026 - joining_year` |
| `bmi` | `weight_kg / (height_cm / 100)²` |
 
### 3. Target Variable
- `fitness_score` is binned into 3 equal-frequency tiers using `pd.qcut`:
  - **Beginner** — bottom third
  - **Intermediate** — middle third
  - **Advanced** — top third
 
### 4. Models
| Model | Task | Estimators | Max Depth |
|---|---|---|---|
| `RandomForestClassifier` | Fitness tier prediction | 200 | 15 |
| `RandomForestRegressor` | Raw fitness score estimation | 200 | None |
 
---
 
## ▶️ Usage
 
```bash
python gym_fitness_model.py
```
 
### Predicting for a new member
 
Use the built-in `predict_fitness_tier()` function:
 
```python
sample = {
    'member_id'        : 'ID_777 Powerlifting',
    'weight_kg'        : 85.0,
    'height_cm'        : 180,
    'calories_intake'  : '3200 kcal',
    'protein_g'        : 180,
    'workout_days_week': 5,
    'experience_level' : 'Advanced',
    'joining_year'     : 2023,
    'fat_percentage'   : 15.0,
}
 
result = predict_fitness_tier(clf_model, sample)
print(result)  # → 'Advanced'
```
 
---
 
## 📈 Output
 
### Console
```
● Classification Metrics
  Accuracy   : 0.9XXX
  Precision  : 0.9XXX
  Recall     : 0.9XXX
  F1 Score   : 0.9XXX
  CV F1 (5-fold): 0.9XXX ± 0.0XXX
 
● Per-Class Report
              precision    recall  f1-score
  Beginner       ...
  Intermediate   ...
  Advanced       ...
 
● Regression Metrics
  R² Score  : 0.9XXX
  RMSE      : X.XX points
 
● Sample Prediction
  Result    : Advanced
```
 
### Dashboard (`fitness_model_dashboard.png`)
 
| Panel | Description |
|---|---|
| Confusion Matrix | Predicted vs actual fitness tier |
| Feature Importances | Top 12 features ranked by importance |
| Actual vs Predicted | Regression scatter plot with R² and RMSE |
 
---
 
## 🛠️ Key Fixes from Original
 
- **Regression model** was incorrectly using `RandomForestClassifier` — replaced with `RandomForestRegressor`
- **`predict_fitness_tier()`** was referencing dropped columns and skipping BMI engineering — fully rewritten
- **Stratified split** added to preserve class balance across train/test sets
- **Column alignment** in prediction now uses captured `model_columns` for exact match with training
 
---
 
## 📝 Notes
 
- The `fitness_tier` labels (Beginner / Intermediate / Advanced) are derived from the data distribution, not fixed thresholds — boundaries may shift with different datasets
- `experience_level` encoding assumes the four known labels; unknown values default to `0` (Beginner)
- The dashboard PNG is saved automatically after every run
 
---
 
## 📄 License
 
This project is for educational and personal use.
