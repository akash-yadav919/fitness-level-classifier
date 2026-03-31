import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
df = pd.read_csv("Code (AI & ML)\Gym_Fitness_Data.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(3))

# ─────────────────────────────────────────────
# 2. Data Cleaning
# ─────────────────────────────────────────────
def strip_unit(df, column, unit):
    """Remove a unit string from a column and convert to numeric."""
    df[column] = (
        df[column].astype(str)
        .str.replace(unit, '', regex=False)
        .str.strip()
        .replace('null', np.nan)
    )
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

df = strip_unit(df, 'calories_intake', 'kcal')

# Fill missing values with column median
for col in ['calories_intake', 'weight_kg', 'height_cm', 'protein_g', 'fat_percentage']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────
# Extract workout type from member_id (e.g. "ID_777 Powerlifting" → "Powerlifting")
df['workout_type'] = df['member_id'].apply(
    lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else 'General'
)
df.drop(columns=['member_id'], inplace=True)

# Encode experience level
exp_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2, 'Elite': 3}
df['experience_level'] = df['experience_level'].map(exp_map).fillna(0).astype(int)

# Member seniority (years since joining)
current_year = 2026
df['member_seniority'] = current_year - df['joining_year']
df.drop(columns=['joining_year'], inplace=True)

# BMI as an extra feature
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

print("\nFeature engineering complete. Columns:", df.columns.tolist())

# ─────────────────────────────────────────────
# 4. Target Variable Creation
# ─────────────────────────────────────────────
df['fitness_tier'] = pd.qcut(
    df['fitness_score'], q=3,
    labels=['Beginner', 'Intermediate', 'Advanced']
)
y_regression = df['fitness_score'].copy()

# ─────────────────────────────────────────────
# 5. Encoding & Scaling
# ─────────────────────────────────────────────
numerical_cols = [
    'weight_kg', 'height_cm', 'calories_intake', 'protein_g',
    'workout_days_week', 'experience_level', 'fat_percentage',
    'member_seniority', 'bmi'
]

df_encoded = pd.get_dummies(
    df.drop(columns=['fitness_score']),
    columns=['workout_type'],
    drop_first=True
)

X = df_encoded.drop(columns=['fitness_tier'])
y_classification = df_encoded['fitness_tier']

# Scale only the numerical columns that exist in X
num_cols_present = [c for c in numerical_cols if c in X.columns]
scaler = StandardScaler()
X[num_cols_present] = scaler.fit_transform(X[num_cols_present])

# Store column order for prediction
model_columns = X.columns.tolist()

# ─────────────────────────────────────────────
# 6. Classification Model
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

clf_model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1,
    max_depth=15, min_samples_leaf=2
)
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)

# Cross-validation
cv_scores = cross_val_score(clf_model, X, y_classification, cv=5, scoring='f1_weighted')

# ─────────────────────────────────────────────
# 7. Regression Model  (FIXED: use Regressor, not Classifier)
# ─────────────────────────────────────────────
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg_model.fit(X_reg_train, y_reg_train)
y_pred_reg = reg_model.predict(X_reg_test)

rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
r2   = r2_score(y_reg_test, y_pred_reg)

# ─────────────────────────────────────────────
# 8. Prediction Function
# ─────────────────────────────────────────────
def predict_fitness_tier(model, input_data: dict) -> str:
    """Predict fitness tier for a new gym member."""
    row = input_data.copy()

    # Parse calories if passed as string
    if isinstance(row.get('calories_intake'), str):
        row['calories_intake'] = float(row['calories_intake'].replace('kcal', '').strip())

    # Feature engineering (mirrors training pipeline)
    row['workout_type'] = row.pop('member_id').split(' ')[1] \
        if len(row.get('member_id', '').split(' ')) > 1 else 'General'
    row['member_seniority'] = current_year - row.pop('joining_year')
    row['experience_level']  = exp_map.get(row['experience_level'], 0)
    row['bmi'] = row['weight_kg'] / ((row['height_cm'] / 100) ** 2)

    input_df = pd.DataFrame([row])
    input_df = pd.get_dummies(input_df, columns=['workout_type'], drop_first=False)

    # Align columns with training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    input_df[num_cols_present] = scaler.transform(input_df[num_cols_present])

    return model.predict(input_df)[0]

# ─────────────────────────────────────────────
# 9. Print Results
# ─────────────────────────────────────────────
SEP = "─" * 70
print(f"\n{SEP}")
print("  MODEL EVALUATION RESULTS")
print(SEP)

print("\n● Classification Metrics")
print(f"  Accuracy   : {accuracy_score(y_test, y_pred_clf):.4f}")
print(f"  Precision  : {precision_score(y_test, y_pred_clf, average='weighted'):.4f}")
print(f"  Recall     : {recall_score(y_test, y_pred_clf, average='weighted'):.4f}")
print(f"  F1 Score   : {f1_score(y_test, y_pred_clf, average='weighted'):.4f}")
print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n● Per-Class Report")
print(classification_report(y_test, y_pred_clf, target_names=['Beginner', 'Intermediate', 'Advanced']))

print("\n● Regression Metrics (fitness score prediction)")
print(f"  R² Score  : {r2:.4f}")
print(f"  RMSE      : {rmse:.2f} points")

# Sample prediction
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
predicted_tier = predict_fitness_tier(clf_model, sample)
print(f"\n● Sample Prediction")
print(f"  Input  : Powerlifting, 85 kg, 180 cm, 3200 kcal, 180g protein, 5 days/week, Advanced")
print(f"  Result : {predicted_tier}")
print(SEP)

# ─────────────────────────────────────────────
# 10. Visualizations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Gym Fitness Model — Evaluation Dashboard", fontsize=15, fontweight='bold')

# — Confusion Matrix —
cm = confusion_matrix(y_test, y_pred_clf, labels=['Beginner', 'Intermediate', 'Advanced'])
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
    xticklabels=['Beginner', 'Intermediate', 'Advanced'],
    yticklabels=['Beginner', 'Intermediate', 'Advanced']
)
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# — Top Feature Importances —
importances = pd.Series(clf_model.feature_importances_, index=model_columns)
top_features = importances.nlargest(12).sort_values()
top_features.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Top 12 Feature Importances')
axes[1].set_xlabel('Importance')

# — Actual vs Predicted (Regression) —
axes[2].scatter(y_reg_test, y_pred_reg, alpha=0.4, color='coral', edgecolors='k', linewidths=0.3)
lims = [min(y_reg_test.min(), y_pred_reg.min()), max(y_reg_test.max(), y_pred_reg.max())]
axes[2].plot(lims, lims, 'k--', linewidth=1.2, label='Perfect fit')
axes[2].set_title(f'Regression: Actual vs Predicted\nR²={r2:.3f}, RMSE={rmse:.2f}')
axes[2].set_xlabel('Actual Fitness Score')
axes[2].set_ylabel('Predicted Fitness Score')
axes[2].legend()

plt.tight_layout()
plt.savefig("fitness_model_dashboard.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nDashboard saved as 'fitness_model_dashboard.png'")