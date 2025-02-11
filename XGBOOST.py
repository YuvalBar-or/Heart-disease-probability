import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

# Load the dataset
df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# Add noise only to features (X) to avoid modifying the target (y)
np.random.seed(42)
X_noisy = df.drop(columns=['target']) + np.random.normal(0, 0.1, df.drop(columns=['target']).shape)

# Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
X_train = X_noisy.iloc[train_df.index]  # Use noisy features for training
y_train = train_df['target']
X_test = X_noisy.iloc[test_df.index]   # Use noisy features for testing
y_test = test_df['target']

# XGBoost Model with reduced complexity to avoid overfitting
xgboost_model = xgb.XGBClassifier(
    max_depth=4,  # Reduce the depth of the trees to limit model complexity
    learning_rate=0.1,  # Learning rate is not too high, not too low
    n_estimators=50,  # Fewer trees to reduce complexity
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),  # Handling class imbalance
    reg_alpha=0.5,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
)

xgboost_model.fit(X_train, y_train)
xgb_pred = xgboost_model.predict(X_test)
xgb_pred_proba = xgboost_model.predict_proba(X_test)

# Print Performance Metrics
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))
auc = roc_auc_score(y_test, xgb_pred_proba[:, 1])
print(f"XGBoost AUC: {auc:.2f}")

# Feature Importance Visualization
feature_importances = xgboost_model.feature_importances_
features = X_train.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=features, color='darkorange')
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
