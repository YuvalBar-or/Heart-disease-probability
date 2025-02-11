import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

# Check class balance
print("Class balance in training set:\n", y_train.value_counts())
print("Class balance in test set:\n", y_test.value_counts())

# Random Forest Model with class_weight='balanced' to handle imbalanced classes
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
random_forest.fit(X_train, y_train)

# Predict and evaluate
rf_pred = random_forest.predict(X_test)
rf_pred_proba = random_forest.predict_proba(X_test)

# AUC on both training and testing data
rf_pred_train = random_forest.predict(X_train)
rf_pred_proba_train = random_forest.predict_proba(X_train)

print(f"Training AUC: {roc_auc_score(y_train, rf_pred_proba_train[:, 1]):.2f}")
print(f"Test AUC: {roc_auc_score(y_test, rf_pred_proba[:, 1]):.2f}")

# Print Classification Report
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Visualization
feature_importances = random_forest.feature_importances_
features = X_train.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=features, color='teal')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
