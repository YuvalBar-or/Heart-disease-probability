import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

# Decision Tree Model
decision_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=4, random_state=42)
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(X_test)
dt_pred_proba = decision_tree.predict_proba(X_test)

# Print Performance Metrics
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print(f"Decision Tree AUC: {roc_auc_score(y_test, dt_pred_proba[:, 1]):.2f}")

# Visualization of Decision Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=X_train.columns, class_names=["Class 0", "Class 1"])
plt.title("Decision Tree Structure")
plt.show()
