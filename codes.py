import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Create a more realistic synthetic network traffic dataset
np.random.seed(42)

# Set up a more realistic scenario
num_normal = 10000  # 50% normal traffic
num_anomalies = 10000  # 50% anomalies 

# Normal traffic data with some variability
normal_data = pd.DataFrame({
    'Timestamp': pd.date_range(start='2025-02-01', periods=num_normal, freq='S'),
    'Source_IP': [f"192.168.{np.random.randint(1, 5)}.{np.random.randint(1, 255)}" for _ in range(num_normal)],
    'Destination_IP': [f"10.0.{np.random.randint(1, 10)}.{np.random.randint(1, 255)}" for _ in range(num_normal)],
    'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_normal, p=[0.7, 0.25, 0.05]),
    # Create more variance in normal traffic
    'Packets_per_sec': np.random.gamma(shape=5, scale=60, size=num_normal),  # Gamma distribution for normal traffic
    # Some normal traffic can have large packet sizes (creates overlap)
    'Packet_Size': np.random.normal(loc=500, scale=200, size=num_normal),
    'Connection_Duration': np.random.exponential(scale=300, size=num_normal),  # Additional feature
    'Is_Anomaly': 0  # Normal traffic
})

# Create different types of anomalies
anomaly_data = pd.DataFrame()

# Type 1: High packet rate anomalies (50% of anomalies)
num_type1 = num_anomalies // 2
type1_anomalies = pd.DataFrame({
    'Timestamp': pd.date_range(start='2025-03-01', periods=num_type1, freq='S'),
    'Source_IP': [f"192.168.{np.random.randint(1, 5)}.{np.random.randint(1, 255)}" for _ in range(num_type1)],
    'Destination_IP': [f"10.0.{np.random.randint(1, 10)}.{np.random.randint(1, 255)}" for _ in range(num_type1)],
    'Protocol': np.random.choice(['TCP', 'UDP'], num_type1, p=[0.9, 0.1]),  # Mostly TCP
    'Packets_per_sec': np.random.gamma(shape=20, scale=300, size=num_type1),  # Higher packet rates
    # But with some normal-like packet sizes (harder to detect)
    'Packet_Size': np.random.normal(loc=600, scale=250, size=num_type1),
    'Connection_Duration': np.random.exponential(scale=10, size=num_type1),  # Short connections
    'Is_Anomaly': 1  # Anomalous traffic
})

# Type 2: Large packet size anomalies (50% of anomalies)
num_type2 = num_anomalies - num_type1
type2_anomalies = pd.DataFrame({
    'Timestamp': pd.date_range(start='2025-03-02', periods=num_type2, freq='S'),
    'Source_IP': [f"192.168.{np.random.randint(1, 5)}.{np.random.randint(1, 255)}" for _ in range(num_type2)],
    'Destination_IP': [f"10.0.{np.random.randint(1, 10)}.{np.random.randint(1, 255)}" for _ in range(num_type2)],
    'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_type2, p=[0.5, 0.3, 0.2]),
    # Some with normal packet rates (harder to detect)
    'Packets_per_sec': np.random.gamma(shape=6, scale=70, size=num_type2),
    'Packet_Size': np.random.normal(loc=1800, scale=500, size=num_type2),  # Large packets
    'Connection_Duration': np.random.exponential(scale=500, size=num_type2),  # Various durations
    'Is_Anomaly': 1  # Anomalous traffic
})

# Add some noise and outliers to make detection harder
# Add a few normal records that look like anomalies
noise_indices = np.random.choice(normal_data.index, 500, replace=False)
normal_data.loc[noise_indices, 'Packets_per_sec'] = np.random.gamma(shape=15, scale=200, size=500)
normal_data.loc[noise_indices, 'Packet_Size'] = np.random.normal(loc=1200, scale=300, size=500)

# Add a few anomalies that look like normal traffic
anomaly_noise_indices1 = np.random.choice(type1_anomalies.index, 100, replace=False)
type1_anomalies.loc[anomaly_noise_indices1, 'Packets_per_sec'] = np.random.gamma(shape=5, scale=60, size=100)
type1_anomalies.loc[anomaly_noise_indices1, 'Packet_Size'] = np.random.normal(loc=500, scale=200, size=100)

# Combine all data
anomaly_data = pd.concat([type1_anomalies, type2_anomalies])
network_data = pd.concat([normal_data, anomaly_data]).sample(frac=1).reset_index(drop=True)

# Clean up any negative values from normal distributions
network_data['Packet_Size'] = network_data['Packet_Size'].apply(lambda x: max(64, x))
network_data['Packets_per_sec'] = network_data['Packets_per_sec'].apply(lambda x: max(1, x))

# Write the data to a CSV file
network_data.to_csv('network_traffic_data.csv', index=False)

# Plot the data to understand patterns
plt.figure(figsize=(16, 12))

# Plot 1: Packets per second vs Anomaly Flag
plt.subplot(2, 2, 1)
sns.boxplot(x='Is_Anomaly', y='Packets_per_sec', data=network_data)
plt.title('Packets per Second vs Anomaly Flag')
plt.ylabel('Packets per Second')
plt.xlabel('Is Anomaly (1=Yes, 0=No)')
plt.yscale('log')  # Better for showing distribution

# Plot 2: Packet Size vs Anomaly Flag
plt.subplot(2, 2, 2)
sns.boxplot(x='Is_Anomaly', y='Packet_Size', data=network_data)
plt.title('Packet Size vs Anomaly Flag')
plt.ylabel('Packet Size')
plt.xlabel('Is Anomaly (1=Yes, 0=No)')

# Plot 3: Scatter plot of features
plt.subplot(2, 2, 3)
sns.scatterplot(x='Packets_per_sec', y='Packet_Size', hue='Is_Anomaly', 
                data=network_data.sample(5000), alpha=0.6)  # Sample for better visualization
plt.title('Packets per Second vs Packet Size (Colored by Anomaly)')
plt.xlabel('Packets per Second')
plt.ylabel('Packet Size')

# Plot 4: Connection Duration by Anomaly Type
plt.subplot(2, 2, 4)
sns.boxplot(x='Is_Anomaly', y='Connection_Duration', data=network_data)
plt.title('Connection Duration by Anomaly Type')
plt.ylabel('Connection Duration (s)')
plt.xlabel('Is Anomaly (1=Yes, 0=No)')
plt.yscale('log')  # Better for showing distribution

plt.tight_layout()
plt.savefig('network_data_visualization.png')
plt.show()

# Prepare data for modeling
features = ['Packets_per_sec', 'Packet_Size', 'Connection_Duration']
X = network_data[features]
y = network_data['Is_Anomaly']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets (80% train, 20% test)
# Using random_state=None to get different splits each time
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=None)

# Create dataframes for training and testing data
train_indices = y_train.index
test_indices = y_test.index

train_data = network_data.loc[train_indices].copy()
test_data = network_data.loc[test_indices].copy()

# Write training and testing data to separate files
train_data.to_csv('network_traffic_train.csv', index=False)
test_data.to_csv('network_traffic_test.csv', index=False)

# Initialize results dataframe to store predictions from both models
results = network_data.copy()
results['IF_Predicted'] = np.nan
results['RF_Predicted'] = np.nan

# 1. ISOLATION FOREST MODEL
print("\n" + "=" * 50)
print("ISOLATION FOREST MODEL")
print("=" * 50)

# Train Isolation Forest Model on training data
# Using a slightly incorrect contamination parameter (not exactly matching the true anomaly rate)
if_model = IsolationForest(contamination=0.15, random_state=42, n_estimators=200)
if_model.fit(X_train)

# Predict on training data
train_if_predictions = if_model.predict(X_train)
train_if_predictions = np.where(train_if_predictions == -1, 1, 0)  # Convert -1 to 1 (anomaly) and 1 to 0 (normal)
train_data['IF_Predicted'] = train_if_predictions

# Predict on test data
test_if_predictions = if_model.predict(X_test)
test_if_predictions = np.where(test_if_predictions == -1, 1, 0)  # Convert -1 to 1 (anomaly) and 1 to 0 (normal)
test_data['IF_Predicted'] = test_if_predictions

# Store predictions in results dataframe
results.loc[train_indices, 'IF_Predicted'] = train_if_predictions
results.loc[test_indices, 'IF_Predicted'] = test_if_predictions

# Evaluate on training data
train_if_accuracy = accuracy_score(train_data['Is_Anomaly'], train_data['IF_Predicted'])
train_if_report = classification_report(train_data['Is_Anomaly'], train_data['IF_Predicted'])
train_if_cm = confusion_matrix(train_data['Is_Anomaly'], train_data['IF_Predicted'])

# Evaluate on test data
test_if_accuracy = accuracy_score(test_data['Is_Anomaly'], test_data['IF_Predicted'])
test_if_report = classification_report(test_data['Is_Anomaly'], test_data['IF_Predicted'])
test_if_cm = confusion_matrix(test_data['Is_Anomaly'], test_data['IF_Predicted'])

# Print Isolation Forest results
print("ISOLATION FOREST TRAINING RESULTS:")
print(f"Training Data Accuracy: {train_if_accuracy:.4f}")
print("Training Classification Report:\n", train_if_report)
print("Training Confusion Matrix:\n", train_if_cm)

print("\nISOLATION FOREST TEST RESULTS:")
print(f"Test Data Accuracy: {test_if_accuracy:.4f}")
print("Test Classification Report:\n", test_if_report)
print("Test Confusion Matrix:\n", test_if_cm)

# 2. RANDOM FOREST MODEL
print("\n" + "=" * 50)
print("RANDOM FOREST MODEL")
print("=" * 50)

# Train Random Forest Model on training data with class weights to address imbalance
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                 class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Predict on training data
train_rf_predictions = rf_model.predict(X_train)
train_data['RF_Predicted'] = train_rf_predictions

# Predict on test data
test_rf_predictions = rf_model.predict(X_test)
test_data['RF_Predicted'] = test_rf_predictions

# Store predictions in results dataframe
results.loc[train_indices, 'RF_Predicted'] = train_rf_predictions
results.loc[test_indices, 'RF_Predicted'] = test_rf_predictions

# Evaluate on training data
train_rf_accuracy = accuracy_score(train_data['Is_Anomaly'], train_data['RF_Predicted'])
train_rf_report = classification_report(train_data['Is_Anomaly'], train_data['RF_Predicted'])
train_rf_cm = confusion_matrix(train_data['Is_Anomaly'], train_data['RF_Predicted'])

# Evaluate on test data
test_rf_accuracy = accuracy_score(test_data['Is_Anomaly'], test_data['RF_Predicted'])
test_rf_report = classification_report(test_data['Is_Anomaly'], test_data['RF_Predicted'])
test_rf_cm = confusion_matrix(test_data['Is_Anomaly'], test_data['RF_Predicted'])

# Print Random Forest results
print("RANDOM FOREST TRAINING RESULTS:")
print(f"Training Data Accuracy: {train_rf_accuracy:.4f}")
print("Training Classification Report:\n", train_rf_report)
print("Training Confusion Matrix:\n", train_rf_cm)

print("\nRANDOM FOREST TEST RESULTS:")
print(f"Test Data Accuracy: {test_rf_accuracy:.4f}")
print("Test Classification Report:\n", test_rf_report)
print("Test Confusion Matrix:\n", test_rf_cm)

# Save the results to CSV
results.to_csv('network_traffic_with_all_predictions.csv', index=False)

# Plot confusion matrices for both models
plt.figure(figsize=(15, 10))

# Isolation Forest - Training
plt.subplot(2, 2, 1)
sns.heatmap(train_if_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], 
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Isolation Forest - Training Data\nAccuracy: {train_if_accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Isolation Forest - Testing
plt.subplot(2, 2, 2)
sns.heatmap(test_if_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], 
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Isolation Forest - Test Data\nAccuracy: {test_if_accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Random Forest - Training
plt.subplot(2, 2, 3)
sns.heatmap(train_rf_cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal', 'Anomaly'], 
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Random Forest - Training Data\nAccuracy: {train_rf_accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Random Forest - Testing
plt.subplot(2, 2, 4)
sns.heatmap(test_rf_cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal', 'Anomaly'], 
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Random Forest - Test Data\nAccuracy: {test_rf_accuracy:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('model_confusion_matrices.png')
plt.show()

# Feature importance for Random Forest
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importance in Random Forest Model')
plt.barh(range(len(indices)), importances[indices], color='g')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ROC Curves for both models
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(12, 6))

# Isolation Forest - Training vs Testing
if_train_scores = if_model.decision_function(X_train) * -1  # Invert scores for ROC
if_test_scores = if_model.decision_function(X_test) * -1  # Invert scores for ROC

fpr_if_train, tpr_if_train, _ = roc_curve(y_train, if_train_scores)
fpr_if_test, tpr_if_test, _ = roc_curve(y_test, if_test_scores)

roc_auc_if_train = auc(fpr_if_train, tpr_if_train)
roc_auc_if_test = auc(fpr_if_test, tpr_if_test)

# Random Forest - Training vs Testing
rf_train_proba = rf_model.predict_proba(X_train)[:, 1]
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, rf_train_proba)
fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, rf_test_proba)

roc_auc_rf_train = auc(fpr_rf_train, tpr_rf_train)
roc_auc_rf_test = auc(fpr_rf_test, tpr_rf_test)

# Plot all ROC curves
plt.plot(fpr_if_train, tpr_if_train, 'b-', label=f'IF Train (AUC = {roc_auc_if_train:.4f})')
plt.plot(fpr_if_test, tpr_if_test, 'b--', label=f'IF Test (AUC = {roc_auc_if_test:.4f})')
plt.plot(fpr_rf_train, tpr_rf_train, 'g-', label=f'RF Train (AUC = {roc_auc_rf_train:.4f})')
plt.plot(fpr_rf_test, tpr_rf_test, 'g--', label=f'RF Test (AUC = {roc_auc_rf_test:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.show()

print("\n" + "=" * 50)
print("Files generated:")
print("- network_traffic_data.csv (original data)")
print("- network_traffic_train.csv (training data)")
print("- network_traffic_test.csv (testing data)")
print("- network_traffic_with_all_predictions.csv (all data with predictions from both models)")
print("- network_data_visualization.png (feature visualization plots)")
print("- model_confusion_matrices.png (confusion matrices for both models)")
print("- feature_importance.png (feature importance for Random Forest)")
print("- roc_curves.png (ROC curves comparing model performance)")

print("\nData Summary:")
print(f"Total records: {len(network_data)}")
print(f"Normal traffic: {len(network_data[network_data['Is_Anomaly'] == 0])} records ({len(network_data[network_data['Is_Anomaly'] == 0])/len(network_data)*100:.1f}%)")
print(f"Anomalous traffic: {len(network_data[network_data['Is_Anomaly'] == 1])} records ({len(network_data[network_data['Is_Anomaly'] == 1])/len(network_data)*100:.1f}%)")