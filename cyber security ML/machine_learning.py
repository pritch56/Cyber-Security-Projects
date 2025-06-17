import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load only the first 1000 rows of the dataset
file_path = r'C:\\Users\\Pritch\\Downloads\\archive\\cic-collection.parquet'
data_full = pd.read_parquet(file_path)
data = data_full

# Define features and target variable
features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Max", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "SYN Flag Count", "URG Flag Count", "Avg Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Act Data Packets", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]
target = 'ClassLabel'

# Drop missing values
data = data.dropna(subset=features + [target])

# Ensure target is categorical
data[target] = data[target].astype('category')

# Train/test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# File path for saving the model
model_file_path = 'Network_Threat_ML2.pkl'

# Load or train model
if os.path.exists(model_file_path):
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from disk.")
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved to disk.")
    print("Saving model to:", os.path.abspath(model_file_path))


# Predict and evaluate
y_pred = model.predict(X_test)

print("\nOverall Classification Report:")
print(classification_report(y_test, y_pred))
print("Overall Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

# Get class labels
attack_types = data[target].cat.categories.tolist()

# Plot correlation heatmaps per class
plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
sns.set(font_scale=1.1)

num_classes = len(attack_types)
fig_corr, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 10))
axes = axes.flatten()

# Plot and save correlation heatmaps per class
plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
sns.set(font_scale=1.1)

num_classes = len(attack_types)
fig_corr, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 10))
axes = axes.flatten()

correlation_dir = 'correlation_matrices'
os.makedirs(correlation_dir, exist_ok=True)

for i, attack in enumerate(attack_types):
    if i >= len(axes):
        break
    data_class = data[data[target] == attack]
    corr = data_class[features].corr()
    
    # Save correlation matrix as CSV
    corr_file_path = os.path.join(correlation_dir, f'{attack}_correlation.csv')
    corr.to_csv(corr_file_path)
    
    sns.heatmap(corr, ax=axes[i], annot=False, cmap='coolwarm', square=True,
                cbar=i == num_classes - 1, vmin=-1, vmax=1)
    axes[i].set_title(f'Feature Correlation: {attack}', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='y', rotation=0)
    
    print(f"Saved correlation matrix for '{attack}' to {corr_file_path}")

plt.tight_layout()
plt.show()

# Accuracy per class
print("\nAccuracy for each class in test data:")
for attack in attack_types:
    indices = y_test[y_test == attack].index
    if len(indices) == 0:
        print(f"- {attack}: No samples in test set.")
        continue
    y_true_class = y_test.loc[indices]
    y_pred_class = pd.Series(y_pred, index=y_test.index).loc[indices]
    acc = accuracy_score(y_true_class, y_pred_class)
    print(f"- {attack}: Accuracy = {acc:.4f} [{len(indices)} samples]")
