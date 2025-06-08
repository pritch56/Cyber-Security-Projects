import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib  # Import joblib for saving and loading models

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Load the dataset from a Parquet file
file_path = r'C:\\Users\\Pritch\\Downloads\\archive\\cic-collection.parquet'  # Adjust path accordingly
data_full = pd.read_parquet(file_path)
data = data_full  # Use the full dataset or limit to first 1000 rows for faster processing

# Define features and target variable
features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packets Length Total', 'Bwd Packets Length Total',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean', 
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 
    'Bwd Packet Length Mean', 'Fwd Seg Size Min', 
    'Active Mean', 'Active Std', 'Active Max', 
    'Active Min', 'Idle Mean', 'Idle Std', 
    'Idle Max', 'Idle Min'
]
target = 'ClassLabel'

# Drop rows with missing values (optional)
data = data.dropna(subset=features + [target])

# Ensure target is categorical (strings like 'Benign', etc.)
data[target] = data[target].astype('category')

# Split into train/test
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the model file path
model_file_path = 'random_forest_model.joblib'

# Check if the model already exists
if os.path.exists(model_file_path):
    # Load the model
    model = joblib.load(model_file_path)
    print("Model loaded from disk.")
else:
    # Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model to disk
    joblib.dump(model, model_file_path)
    print("Model trained and saved to disk.")

# Predict test
y_pred = model.predict(X_test)

# Complete classification report and overall accuracy
print("Overall Classification Report:")
print(classification_report(y_test, y_pred))
print("Overall Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

# Get list of class labels from the category dtype directly
attack_types = data[target].cat.categories.tolist()

# Prepare to plot correlation matrices and print per-class accuracies
# Check available styles and set a default
available_styles = plt.style.available
if 'seaborn-whitegrid' in available_styles:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('default')  # Fallback to default style

sns.set(font_scale=1.1)
num_classes = len(attack_types)

# Create a figure large enough to hold subplots of correlation matrices
fig_corr, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 10))
axes = axes.flatten()

for i, attack in enumerate(attack_types):
    # Filter data for that attack type
    data_class = data[data[target] == attack]
    
    # Compute correlation matrix on features only
    corr = data_class[features].corr()
    
    # Plot correlation matrix heatmap for this class
    sns.heatmap(corr, ax=axes[i], annot=False, cmap='coolwarm', square=True,
                cbar=i==num_classes-1,  # only last plot has color bar
                vmin=-1, vmax=1)
    axes[i].set_title(f'Feature Correlation: {attack}', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()

# Calculate test accuracy per class on the test set
print("\nAccuracy for each class in test data:")
for attack in attack_types:
    # Get indices in test data where y_test == attack
    indices = y_test[y_test == attack].index
    if len(indices) == 0:
        print(f"- {attack}: No samples in test set.")
        continue
    y_true_class = y_test.loc[indices]
    y_pred_class = pd.Series(y_pred, index=y_test.index).loc[indices]
    acc = accuracy_score(y_true_class, y_pred_class)
    print(f"- {attack}: Accuracy = {acc:.4f} [{len(indices)} samples]")
