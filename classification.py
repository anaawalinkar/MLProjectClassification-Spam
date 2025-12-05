"""
Classification Task: Predict class labels for 4 datasets
Handles missing values (represented as 1.00000000000000e+99)
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Missing value indicator
MISSING_VALUE = 1.00000000000000e+99

def load_data(data_path, label_path=None):
    """Load data from text file, handling tab-separated values"""
    # Read data as space/tab separated
    data = pd.read_csv(data_path, sep='\s+', header=None)
    
    # Convert to numpy array
    data_array = data.values
    
    # Handle missing values
    data_array[data_array == MISSING_VALUE] = np.nan
    
    # Fill missing values with median of each feature
    for col in range(data_array.shape[1]):
        col_data = data_array[:, col]
        if np.isnan(col_data).any():
            median_val = np.nanmedian(col_data)
            if np.isnan(median_val):
                median_val = 0.0  # If all values are missing, use 0
            data_array[np.isnan(data_array[:, col]), col] = median_val
    
    labels = None
    if label_path:
        labels = pd.read_csv(label_path, header=None).values.flatten()
    
    return data_array, labels

def train_and_predict(train_data, train_labels, test_data, dataset_num):
    """Train classifier and make predictions"""
    print(f"\nProcessing Dataset {dataset_num}...")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of classes: {len(np.unique(train_labels))}")
    
    # Standardize features
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # Try multiple classifiers and select the best one
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    best_score = -1
    best_clf = None
    best_name = None
    
    for name, clf in classifiers.items():
        try:
            # Use cross-validation to select best model
            scores = cross_val_score(clf, train_data_scaled, train_labels, cv=5, scoring='accuracy')
            mean_score = scores.mean()
            print(f"  {name} CV Accuracy: {mean_score:.4f} (+/- {scores.std():.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_clf = clf
                best_name = name
        except Exception as e:
            print(f"  {name} failed: {str(e)}")
            continue
    
    if best_clf is None:
        # Fallback to RandomForest
        best_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        best_name = "RandomForest (fallback)"
    
    print(f"  Selected: {best_name} with CV accuracy: {best_score:.4f}")
    
    # Train the best classifier
    best_clf.fit(train_data_scaled, train_labels)
    
    # Make predictions
    predictions = best_clf.predict(test_data_scaled)
    
    return predictions

def main():
    """Main function to process all 4 classification datasets"""
    base_path = "classification"
    
    datasets = [
        {
            'train_data': f"{base_path}/TrainData1.txt",
            'train_label': f"{base_path}/TrainLabel1.txt",
            'test_data': f"{base_path}/TestData1.txt",
            'output': "AwalinkarClassification1.txt"
        },
        {
            'train_data': f"{base_path}/TrainData2.txt",
            'train_label': f"{base_path}/TrainLabel2.txt",
            'test_data': f"{base_path}/TestData2.txt",
            'output': "AwalinkarClassification2.txt"
        },
        {
            'train_data': f"{base_path}/TrainData3.txt",
            'train_label': f"{base_path}/TrainLabel3.txt",
            'test_data': f"{base_path}/TestData3.txt",
            'output': "AwalinkarClassification3.txt"
        },
        {
            'train_data': f"{base_path}/TrainData4.txt",
            'train_label': f"{base_path}/TrainLabel4.txt",
            'test_data': f"{base_path}/TestData4.txt",
            'output': "AwalinkarClassification4.txt"
        }
    ]
    
    for i, dataset in enumerate(datasets, 1):
        try:
            # Load data
            train_data, train_labels = load_data(dataset['train_data'], dataset['train_label'])
            test_data, _ = load_data(dataset['test_data'])
            
            # Train and predict
            predictions = train_and_predict(train_data, train_labels, test_data, i)
            
            # Save predictions (saved in current working directory)
            output_path = dataset['output']
            np.savetxt(output_path, predictions, fmt='%d')
            print(f"  Predictions saved to: {output_path}")
            print(f"  Full path: {os.path.abspath(output_path)}")
            print(f"  Number of predictions: {len(predictions)}")
            
        except Exception as e:
            print(f"Error processing dataset {i}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

