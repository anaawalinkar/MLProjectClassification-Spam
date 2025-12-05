# Classification and Spam Detection Project

This project implements machine learning solutions for:
1. **Classification Task**: 4 datasets with numeric features
2. **Spam Email Detection**: Binary classification of emails as Spam or Ham

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn
```

## Project Structure

```
data/
├── classification/
│   ├── TrainData1.txt - TrainData4.txt
│   ├── TrainLabel1.txt - TrainLabel4.txt
│   └── TestData1.txt - TestData4.txt
├── Spam Email Detection/
│   ├── spam_train1.csv
│   ├── spam_train2.csv
│   └── spam_test.csv
├── classification.py          # Classification script
├── spam_detection.py         # Spam detection script
└── README.md                 # This file
```

## Running the Code

### 1. Classification Task

Run the classification script to process all 4 datasets:

```bash
python classification.py
```

This will:
- Load training data and labels for all 4 datasets
- Handle missing values (represented as 1.00000000000000e+99)
- Train multiple classifiers (Random Forest, Gradient Boosting, SVM)
- Select the best model using cross-validation
- Generate predictions for test data
- Save results to:
  - `AwalinkarClassification1.txt`
  - `AwalinkarClassification2.txt`
  - `AwalinkarClassification3.txt`
  - `AwalinkarClassification4.txt`

### 2. Spam Email Detection

Run the spam detection script:

```bash
python spam_detection.py
```

This will:
- Load and combine training datasets (spam_train1.csv and spam_train2.csv)
- Preprocess text data (lowercase, remove URLs, normalize)
- Extract features using TF-IDF vectorization
- Train multiple models (Random Forest, Gradient Boosting, SVM, Naive Bayes)
- Select the best model using cross-validation
- Generate predictions for test data
- Save results to: `AwalinkarSpam.txt`

## Output Files

**Output Location**: All prediction files are saved in the same directory where you run the scripts (the current working directory). 

The scripts generate the following prediction files:

1. **AwalinkarClassification1.txt** - Predictions for TestData1 (53 samples, 5 classes)
2. **AwalinkarClassification2.txt** - Predictions for TestData2 (74 samples, 11 classes)
3. **AwalinkarClassification3.txt** - Predictions for TestData3 (1092 samples, 9 classes)
4. **AwalinkarClassification4.txt** - Predictions for TestData4 (480 samples, 6 classes)
5. **AwalinkarSpam.txt** - Predictions for spam test data (6447 samples, ham/spam)

Each classification file contains one integer per line representing the predicted class.
The spam file contains one label per line (either "ham" or "spam").

## Methodology

### Classification Task

1. **Data Preprocessing**:
   - Missing values (1.00000000000000e+99) are replaced with feature medians
   - Features are standardized using StandardScaler

2. **Model Selection**:
   - Multiple classifiers are evaluated using 5-fold cross-validation
   - Best model is selected based on accuracy
   - Models tested: Random Forest, Gradient Boosting, SVM

3. **Prediction**:
   - Selected model is trained on full training data
   - Predictions are made on test data

### Spam Detection Task

1. **Text Preprocessing**:
   - Convert to lowercase
   - Remove URLs and email addresses
   - Remove special characters
   - Normalize whitespace

2. **Feature Extraction**:
   - TF-IDF vectorization with:
     - Max 5000 features
     - Unigrams and bigrams
     - English stop words removal
     - Min document frequency: 2

3. **Model Selection**:
   - Multiple classifiers evaluated using stratified 5-fold cross-validation
   - Best model selected based on F1-score (handles class imbalance)
   - Models tested: Random Forest, Gradient Boosting, SVM, Naive Bayes

4. **Prediction**:
   - Selected model trained on combined training data
   - Predictions made on test data

## Notes

- Missing values in classification data are handled by replacing with feature medians
- For spam detection, training data from both files is combined
- Cross-validation ensures model generalization
- All models use random_state=42 for reproducibility

## Author

Awalinkar

