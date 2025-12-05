import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    #Preprocess text: lowercase, remove special chars, normalize
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    #remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    #remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    #remove special characters/keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    #remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_spam_data(file_path):
    """Load spam data from CSV file"""
    df = pd.read_csv(file_path)
    
    #handle different formats
    if 'v1' in df.columns:
        df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
        df['text'] = df['v2'].fillna('')
    elif 'label' in df.columns and 'text' in df.columns:
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['text'] = df['text'].fillna('')
    elif 'label_num' in df.columns:
        df['label'] = df['label_num']
        df['text'] = df['text'].fillna('')
    else:
        # Try to infer columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            if 'text' in col.lower() or 'message' in col.lower():
                text_col = col
            if 'label' in col.lower():
                label_col = col
        
        if text_col and label_col:
            df['text'] = df[text_col].fillna('')
            if df[label_col].dtype == 'object':
                df['label'] = df[label_col].map({'ham': 0, 'spam': 1})
            else:
                df['label'] = df[label_col]
        else:
            raise ValueError(f"Cannot infer columns from {file_path}")
    
    return df

def train_and_evaluate_models(X_train, y_train, X_test):
    """Train multiple models and select the best one"""
    print("\nTraining and evaluating models...")
    
    # Vectorize text data
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'NaiveBayes': MultinomialNB(alpha=0.1)
    }
    
    best_score = -1
    best_clf = None
    best_name = None
    best_vectorizer = None
    
    #use stratified k-fold 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        try:
            print(f"\nEvaluating {name}...")
            scores = cross_val_score(clf, X_train_vec, y_train, cv=cv, scoring='f1', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"  {name} CV F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_clf = clf
                best_name = name
                best_vectorizer = vectorizer
        except Exception as e:
            print(f"  {name} failed: {str(e)}")
            continue
    
    if best_clf is None:
        # Fallback
        best_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        best_name = "RandomForest (fallback)"
        best_vectorizer = vectorizer
    
    print(f"\nSelected: {best_name} with CV F1-Score: {best_score:.4f}")
    
    #train best classifier
    print(f"Training {best_name}...")
    best_clf.fit(X_train_vec, y_train)
    
    #make predictions
    print("Making predictions...")
    predictions = best_clf.predict(X_test_vec)
    
    #if we need probabilities for evaluation
    try:
        probabilities = best_clf.predict_proba(X_test_vec)[:, 1]
    except:
        probabilities = None
    
    return predictions, probabilities, best_name

def main():
    """Main function for spam detection"""
    print("=" * 60)
    print("Spam Email Detection")
    print("=" * 60)
    
    #load training data
    print("\nLoading training data...")
    train1 = load_spam_data("Spam Email Detection/spam_train1.csv")
    train2 = load_spam_data("Spam Email Detection/spam_train2.csv")
    
    print(f"Train1 shape: {train1.shape}")
    print(f"Train2 shape: {train2.shape}")
    print(f"Train1 - Ham: {(train1['label'] == 0).sum()}, Spam: {(train1['label'] == 1).sum()}")
    print(f"Train2 - Ham: {(train2['label'] == 0).sum()}, Spam: {(train2['label'] == 1).sum()}")
    
    #combine training data
    train_combined = pd.concat([train1, train2], ignore_index=True)
    print(f"\nCombined training data shape: {train_combined.shape}")
    print(f"Combined - Ham: {(train_combined['label'] == 0).sum()}, Spam: {(train_combined['label'] == 1).sum()}")
    
    #preprocess text
    print("\nPreprocessing text...")
    train_combined['text_processed'] = train_combined['text'].apply(preprocess_text)
    
    #load test data
    print("\nLoading test data...")
    test_df = pd.read_csv("Spam Email Detection/spam_test.csv")
    
    #handle test data format
    if 'message' in test_df.columns:
        test_df['text'] = test_df['message'].fillna('')
    elif 'text' not in test_df.columns:
        #try to find text column
        for col in test_df.columns:
            if 'text' in col.lower() or 'message' in col.lower():
                test_df['text'] = test_df[col].fillna('')
                break
    
    if 'text' not in test_df.columns:
        raise ValueError("Cannot find text column in test data")
    
    test_df['text_processed'] = test_df['text'].apply(preprocess_text)
    
    print(f"Test data shape: {test_df.shape}")
    
    #prepare data
    X_train = train_combined['text_processed'].values
    y_train = train_combined['label'].values
    X_test = test_df['text_processed'].values
    
    #train and predict
    predictions, probabilities, model_name = train_and_evaluate_models(X_train, y_train, X_test)
    
    #convert predictions to labels (0=ham, 1=spam)
    #save as ham/spam strings
    predictions_labels = ['ham' if p == 0 else 'spam' for p in predictions]
    
    #save predictions (saved in current working directory)
    output_file = "AwalinkarSpam.txt"
    with open(output_file, 'w') as f:
        for label in predictions_labels:
            f.write(f"{label}\n")
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"Full path: {os.path.abspath(output_file)}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Ham predictions: {predictions_labels.count('ham')}")
    print(f"Spam predictions: {predictions_labels.count('spam')}")
    
    #print statistics
    print(f"\nModel used: {model_name}")

if __name__ == "__main__":
    main()

