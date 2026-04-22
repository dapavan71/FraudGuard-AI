import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve

def generate_dummy_data(n_samples=20000):
    print(f"Generating {n_samples} dummy records for demonstration...")
    # Time
    time_col = np.random.randint(0, 172800, size=n_samples)
    # V1-V28
    v_cols = np.random.randn(n_samples, 28)
    # Amount
    amount_col = np.abs(np.random.randn(n_samples)) * 100
    
    # Class (imbalanced, ~0.005 fraud just so we get enough for SMOTE)
    y = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
    
    # Let's adjust fraud features a bit so the model can actually learn something in dummy data
    fraud_indices = np.where(y == 1)[0]
    v_cols[fraud_indices] += 2.0 

    data = np.column_stack([time_col, v_cols, amount_col, y])
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(data, columns=columns)
    df['Class'] = df['Class'].astype(int)
    return df

def main():
    dataset_path = 'creditcard.csv'
    if os.path.exists(dataset_path):
        print("Loading real dataset...")
        df = pd.read_csv(dataset_path)
    else:
        df = generate_dummy_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

    # Preprocessing
    scaler_time = StandardScaler()
    scaler_amount = StandardScaler()

    df['Scaled_Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Scaled_Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Keep track of feature order
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    xgb_model.fit(X_train_smote, y_train_smote)
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print("XGBoost Test Classification Report:")
    print(classification_report(y_test, y_pred_xgb, zero_division=0))

    # Threshold tuning based on Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_xgb)
    # Find threshold with high recall but decent precision (f1)
    fscore = (2 * precision * recall) / (precision + recall + 1e-8)
    ix = np.argmax(fscore)
    best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
    print(f"Best tuned threshold based on F-Score: {best_thresh:.4f}")

    print("Training Isolation Forest...")
    iso_model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    # Train only on non-fraud training data for purely unsupervised anomaly detection
    X_train_normal = X_train[y_train == 0]
    iso_model.fit(X_train_normal)

    # Save everything
    print("Saving models and scalers...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(iso_model, 'models/iso_model.joblib')
    joblib.dump(scaler_time, 'models/scaler_time.joblib')
    joblib.dump(scaler_amount, 'models/scaler_amount.joblib')
    joblib.dump({'features': feature_names, 'threshold': best_thresh}, 'models/meta.joblib')
    print("Successfully built and saved project artifacts.")

if __name__ == "__main__":
    main()
