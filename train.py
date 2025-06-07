# fraud_detection/train.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    PrecisionRecallDisplay
)

# ===== Configuration =====
N_JOBS = -1  # Use all CPU cores

# Setup directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def load_data():
    """Load preprocessed and split data"""
    try:
        # Directly load split and scaled data
        X_train, X_test, y_train, y_test = joblib.load("processed_data/processed_data.pkl")

        print(f"\nData shapes:")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Class dist - Train: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Class dist - Test: {pd.Series(y_test).value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"\n❌ Error loading data: {str(e)}")
        exit()

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with fraud-tuned parameters"""
    print("\n=== Training XGBoost ===")
    model = XGBClassifier(
        scale_pos_weight=100,
        eval_metric='aucpr',
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300,
        random_state=42,
        n_jobs=N_JOBS
    )

    start = time.time()
    model.fit(
        X_train, y_train
        # Remove early_stopping_rounds and eval_set for now
    )
    print(f"Training time: {(time.time()-start)/60:.1f} min")
    return model


def evaluate_model(name, model, X_test, y_test):
    """Evaluate model with fraud-specific metrics"""
    print(f"\n=== {name} Evaluation ===")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Avg Precision: {average_precision_score(y_test, y_proba):.4f}")

        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=name)
        plt.title(f'Precision-Recall Curve - {name}')
        plt.savefig(f"reports/{name.lower()}_pr_curve.png")
        plt.close()

        # Top 100 risky predictions
        high_risk = pd.DataFrame({
            'prob': y_proba,
            'actual': y_test
        }).sort_values('prob', ascending=False).head(100)
        print("\nTop 100 predicted frauds:")
        print(f"Actual frauds caught: {high_risk['actual'].sum()}/{len(high_risk)}")

def save_artifacts(model, feature_names):
    """Save model and feature importance"""
    joblib.dump(model, "models/xgb_model.pkl")

    if hasattr(model, 'feature_importances_'):
        pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).to_csv(
            "reports/feature_importance.csv",
            index=False
        )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*50)

    # Load
    X_train, X_test, y_train, y_test = load_data()

    # Train
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # Evaluate
    evaluate_model("XGBoost", model, X_test, y_test)

    # Save artifacts
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]  # Use real names if needed
    save_artifacts(model, feature_names)

    print("\n✔ Training complete!")
