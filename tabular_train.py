import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_PATH = "combined_patient_clinical_data.csv"

BETHESDA_ORDER = {"Btd I": 1, "Btd II": 2, "Btd III": 3, "Btd IV": 4, "Btd V": 5, "Btd VI": 6}

def load_and_prepare():
    df = pd.read_csv(CSV_PATH)

    before = len(df)
    df = df[df["diagnosis_class"].isin(["Benign", "Malignant"])].copy()
    print(f"Dropped {before - len(df)} rows with unverified/indeterminate labels.")

    df["bethesda_ordinal"] = df["bethesda_category"].map(BETHESDA_ORDER)
    df["gender_encoded"] = LabelEncoder().fit_transform(df["gender"])

    text_blob = (df["usg_trd"].astype(str) + " " + df["site"].astype(str)).str.lower()
    df["is_solitary_nodule"] = text_blob.str.contains("sol").astype(int)
    df["is_multinodular"] = text_blob.str.contains("mn|multi").astype(int)
    df["is_cystic"] = text_blob.str.contains("cyst|cg").astype(int)

    features = ["age", "gender_encoded", "bethesda_ordinal",
                "is_solitary_nodule", "is_multinodular", "is_cystic"]

    X = df[features].fillna(0)
    y = (df["diagnosis_class"] == "Malignant").astype(int)

    return X, y, features, df

def main():
    X, y, features, df = load_and_prepare()
    print(f"\nDataset: {len(X)} patients | Malignant: {y.sum()} | Benign: {(y==0).sum()}")
    print(f"Features used: {features}\n")

    if y.sum() < 3:
        print("⚠️ Very few malignant examples — results will be extremely unstable.\n")

    loo = LeaveOneOut()

    for name, model in [
        ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ]:
        preds = cross_val_predict(model, X, y, cv=loo)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)

        print(f"=== {name} (Leave-One-Out CV) ===")
        print(f"Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix [Benign, Malignant]:")
        print(cm)
        print(classification_report(y, preds, target_names=["Benign", "Malignant"], zero_division=0))
        print()

    rf_final = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf_final.fit(X, y)
    importances = sorted(zip(features, rf_final.feature_importances_), key=lambda x: -x[1])
    print("=== Feature Importances (Random Forest) ===")
    for feat, imp in importances:
        print(f"  {feat}: {imp:.4f}")

if __name__ == "__main__":
    main()