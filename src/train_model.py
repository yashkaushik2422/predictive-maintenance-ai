import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

DATA_PATH = "data/processed/machine_failure_data.csv"

def main():
    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Separate features and target
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    # Convert categorical Type column to numeric
    X = pd.get_dummies(X, columns=["Type"], drop_first=True)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining set:", X_train.shape)
    print("Testing set:", X_test.shape)

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Compare multiple models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = {}

    for name, candidate_model in models.items():
        scores = cross_val_score(
            candidate_model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        results[name] = scores

        print(f"\n{name}")
        print(f"F1 scores: {[round(score, 4) for score in scores]}")
        print(f"Mean F1:   {scores.mean():.4f}")
        print(f"Std F1:    {scores.std():.4f}")

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1"
    )

    print("\n5-Fold Cross-Validation F1 Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"Fold {i}: {score:.4f}")

    print(f"Mean F1: {cv_scores.mean():.4f}")
    print(f"Std F1:  {cv_scores.std():.4f}")
    
    # Train model
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Model training complete.")

    # Threshold tuning
    print("\n" + "=" * 50)
    print("THRESHOLD ANALYSIS")
    print("=" * 50)

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    for threshold in thresholds:
        y_threshold_pred = (y_prob >= threshold).astype(int)

        threshold_precision = precision_score(
            y_test,
            y_threshold_pred,
            zero_division=0
        )

        threshold_recall = recall_score(
            y_test,
            y_threshold_pred,
            zero_division=0
        )

        threshold_f1 = f1_score(
            y_test,
            y_threshold_pred,
            zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {threshold_precision:.4f} | "
            f"Recall: {threshold_recall:.4f} | "
            f"F1: {threshold_f1:.4f}"
        )

    # Maintenance cost analysis
    print("\n" + "=" * 50)
    print("MAINTENANCE COST ANALYSIS")
    print("=" * 50)

    failure_cost = 10000
    inspection_cost = 100

    print(f"Assumed missed-failure cost: €{failure_cost:,}")
    print(f"Assumed inspection cost:     €{inspection_cost:,}")

    best_threshold = None
    lowest_cost = float("inf")

    for threshold in thresholds:
        y_threshold_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_threshold_pred
        ).ravel()

        total_cost = (
            fn * failure_cost
            + fp * inspection_cost
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"False Positives: {fp:3d} | "
            f"False Negatives: {fn:3d} | "
            f"Total Cost: €{total_cost:,}"
        )

        if total_cost < lowest_cost:
            lowest_cost = total_cost
            best_threshold = threshold

    print("\nRecommended threshold based on assumed costs:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Estimated test-set cost: €{lowest_cost:,}")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)

    print(importance.round(4))
    
    # Save trained model
    model_path = "models/predictive_maintenance_model.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()