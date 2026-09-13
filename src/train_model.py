import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
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

    # Train model
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Model training complete.")

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