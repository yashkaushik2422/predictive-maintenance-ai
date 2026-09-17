import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


DATA_PATH = "data/processed/machine_failure_data.csv"
MODEL_PATH = "models/predictive_maintenance_model.joblib"
OUTPUT_PATH = "data/processed/shap_global_summary.png"

SAMPLE_SIZE = 1000
MACHINE_INDEX = 69


def main():
    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    # Convert machine type into numeric dummy variables
    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=True
    )

    # Load trained Random Forest model
    model = joblib.load(MODEL_PATH)

    # Create SHAP explainer for the tree-based model
    explainer = shap.TreeExplainer(model)

    # Select a reproducible representative sample
    X_sample = X.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

    shap_values = explainer.shap_values(X_sample)

    # Select SHAP values for the failure class
    failure_shap_values = shap_values[:, :, 1]

    print("\n" + "=" * 50)
    print("GLOBAL SHAP EXPLANATION")
    print("=" * 50)

    print(f"\nRepresentative sample size: {len(X_sample)}")

    print("\nFeatures:")
    for feature in X.columns:
        print(f"  - {feature}")

    # Create and save global SHAP summary plot
    shap.summary_plot(
        failure_shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nGlobal SHAP plot saved to:")
    print(OUTPUT_PATH)

    # ---------------------------------------------------------
    # Individual machine explanation
    # ---------------------------------------------------------

    X_machine = X.loc[[MACHINE_INDEX]]

    machine_shap = explainer.shap_values(X_machine)

    if hasattr(machine_shap, "values"):
        machine_shap_values = machine_shap.values
    else:
        machine_shap_values = machine_shap

    # Select failure-class SHAP values
    machine_shap_values = machine_shap_values[:, :, 1]

    explanation = pd.DataFrame(
        {
            "Feature": X.columns,
            "SHAP value": machine_shap_values[0]
        }
    )

    explanation["Absolute SHAP"] = (
        explanation["SHAP value"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute SHAP",
        ascending=False
    )

    print("\n" + "=" * 50)
    print("INDIVIDUAL MACHINE EXPLANATION")
    print("=" * 50)

    print(f"\nMachine index: {MACHINE_INDEX}")

    print("\nMachine conditions:")
    print(X_machine)

    print("\nActual outcome:")
    print(y.loc[MACHINE_INDEX])

    print("\nSHAP contributions:")
    print(explanation.to_string(index=False))


if __name__ == "__main__":
    main()