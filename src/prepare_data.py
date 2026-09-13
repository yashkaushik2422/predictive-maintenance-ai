import pandas as pd

RAW_DATA = "data/raw/ai4i2020.csv"
PROCESSED_DATA = "data/processed/machine_failure_data.csv"


def main():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA)

    print(f"Original shape: {df.shape}")

    # Remove identifiers that should not be used for prediction.
    df = df.drop(columns=["UDI", "Product ID"])

    # Remove failure-mode columns.
    # These describe specific failure mechanisms and would create
    # information leakage for our prediction task.
    failure_mode_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=failure_mode_columns)

    # Check for missing values.
    missing = df.isnull().sum().sum()

    if missing > 0:
        raise ValueError(f"Dataset contains {missing} missing values.")

    # Save cleaned dataset.
    df.to_csv(PROCESSED_DATA, index=False)

    print(f"Cleaned shape: {df.shape}")
    print(f"Saved to: {PROCESSED_DATA}")
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()