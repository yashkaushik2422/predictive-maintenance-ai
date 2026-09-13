import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/machine_failure_data.csv"
OUTPUT_DIR = "data/processed"


def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    print("\nFailure counts:")
    print(df["Machine failure"].value_counts())

    print("\nFailure rates:")
    print(df["Machine failure"].value_counts(normalize=True))

    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    print("\nAverage sensor values by machine failure:")
    print(
        df.groupby("Machine failure")[numerical_columns]
        .mean()
        .round(2)
    )

    # Create and save plots without opening popup windows.
    for column in numerical_columns:
        plt.figure(figsize=(8, 5))

        df.boxplot(
            column=column,
            by="Machine failure"
        )

        plt.title(f"{column} vs Machine Failure")
        plt.suptitle("")
        plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
        plt.ylabel(column)
        plt.tight_layout()

        filename = (
            column
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        output_path = f"{OUTPUT_DIR}/{filename}_failure.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()