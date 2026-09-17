# Predictive Maintenance AI

A machine learning project that predicts industrial machine failure risk from operating conditions, with an interactive Streamlit dashboard for exploring "what-if" scenarios.

Built with Python, Pandas, Scikit-learn, SHAP, Matplotlib, and Streamlit.

## What it does

The project takes a machine's operating conditions — type, temperature, rotational speed, torque, and tool wear — and estimates its probability of failure.

The dashboard then presents:

- Failure probability
- Risk level
- Maintenance recommendation
- Local SHAP explanation showing which features influenced the prediction
- Global feature importance from the trained Random Forest model

```
Raw data → data cleaning → exploratory analysis → feature engineering
   → model comparison → cross-validation → Random Forest
   → SHAP explainability → Streamlit dashboard
```

## Results

The final Random Forest model was evaluated on a held-out test set of 2,000 machines.

| Metric | Score |
|---|---:|
| Accuracy | 98.2% |
| Precision | 75.0% |
| Recall | 70.6% |
| F1 Score | 72.7% |

The test set contained 68 actual failures. The model detected 48 of them, missed 20, and produced 16 false alarms.

Because failures represent only 3.39% of the dataset, accuracy alone isn't a sufficient evaluation metric — precision, recall, and F1 score give a much clearer picture of failure-detection performance.

## Model comparison

Three classification models were evaluated using stratified 5-fold cross-validation on the training data.

| Model | Mean F1 | Std F1 |
|---|---:|---:|
| Logistic Regression | 0.235 | 0.009 |
| Random Forest | 0.693 | 0.034 |
| Gradient Boosting | 0.683 | 0.020 |

Random Forest was selected for the final test-set evaluation. Cross-validation was performed before touching the test set, keeping it fully separate from model selection.

## The model

The final model is a Random Forest classifier:

- `n_estimators = 200`
- `class_weight = "balanced"`
- `random_state = 42`

It uses a stratified 80/20 train-test split, since failures are relatively rare in the data.

**Input features:** machine type, air temperature, process temperature, rotational speed, torque, tool wear.

`UDI` and `Product ID` were dropped as identifiers rather than operating conditions. The failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were also removed — they directly describe specific failure modes and would leak the answer into the input, making the task unrealistically easy.

## Feature importance

| Feature | Importance |
|---|---:|
| Torque | 32.2% |
| Rotational Speed | 28.6% |
| Tool Wear | 21.3% |
| Air Temperature | 9.9% |
| Process Temperature | 6.4% |
| Machine Type | <1% |

Torque, rotational speed, and tool wear together account for roughly 82% of the model's total feature importance. These values describe how the trained model uses its inputs — they don't establish that any feature *causes* failure.

## Threshold and maintenance cost analysis

The project also looks at how the probability threshold affects the trade-off between missed failures and false alarms. A lower threshold catches more potential failures but raises more false alarms.

A simple hypothetical cost model was used to explore this trade-off:

- Failure cost: €10,000
- Inspection cost: €100

These are portfolio assumptions for illustrating decision-making under different error costs, not real operational figures from the dataset.

## Explainability with SHAP

SHAP is used to explain both global model behavior and individual predictions.

- **Global explanation** — a representative sample of 1,000 machines is used to generate a global SHAP summary plot showing which features matter most across the dataset.
- **Individual explanation** — for any single machine, SHAP shows how each feature pushed the model's prediction toward or away from failure (e.g., high torque or high tool wear pushing the risk up).

As with feature importance, SHAP explanations describe model behavior, not physical causation.

## Dashboard

The Streamlit app lets users enter machine operating conditions and interactively explore predicted failure risk. It shows:

- Failure probability
- Low / medium / high risk classification
- Maintenance recommendation
- Local SHAP contribution chart
- Global feature importance
- Dataset-level failure statistics
- Average machine conditions by outcome

Run it with:

```bash
streamlit run src/app.py
```

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI ML Repository):

- 10,000 machine records
- 339 failures (3.39% failure rate)
- Multiple operating-condition and machine-type variables

The dataset is synthetic but designed to represent realistic predictive-maintenance scenarios.

## Running it locally

Clone the repository and create a virtual environment:

```bash
git clone <your-repository-url>
cd predictive-maintenance-ai

python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline, then launch the dashboard:

```bash
python src/prepare_data.py
python src/explore_data.py
python src/train_model.py
python src/explain_model.py

streamlit run src/app.py
```

## Project structure

```
predictive-maintenance-ai/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned data, plots, and SHAP output
├── models/
│   └── predictive_maintenance_model.joblib
├── notebook/
├── src/
│   ├── prepare_data.py         # Data cleaning and preprocessing
│   ├── explore_data.py         # Exploratory data analysis
│   ├── train_model.py          # Model training and evaluation
│   ├── explain_model.py        # Global and local SHAP analysis
│   └── app.py                  # Streamlit dashboard
├── .gitignore
├── Objective.txt
├── README.md
└── requirements.txt
```

## Limitations

This is a portfolio-scale prototype, not a production predictive-maintenance system:

- Trained on synthetic historical data — not validated on real industrial equipment
- No live sensor, PLC, SCADA, or IoT integration; dashboard inputs are manually simulated
- Maintenance cost values are hypothetical
- Feature importance and SHAP values describe learned model behavior, not causal evidence
- Production deployment would need monitoring, retraining, calibration, real-equipment validation, and integration with operational systems

## Author

Yash Kaushik
