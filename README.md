# Predictive Maintenance AI

A machine learning project that predicts equipment failure risk from operating conditions, with an interactive Streamlit dashboard for exploring "what-if" scenarios.

Built with Python, Pandas, Scikit-learn, and Streamlit.

## What it does

Feed in a machine's operating conditions (type, temperature, speed, torque, tool wear) and the model estimates the probability of failure, along with a risk level and a maintenance recommendation.

```
Raw data → cleaning → EDA → feature engineering → Random Forest → evaluation → Streamlit dashboard
```

## Results

| Metric | Score |
|---|---:|
| Accuracy | 98.2% |
| Precision | 75.0% |
| Recall | 70.6% |
| F1 Score | 72.7% |

On 2,000 held-out test machines (3.4% failure rate), the model caught 48 of 68 actual failures, missed 20, and raised 16 false alarms. Recall matters most here — a missed failure is usually far more costly than an unnecessary inspection.

## The model

A Random Forest classifier (`n_estimators=200`, `class_weight='balanced'`) trained on a stratified 80/20 split, since failures are rare in the data.

**Inputs:** machine type, air temperature, process temperature, rotational speed, torque, tool wear.

ID columns (UDI, Product ID) and failure-mode flags (TWF, HDF, PWF, OSF, RNF) were dropped — the former aren't useful signals, the latter risk leaking the answer into the input.

**What drives predictions:**

| Feature | Importance |
|---|---:|
| Torque | 32.2% |
| Rotational Speed | 28.6% |
| Tool Wear | 21.3% |
| Air Temperature | 9.9% |
| Process Temperature | 6.4% |
| Machine Type | <1% |

Torque, speed, and tool wear alone account for ~82% of the model's decisions. This reflects how the model weighs inputs, not proof of what actually causes failures.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI ML Repository) — 10,000 synthetic but realistic machine records, 339 failures (3.39% failure rate).

## Running it locally

```bash
git clone <your-repository-url>
cd predictive-maintenance-ai

python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt

python src/prepare_data.py
python src/explore_data.py
python src/train_model.py
streamlit run src/app.py
```

## Project structure

```
predictive-maintenance-ai/
├── data/
│   ├── raw/            # original dataset
│   └── processed/      # cleaned data + plots
├── models/              # trained model (.joblib)
├── notebook/
├── src/
│   ├── prepare_data.py
│   ├── explore_data.py
│   ├── train_model.py
│   └── app.py
└── requirements.txt
```

## Limitations

This is a portfolio-scale prototype, not a production monitoring system:

- Trained on synthetic, historical tabular data — not validated on real equipment
- No live sensor, PLC, SCADA, or IoT integration; dashboard inputs are simulated
- Feature importance shows correlation, not causation

## Author

Yash Kaushik
