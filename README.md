\# Predictive Maintenance AI



An end-to-end machine learning pipeline for estimating industrial machine failure probabilities based on operating conditions.



This repository provides a complete prototype workflow: from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and deployment through an interactive Streamlit dashboard.



> \*\*Disclaimer:\*\* This is a software prototype using the synthetic UCI AI4I 2020 Predictive Maintenance Dataset. It is designed for demonstration purposes and is not connected to live SCADA, PLC, or IoT sensor systems.



\---



\## Dataset \& Features



The project uses the \*\*AI4I 2020 Predictive Maintenance Dataset\*\*, containing 10,000 observations.



The target variable is binary:



\* `0` = Healthy

\* `1` = Machine Failure



The dataset is highly imbalanced, with a baseline failure rate of \*\*3.39%\*\*.



\### Predictive Features



\*\*Categorical\*\*



\* Machine Type (`L`, `M`, `H`) — one-hot encoded



\*\*Numerical\*\*



\* Air temperature \[K]

\* Process temperature \[K]

\* Rotational speed \[rpm]

\* Torque \[Nm]

\* Tool wear \[min]



`UDI`, `Product ID`, and the specific failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were excluded to remove identifier features and reduce the risk of target leakage.



\---



\## Modeling \& Performance



A \*\*Random Forest Classifier\*\* was trained using:



\* `n\_estimators = 200`

\* `class\_weight = "balanced"`

\* 80/20 stratified train-test split



Random Forest was selected as a strong baseline for structured/tabular data and nonlinear relationships.



\### Evaluation Metrics



Because the dataset is highly imbalanced, the evaluation focuses on \*\*Recall, Precision, and F1 Score\*\* rather than accuracy alone.



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 0.9820 |

| Precision | 0.7500 |

| Recall    | 0.7059 |

| F1 Score  | 0.7273 |



The model detected \*\*48 of 68 actual failures\*\* in the held-out test set while producing \*\*16 false alarms\*\*.



\### Feature Importance



The most important features were:



| Feature                | Importance |

| ---------------------- | ---------: |

| Torque \[Nm]            |      32.2% |

| Rotational speed \[rpm] |      28.6% |

| Tool wear \[min]        |      21.3% |



Together, these three features account for approximately \*\*82% of the Random Forest's feature importance\*\*.



\---



\## Installation \& Usage



\### 1. Clone the repository



```bash

git clone <your-repository-url>

cd predictive-maintenance-ai

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the environment



\*\*Windows PowerShell:\*\*



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\*\*Linux/macOS:\*\*



```bash

source .venv/bin/activate

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Execute the pipeline



```bash

python src/prepare\_data.py

python src/explore\_data.py

python src/train\_model.py

```



The trained model is saved to:



```text

models/predictive\_maintenance\_model.joblib

```



\### 6. Launch the monitoring dashboard



```bash

streamlit run src/app.py

```



\---



\## Dashboard Features



The Streamlit application acts as a \*\*decision-support interface\*\* where users can enter current or simulated machine operating conditions.



It provides:



\### Failure Probability



Estimated probability of machine failure based on the selected operating conditions.



\### Risk Categorization



\* \*\*Low:\*\* < 30%

\* \*\*Medium:\*\* 30–69%

\* \*\*High:\*\* ≥ 70%



\### Operational Risk Indicators



The dashboard compares selected machine conditions with healthy-machine patterns in the dataset to highlight potentially elevated conditions such as:



\* High torque

\* Low rotational speed

\* High tool wear

\* Elevated temperature



These indicators are intended to support interpretation and are \*\*not causal explanations\*\*.



\### Maintenance Recommendation



The predicted risk level is translated into a simple operational recommendation, such as continued monitoring or inspection.



\---



\## Project Structure



```text

predictive-maintenance-ai/

├── data/

│   ├── raw/             # Original AI4I dataset

│   └── processed/       # Cleaned data and EDA visualizations

├── models/              # Serialized .joblib models

├── notebook/            # Jupyter notebooks for exploratory work

├── src/

│   ├── app.py

│   ├── prepare\_data.py

│   ├── explore\_data.py

│   └── train\_model.py

├── README.md

├── requirements.txt

└── .gitignore

```



\---



\## Limitations \& Roadmap



\### Current Limitations



\* Uses synthetic data; the model requires validation and retraining on domain-specific industrial data before production use.

\* Dashboard inputs are manually entered rather than received from live IoT/sensor systems.

\* No formal local explainability method such as SHAP is currently implemented.

\* Model probability thresholds have not been optimized for a specific maintenance cost structure.



\### Future Improvements



\* Implement probability calibration and threshold optimization.

\* Compare Random Forest against gradient-boosting models such as XGBoost or LightGBM.

\* Integrate SHAP for formal local feature explanations.

\* Integrate live sensor/IoT data.

\* Containerize the application with Docker.

\* Implement automated testing and CI/CD pipelines.



\---



\## Author



\*\*Yash Kaushik\*\*



