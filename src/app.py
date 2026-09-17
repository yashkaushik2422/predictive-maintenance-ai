import joblib
import pandas as pd
import streamlit as st
import shap

MODEL_PATH = "models/predictive_maintenance_model.joblib"
DATA_PATH = "data/processed/machine_failure_data.csv"


# ---------------------------------------------------------
# Load model and reference data
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
reference_data = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("⚙️ Predictive Maintenance AI")

st.write(
    "Enter the current machine operating conditions to estimate "
    "the probability of machine failure."
)


# ---------------------------------------------------------
# Sidebar - Machine Conditions
# ---------------------------------------------------------

st.sidebar.header("Machine Conditions")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.slider(
    "Air Temperature [K]",
    min_value=295.0,
    max_value=305.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.slider(
    "Process Temperature [K]",
    min_value=305.0,
    max_value=314.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.slider(
    "Rotational Speed [rpm]",
    min_value=1100,
    max_value=2900,
    value=1500,
    step=10
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    min_value=3.0,
    max_value=77.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.slider(
    "Tool Wear [min]",
    min_value=0,
    max_value=253,
    value=100,
    step=1
)


# ---------------------------------------------------------
# Create model input
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Air temperature [K]": [air_temperature],
    "Process temperature [K]": [process_temperature],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "Type_L": [1 if machine_type == "L" else 0],
    "Type_M": [1 if machine_type == "M" else 0]
})


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

failure_probability = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]


# ---------------------------------------------------------
# Risk level
# ---------------------------------------------------------

if failure_probability >= 0.70:
    risk_level = "HIGH"
elif failure_probability >= 0.30:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"


# ---------------------------------------------------------
# Prediction summary
# ---------------------------------------------------------

st.subheader("Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Failure Probability",
        f"{failure_probability * 100:.1f}%"
    )

with col2:
    st.metric(
        "Risk Level",
        risk_level
    )

with col3:
    if prediction == 1:
        st.error("🔴 Maintenance Warning")
    else:
        st.success("🟢 Machine Healthy")


# ---------------------------------------------------------
# Risk visualization
# ---------------------------------------------------------

st.progress(
    failure_probability,
    text=f"Failure Risk: {failure_probability * 100:.1f}%"
)

st.caption(
    "Risk thresholds: LOW < 30%  |  MEDIUM 30–69%  |  HIGH ≥ 70%"
)

# ---------------------------------------------------------
# Current machine conditions
# ---------------------------------------------------------

st.subheader("Current Machine Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Machine Type",
        machine_type
    )

with col2:
    st.metric(
        "Air Temperature",
        f"{air_temperature:.1f} K"
    )

with col3:
    st.metric(
        "Process Temperature",
        f"{process_temperature:.1f} K"
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Rotational Speed",
        f"{rotational_speed} rpm"
    )

with col5:
    st.metric(
        "Torque",
        f"{torque:.1f} Nm"
    )

with col6:
    st.metric(
        "Tool Wear",
        f"{tool_wear} min"
    )

# ---------------------------------------------------------
# Model-driven local explanation
# ---------------------------------------------------------

st.subheader("Why This Prediction?")

st.caption(
    "These explanations show how each machine condition "
    "contributed to the Random Forest prediction. "
    "Positive SHAP values push the prediction toward failure, "
    "while negative values push it toward healthy operation."
)


# Build the same feature structure used during model training
machine_data = pd.DataFrame(
    {
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type": [machine_type],
    }
)

machine_data = pd.get_dummies(
    machine_data,
    columns=["Type"],
    drop_first=True
)


# Make sure the dashboard input has exactly the same
# feature columns as the trained model
model_features = model.feature_names_in_

machine_data = machine_data.reindex(
    columns=model_features,
    fill_value=0
)


# Calculate SHAP values for this machine
machine_shap = explainer.shap_values(machine_data)

if hasattr(machine_shap, "values"):
    machine_shap_values = machine_shap.values
else:
    machine_shap_values = machine_shap


# Select the failure-class SHAP values
machine_shap_values = machine_shap_values[:, :, 1]


# Create explanation table
explanation = pd.DataFrame(
    {
        "Feature": machine_data.columns,
        "SHAP value": machine_shap_values[0],
    }
)

explanation["Absolute SHAP"] = explanation["SHAP value"].abs()

explanation = explanation.sort_values(
    "Absolute SHAP",
    ascending=False
)


# Display the strongest contributors
st.markdown("**Top contributing factors**")

for _, row in explanation.head(4).iterrows():

    feature = row["Feature"]
    shap_value = row["SHAP value"]

    if shap_value > 0:
        icon = "🔴"
        direction = "increased"
    else:
        icon = "🟢"
        direction = "reduced"

    if feature == "Torque [Nm]":
        value = f"{torque:.1f} Nm"
    elif feature == "Rotational speed [rpm]":
        value = f"{rotational_speed} rpm"
    elif feature == "Tool wear [min]":
        value = f"{tool_wear} min"
    elif feature == "Air temperature [K]":
        value = f"{air_temperature:.1f} K"
    elif feature == "Process temperature [K]":
        value = f"{process_temperature:.1f} K"
    elif feature == "Type_L":
        value = "Type L"
    elif feature == "Type_M":
        value = "Type M"
    else:
        value = "current machine condition"

    st.markdown(
        f"{icon} **{feature}** ({value}) — "
        f"{direction} the model's failure prediction."
    )

# ---------------------------------------------------------
# Local SHAP contribution chart
# ---------------------------------------------------------

import matplotlib.pyplot as plt

st.markdown("**Model contribution by feature**")

chart_data = explanation[
    ["Feature", "SHAP value"]
].copy()

chart_data = chart_data.sort_values("SHAP value")

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.barh(
    chart_data["Feature"],
    chart_data["SHAP value"]
)

ax.axvline(
    0,
    linewidth=1
)

ax.set_xlabel("SHAP contribution")
ax.set_ylabel("Feature")

ax.set_title(
    "How each feature influenced this prediction"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)	

# ---------------------------------------------------------
# Maintenance recommendation
# ---------------------------------------------------------

st.subheader("Maintenance Recommendation")


# Get the strongest positive SHAP contributors
positive_contributors = explanation[
    explanation["SHAP value"] > 0
].head(3)


# Convert feature names into maintenance-friendly labels
maintenance_labels = {
    "Torque [Nm]": "torque",
    "Rotational speed [rpm]": "rotational speed",
    "Tool wear [min]": "tool wear",
    "Air temperature [K]": "air temperature",
    "Process temperature [K]": "process temperature",
    "Type_L": "machine type",
    "Type_M": "machine type"
}


contributors = [
    maintenance_labels.get(feature, feature)
    for feature in positive_contributors["Feature"]
]


if risk_level == "HIGH":

    if contributors:
        focus_areas = ", ".join(contributors)

        st.error(
            f"Immediate inspection recommended. "
            f"The strongest model contributors for this machine "
            f"are {focus_areas}."
        )
    else:
        st.error(
            "Immediate inspection recommended based on the "
            "current predicted failure risk."
        )


elif risk_level == "MEDIUM":

    if contributors:
        focus_areas = ", ".join(contributors)

        st.warning(
            f"Continue monitoring the machine closely. "
            f"The strongest positive model contributors are "
            f"{focus_areas}."
        )
    else:
        st.warning(
            "Continue monitoring the machine closely as the "
            "predicted failure risk is elevated."
        )


else:

    st.success(
        "The machine currently shows low predicted failure risk. "
        "Continue normal monitoring and scheduled maintenance."
    )
# ---------------------------------------------------------
# Global model feature importance
# ---------------------------------------------------------

st.subheader("Model Feature Importance")

st.caption(
    "These values describe which features were most important "
    "to the Random Forest model during training. They do not "
    "change with the current machine settings."
)

importance = pd.Series(
    model.feature_importances_,
    index=input_data.columns
).sort_values(ascending=True)

st.bar_chart(importance)

# ---------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------

st.subheader("Training Dataset Overview")

total_machines = len(reference_data)
healthy_machines = (reference_data["Machine failure"] == 0).sum()
failed_machines = (reference_data["Machine failure"] == 1).sum()
failure_rate = failed_machines / total_machines


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Observations",
        f"{total_machines:,}"
    )

with col2:
    st.metric(
        "Healthy Observations",
        f"{healthy_machines:,}"
    )

with col3:
    st.metric(
        "Failure Observations",
        f"{failed_machines:,}"
    )


st.metric(
    "Historical Failure Rate",
    f"{failure_rate * 100:.2f}%"
)


# ---------------------------------------------------------
# Historical operating conditions
# ---------------------------------------------------------

st.subheader("Average Conditions by Machine Outcome")

comparison = (
    reference_data
    .groupby("Machine failure")[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ]
    .mean()
    .round(2)
)

comparison.index = ["Healthy", "Failed"]

st.dataframe(
    comparison,
    width="stretch"
)
