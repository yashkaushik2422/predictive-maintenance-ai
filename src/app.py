import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/predictive_maintenance_model.joblib"
DATA_PATH = "data/processed/machine_failure_data.csv"


# ---------------------------------------------------------
# Load model and reference data
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)
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
# Dynamic operational risk indicators
# ---------------------------------------------------------

st.subheader("Why This Prediction?")

st.caption(
    "These indicators compare the current machine conditions "
    "with patterns observed in the training data. They are "
    "operational risk indicators, not causal explanations."
)


# Calculate reference statistics from the processed dataset
healthy_data = reference_data[
    reference_data["Machine failure"] == 0
]

risk_factors = []


# ----- Torque -----

torque_mean = healthy_data["Torque [Nm]"].mean()
torque_std = healthy_data["Torque [Nm]"].std()

if torque > torque_mean + torque_std:
    risk_factors.append(
        (
            "🔴",
            "High torque",
            f"Torque is {torque:.1f} Nm, which is noticeably "
            f"above the typical healthy-machine level of "
            f"{torque_mean:.1f} Nm."
        )
    )
elif torque > torque_mean:
    risk_factors.append(
        (
            "🟠",
            "Elevated torque",
            f"Torque is {torque:.1f} Nm, above the typical "
            f"healthy-machine level of {torque_mean:.1f} Nm."
        )
    )
else:
    risk_factors.append(
        (
            "🟢",
            "Torque within typical range",
            f"Torque is {torque:.1f} Nm, close to or below "
            f"the healthy-machine average of {torque_mean:.1f} Nm."
        )
    )


# ----- Rotational speed -----

speed_mean = healthy_data["Rotational speed [rpm]"].mean()
speed_std = healthy_data["Rotational speed [rpm]"].std()

if rotational_speed < speed_mean - speed_std:
    risk_factors.append(
        (
            "🔴",
            "Low rotational speed",
            f"Speed is {rotational_speed} rpm, noticeably below "
            f"the typical healthy-machine level of "
            f"{speed_mean:.0f} rpm."
        )
    )
elif rotational_speed < speed_mean:
    risk_factors.append(
        (
            "🟠",
            "Below-average rotational speed",
            f"Speed is {rotational_speed} rpm, below the healthy-"
            f"machine average of {speed_mean:.0f} rpm."
        )
    )
else:
    risk_factors.append(
        (
            "🟢",
            "Rotational speed within typical range",
            f"Speed is {rotational_speed} rpm, around or above "
            f"the healthy-machine average."
        )
    )


# ----- Tool wear -----

wear_mean = healthy_data["Tool wear [min]"].mean()
wear_std = healthy_data["Tool wear [min]"].std()

if tool_wear > wear_mean + wear_std:
    risk_factors.append(
        (
            "🔴",
            "High tool wear",
            f"Tool wear is {tool_wear} min, substantially above "
            f"the healthy-machine average of {wear_mean:.0f} min."
        )
    )
elif tool_wear > wear_mean:
    risk_factors.append(
        (
            "🟠",
            "Elevated tool wear",
            f"Tool wear is {tool_wear} min, above the healthy-"
            f"machine average of {wear_mean:.0f} min."
        )
    )
else:
    risk_factors.append(
        (
            "🟢",
            "Tool wear within typical range",
            f"Tool wear is {tool_wear} min, around or below "
            f"the healthy-machine average."
        )
    )


# ----- Air temperature -----

air_mean = healthy_data["Air temperature [K]"].mean()
air_std = healthy_data["Air temperature [K]"].std()

if air_temperature > air_mean + air_std:
    risk_factors.append(
        (
            "🟠",
            "Elevated air temperature",
            f"Air temperature is {air_temperature:.1f} K, "
            f"above the typical healthy-machine level."
        )
    )
else:
    risk_factors.append(
        (
            "🟢",
            "Air temperature within typical range",
            f"Air temperature is {air_temperature:.1f} K."
        )
    )


# ---------------------------------------------------------
# Display risk indicators
# ---------------------------------------------------------

for icon, title, description in risk_factors:
    if icon == "🔴":
        st.error(f"**{title}** — {description}")
    elif icon == "🟠":
        st.warning(f"**{title}** — {description}")
    else:
        st.success(f"**{title}** — {description}")


# ---------------------------------------------------------
# Maintenance recommendation
# ---------------------------------------------------------

st.subheader("Maintenance Recommendation")

if risk_level == "HIGH":
    st.error(
        "Immediate inspection recommended. Focus on torque, "
        "rotational speed, and tool wear before continued operation."
    )
elif risk_level == "MEDIUM":
    st.warning(
        "Continue monitoring the machine closely. Inspect "
        "torque, rotational speed, and tool wear if the risk "
        "continues to increase."
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
