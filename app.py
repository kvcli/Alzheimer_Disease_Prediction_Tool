import streamlit as st
import pandas as pd
import joblib



# -------------------------
# Load model
# -------------------------
try:
    model = joblib.load('filtered_classifier_model.joblib')
except FileNotFoundError:
    st.error("Error: Model file 'filtered_classifier_model.joblib' not found.")
    st.stop()

# -------------------------
# Human-friendly feature names
# -------------------------
display_names = {
    'MMSE': 'Mini-Mental State Examination (MMSE)',
    'ADL': 'Activities of Daily Living (ADL)',
    'FunctionalAssessment': 'Functional Assessment Score',
    'MemoryComplaints': 'Memory Complaints',
    'BehavioralProblems': 'Behavioral Problems',
    'CholesterolTotal': 'Total Cholesterol (mg/dL)',
    'PhysicalActivity': 'Physical Activity Level',
    'SystolicBP': 'Systolic Blood Pressure (mmHg)',
    'Smoking': 'Smoking Status',
    'AlcoholConsumption': 'Alcohol Consumption'
}

feature_names = list(display_names.keys())

# -------------------------
# Page title
# -------------------------
st.title("🧠 Alzheimer’s Disease Prediction Tool")
st.markdown("Provide the patient's information to estimate likelihood of Alzheimer's disease.")

# -------------------------
# Model accuracy / info block
# -------------------------
with st.expander("📊 Model Performance & Information"):
    st.markdown("""
    **Model accuracy:** 95.81%  
    **Model type:** XGBoost Classifier  
    **Training dataset size:** 2,150 samples  
    **Selected features:** 10 clinically relevant predictors  
                
    This model predicts the *likelihood of cognitive impairment consistent with Alzheimer's disease*.  
    It is **not a medical diagnosis** and should be used for educational purposes only.
    """)

# -------------------------
# Default values & ranges
# -------------------------
default_values = {
    'MMSE': 25.0,
    'ADL': 5.0,
    'FunctionalAssessment': 5.0,
    'MemoryComplaints': 0,
    'BehavioralProblems': 0,
    'CholesterolTotal': 200.0,
    'PhysicalActivity': 5.0,
    'SystolicBP': 120.0,
    'Smoking': 0,
    'AlcoholConsumption': 0
}

feature_ranges = {
    'MMSE': (0.0, 30.0),
    'ADL': (0.0, 10.0),
    'FunctionalAssessment': (0.0, 10.0),
    'CholesterolTotal': (100.0, 300.0),
    'PhysicalActivity': (0.0, 10.0),
    'SystolicBP': (90.0, 200.0)
}

binary_features = ['MemoryComplaints', 'BehavioralProblems', 'Smoking', 'AlcoholConsumption']

input_data = {}

# -------------------------
# Input fields (clean labels)
# -------------------------
st.subheader("📥 Patient Inputs")

for feature in feature_names:
    label = display_names[feature]

    if feature in binary_features:
        input_data[feature] = st.selectbox(
            label,
            options=[0, 1],
            index=default_values[feature],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    else:
        min_val, max_val = feature_ranges.get(feature, (0.0, 1000.0))
        input_data[feature] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_values.get(feature, (min_val + max_val) / 2),
            step=0.1
        )

# -------------------------
# Predict button
# -------------------------
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(input_df)[0]

    # Styled output box
    if prediction == 1:
        st.markdown(
            """
            <div style="padding:15px; background-color:#ffe6e6; border-left:5px solid #cc0000; border-radius:5px;">
                <h3 style="color:#cc0000;">⚠️ High Likelihood of Alzheimer’s</h3>
                <p style="color:#cc0000;">This result suggests cognitive impairment consistent with Alzheimer's disease.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="padding:15px; background-color:#e6ffe6; border-left:5px solid #009933; border-radius:5px;">
                <h3 style="color:#009933;">✔️ Low Likelihood of Alzheimer’s</h3>
                <p style="color:#009933;">The model predicts the patient is unlikely to have Alzheimer's disease.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <style>
        .footer-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 12px;
            backdrop-filter: blur(8px);
            background: rgba(255, 255, 255, 0.55);
            border-top: 1px solid rgba(200, 200, 200, 0.35);
            text-align: center;
            font-size: 15px;
            z-index: 9999;
        }

        /* Dark mode footer */
        body[data-theme="dark"] .footer-container {
            background: rgba(20, 20, 20, 0.65);
            border-top: 1px solid rgba(80, 80, 80, 0.6);
            color: white !important;
        }

        .footer-container a {
            margin: 0 10px;
            color: #0077cc;
            text-decoration: none;
            font-weight: 600;
        }
        .footer-container a:hover {
            text-decoration: underline;
        }

        .footer-icons img {
            width: 22px;
            margin-left: 6px;
            vertical-align: middle;
        }
    </style>

    <div class="footer-container">
        Made with ❤️ by <strong>Abdulaziz Aljaadi</strong>
        <div class="footer-icons">
            <a href="https://github.com/kvcli" target="_blank">
                GitHub <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg">
            </a>
            |
            <a href="https://www.linkedin.com/in/abdulaziz-aljaadi" target="_blank">
                LinkedIn <img src="https://th.bing.com/th/id/R.1307a2648e71d531704a0f5a270ea966?rik=UK6a6u%2fILSTfCg&pid=ImgRaw&r=0">
            </a>
            |
            <a href="https://devthoughtsbyaziz.vercel.app" target="_blank">
                Portfolio <img src="https://icon-library.com/images/website-icon-vector/website-icon-vector-8.jpg">
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)