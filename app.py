import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import plotly.graph_objects as go
import time

# Load the trained CNN model and scaler
model = load_model('optimized_nn_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Initialize session state for prediction control
if 'predicting' not in st.session_state:
    st.session_state['predicting'] = False

# Title and description
st.title("❤️ Heart Disease Prediction ❤️")
st.write("""
### 🩺 Assess Your Heart Health Risk
Welcome to the **Heart Disease Prediction App**, powered by an advanced neural network model trained on clinical data.  

🧑‍⚕️ **How it works:**  
Simply enter your health details below, and the app will calculate the likelihood of heart disease as a percentage.  

⚠️ **Note:** This is not a medical diagnosis. It’s an assessment tool to help you understand your risk and promote informed discussions with your healthcare provider.
""")


# Placeholder for the input form
form_placeholder = st.empty()

if not st.session_state['predicting']:
    with form_placeholder.form("user_inputs"):
        st.write("### Patient Information:")

        # Age
        st.write("#### Age")
        st.info("""
        The age of the patient in years.  
        - Younger patients are generally at lower risk.  
        - Risk increases significantly after the age of 50.
        """)
        age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25)

        # Sex
        st.write("#### Sex")
        st.info("""
        Patient's biological sex:  
        - **0:** Female  
        - **1:** Male  
        Sex plays a significant role in heart disease risk, with males generally at higher risk.
        """)
        sex = st.radio("Select your sex:", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

        # Chest Pain Type
        st.write("#### Chest Pain Type (0-3)")
        st.info("""
        Type of chest pain experienced:  
        - **0:** Typical angina (pain related to reduced blood flow to the heart)  
        - **1:** Atypical angina (chest pain not directly related to heart ischemia)  
        - **2:** Non-anginal pain (pain not related to the heart)  
        - **3:** Asymptomatic (no chest pain)  

        Chest pain is one of the most significant predictors of heart disease.
        """)
        cp = st.slider("Select your chest pain type:", min_value=0, max_value=3, value=0)

        # Resting Blood Pressure
        st.write("#### Resting Blood Pressure (mmHg)")
        st.info("""
        Resting blood pressure measured in mmHg (normal range: 90–120 mmHg).  
        - High blood pressure is a significant risk factor for heart disease.  
        - Consistently elevated levels (>140 mmHg) indicate hypertension.
        """)
        trestbps = st.number_input("Enter your resting blood pressure:", min_value=50, max_value=250, value=120)

        # Serum Cholesterol
        st.write("#### Serum Cholesterol (mg/dL)")
        st.info("""
        Serum cholesterol level in mg/dL (normal range: < 200 mg/dL).  
        - **200–239 mg/dL**: Borderline high  
        - **≥240 mg/dL**: High  
        Elevated cholesterol levels can lead to plaque buildup in arteries, increasing heart disease risk.
        """)
        chol = st.number_input("Enter your serum cholesterol:", min_value=100, max_value=600, value=200)

        # Fasting Blood Sugar
        st.write("#### Fasting Blood Sugar > 120 mg/dL")
        st.info("""
        Indicates whether the patient's fasting blood sugar is greater than 120 mg/dL:  
        - **0:** No  
        - **1:** Yes  

        High fasting blood sugar levels can indicate diabetes, which significantly increases heart disease risk.
        """)
        fbs = st.radio("Fasting blood sugar status:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        # Resting Electrocardiographic Results
        st.write("#### Resting Electrocardiographic Results (0-2)")
        st.info("""
        Results of the resting ECG:  
        - **0:** Normal  
        - **1:** ST-T wave abnormality (T wave inversions, ST elevation/depression)  
        - **2:** Left ventricular hypertrophy (enlargement of the left ventricle)  

        Abnormal ECG results may indicate underlying heart conditions.
        """)
        restecg = st.slider("Select your resting ECG result:", min_value=0, max_value=2, value=0)

        # Maximum Heart Rate Achieved
        st.write("#### Maximum Heart Rate Achieved")
        st.info("""
        Maximum heart rate achieved during exercise (normal range: 120–180 bpm).  
        - Lower maximum heart rates during stress tests may indicate reduced heart function.
        """)
        thalach = st.number_input("Enter your maximum heart rate achieved:", min_value=50, max_value=250, value=150)

        # Exercise-Induced Angina
        st.write("#### Exercise-Induced Angina")
        st.info("""
        Indicates whether the patient experiences angina (chest pain) during exercise:  
        - **0:** No  
        - **1:** Yes  

        Exercise-induced angina is a strong indicator of coronary artery disease.
        """)
        exang = st.radio("Exercise-induced angina status:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        # ST Depression Induced by Exercise (Oldpeak)
        st.write("#### ST Depression Induced by Exercise (Oldpeak)")
        st.info("""
        ST depression induced by exercise relative to rest, measured in mm.  
        Higher values indicate more severe heart abnormalities.
        """)
        oldpeak = st.number_input("Enter your ST depression value:", min_value=0.0, max_value=10.0, value=1.0)

        # Slope of the Peak Exercise ST Segment
        st.write("#### Slope of the Peak Exercise ST Segment (0-2)")
        st.info("""
        Describes the slope of the ST segment during peak exercise:  
        - **0:** Upsloping (least concerning)  
        - **1:** Flat  
        - **2:** Downsloping (most concerning)  

        Downsloping ST segments are more likely to indicate significant heart disease.
        """)
        slope = st.slider("Select the slope of the peak exercise ST segment:", min_value=0, max_value=2, value=1)

        # Number of Major Vessels Colored by Fluoroscopy
        st.write("#### Number of Major Vessels Colored by Fluoroscopy (0-3)")
        st.info("""
        Number of major blood vessels (0–3) colored by fluoroscopy.  
        Higher numbers indicate more blocked vessels, which correlates with higher risk.
        """)
        ca = st.slider("Select the number of major vessels:", min_value=0, max_value=3, value=0)

        # Thalassemia
        st.write("#### Thalassemia (0-3)")
        st.info("""
        A blood disorder that affects hemoglobin levels:  
        - **0:** Normal  
        - **1:** Fixed defect (permanent damage)  
        - **2:** Reversible defect (temporary damage)  
        - **3:** Unknown/Other
        """)
        thal = st.slider("Select your thalassemia type:", min_value=0, max_value=3, value=2)

        # Prediction button inside the form
        submitted = st.form_submit_button("Predict")

    if submitted:
        st.session_state['predicting'] = True  # Start prediction process

# Fullscreen GIF and Results
if st.session_state['predicting']:
    form_placeholder.empty()

    st.markdown(
        """
        <style>
        .full-screen-gif {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('https://media4.giphy.com/media/26xBBS5lCGRF5vyoM/giphy.gif') center center no-repeat;
            background-size: cover;
            z-index: 9999;
        }
        </style>
        <div class="full-screen-gif"></div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(3)

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    probability = prediction[0][0] * 100
    risk_level = "Low Risk" if probability < 30 else "Medium Risk" if probability <= 70 else "High Risk"

    st.markdown("<style>.full-screen-gif { display: none; }</style>", unsafe_allow_html=True)
    st.write(f"### Prediction Probability: **{probability:.2f}%**")
    st.write(f"### Risk Level: **{risk_level}**")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Heart Disease Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black" if risk_level == "High Risk" else "orange" if risk_level == "Medium Risk" else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': probability}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.session_state['predicting'] = False
