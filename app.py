import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import json
import requests


model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


lottie_heart = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_tutvdkg0.json")
lottie_success = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_jzv1iv.json")  # Success alt
lottie_error = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_ydo1amjm.json")   # Error alt


st.set_page_config(page_title="Heart Disease Predictor",  layout="centered")
st.title(" Heart Stroke Prediction By Prajwal")
st.markdown("###  Enter patient health details below")


with st.expander(" Patient Information"):
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])

with st.expander(" Medical Details"):
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholestrol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Fast", "Down"])


if lottie_heart:
    st_lottie(lottie_heart, height=200, key="heart")
else:
    st.info(" Heart animation not available.")


if st.button(" Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholestrol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }

    input_df = pd.DataFrame([raw_input])

    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]  # Reorder columns
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(scaled_input)[0][1]
        except:
            probability = None

    st.markdown("---")
    if prediction == 1:
        if lottie_error:
            st_lottie(lottie_error, height=200)
        st.error(f" High Risk of Heart Disease!" + (f" ({probability * 100:.2f}% confidence)" if probability is not None else ""))
    else:
        if lottie_success:
            st_lottie(lottie_success, height=200)
        st.success(f" Low Risk of Heart Disease." + (f" ({(1 - probability) * 100:.2f}% confidence)" if probability is not None else ""))


st.markdown("---")
st.markdown("###  Or Upload Health Records for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with patient records", type="csv")

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write(" Uploaded Data Preview:")
        st.dataframe(df_uploaded.head())

        df_processed = pd.get_dummies(df_uploaded)
        for col in expected_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[expected_columns]

        scaled_batch = scaler.transform(df_processed)
        batch_predictions = model.predict(scaled_batch)
        df_uploaded['Prediction'] = batch_predictions
        df_uploaded['Risk_Level'] = df_uploaded['Prediction'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')

        st.success("Predictions generated!")
        st.dataframe(df_uploaded)

    except Exception as e:
        st.error(f"Error processing file: {e}")


st.markdown("---")
st.markdown("####  Health Advice & Resources")
st.markdown("""
-  [CDC Heart Disease Resources](https://www.cdc.gov/heart-disease/prevention/?CDC_AAref_Val=https://www.cdc.gov/heartdisease/prevention.htm)
-  [Healthy Diet Tips](https://www.heart.org/en/healthy-living/healthy-eating/eat-smart)
- 🏃 [Exercise for a Healthy Heart](https://www.nhlbi.nih.gov/health/educational/lose_wt/phy_act.htm)
""")
