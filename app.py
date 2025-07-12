import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

# 🔁 Cached loading of model and transformers
@st.cache_resource
def load_resources():
    with open("xgb_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    feature_order = model.get_booster().feature_names
    return model, scaler, encoder, feature_order

model, scaler, encoder, feature_order = load_resources()

# 🧹 Preprocessing function
def preprocess(df_raw):
    df = df_raw.copy()
    df["charges"] = np.log1p(df["charges"])
    num_cols = ["age", "bmi", "charges"]
    df[num_cols] = scaler.transform(df[num_cols])
    cat_cols = ["region", "children"]
    enc = encoder.transform(df[cat_cols])
    enc_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)
    # Reorder to match training
    df = df[feature_order]
    return df

st.title("📈 Insurance Claim Predictor")

mode = st.radio("Input mode:", ("Single Input", "Batch Upload"))
results_df = pd.DataFrame()

if mode == "Single Input":
    with st.form("input_form"):
        age = st.number_input("Age", 18, 100, 30)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0, 0.1)
        charges = st.number_input("Charges", 0.0, 1e6, 5000.0, 1.0)
        sex = st.radio("Sex", ("female", "male"))
        smoker = st.radio("Smoker", ("no", "yes"))
        region = st.selectbox("Region", ["northeast","northwest","southeast","southwest"])
        children = st.selectbox("Children", list(range(6)))
        submitted = st.form_submit_button("Predict")

    if submitted:
        df_raw = pd.DataFrame({
            "age": [age], "bmi": [bmi], "charges": [charges],
            "sex": [1 if sex=="male" else 0],
            "smoker": [1 if smoker=="yes" else 0],
            "region": [region], "children": [children]
        })
        X = preprocess(df_raw)
        proba = model.predict_proba(X)[0,1]
        pred = model.predict(X)[0]
        st.subheader("🔍 Prediction Results")
        st.write(f"**Claim Probability:** {proba:.2%}")
        st.write(f"**Predicted Class:** {'Claim' if pred==1 else 'No Claim'}")
        results_df = df_raw.assign(probability=proba, prediction=pred)

elif mode == "Batch Upload":
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        expected = {"age","bmi","charges","sex","smoker","region","children"}
        missing = expected - set(df_raw.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df_raw["sex"] = df_raw["sex"].map({"male":1,"female":0})
            df_raw["smoker"] = df_raw["smoker"].map({"yes":1,"no":0})
            X = preprocess(df_raw)
            probas = model.predict_proba(X)[:,1]
            preds = model.predict(X)
            results_df = df_raw.assign(probability=probas, prediction=preds)
            st.write("✅ Predictions Preview:")
            st.dataframe(results_df.head())

# 🛠️ Download & optional logging
if not results_df.empty:
    csv = results_df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name="predictions.csv")
    if st.checkbox("🔒 Log to 'pred_logs.csv'"):
        results_df.to_csv("pred_logs.csv", mode="a", header=False, index=False)
        st.success("Logged!")
