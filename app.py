import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Card Fraud Detector")
st.write(
    "Enter transaction details below. The `V1`-`V28` features come from the "
    "original (PCA-anonymized) dataset — in a real production system these "
    "would be computed automatically from the raw transaction, not typed in "
    "by hand. This demo lets you paste in a row from the test set to see the "
    "model in action."
)

with st.expander("Paste a row of V1–V28 values, Amount, and Hour (comma-separated)"):
    raw_input = st.text_area(
        "Example: 0.1, -1.2, 0.5, ... , 2.3, 149.62, 14",
        height=100,
    )

uploaded = st.file_uploader("...or upload a CSV of transactions to score in bulk", type=['csv'])

if st.button("Check for Fraud", type="primary"):
    if raw_input.strip():
        try:
            values = [float(x.strip()) for x in raw_input.split(',')]
            cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
            row = pd.DataFrame([values], columns=cols)
            row['Amount_scaled'] = scaler.transform(row[['Amount']])
            row = row.drop(columns=['Amount'])

            prob = model.predict_proba(row)[0][1]
            pred = model.predict(row)[0]

            if pred == 1:
                st.error(f"⚠️ Likely FRAUD — model confidence: {prob:.1%}")
            else:
                st.success(f"✅ Likely legitimate — fraud probability: {prob:.1%}")

            st.progress(min(prob, 1.0))
        except Exception as e:
            st.warning(f"Couldn't parse that input — check the format. ({e})")
    elif uploaded:
        df_in = pd.read_csv(uploaded)

        df_features = df_in.copy()
        if 'Time' in df_features.columns:
            df_features['Hour'] = (df_features['Time'] // 3600) % 24
            df_features = df_features.drop(columns=['Time'])
        if 'Class' in df_features.columns:
            df_features = df_features.drop(columns=['Class'])

        df_features['Amount_scaled'] = scaler.transform(df_features[['Amount']])
        df_features = df_features.drop(columns=['Amount'])

        expected_cols = [f'V{i}' for i in range(1, 29)] + ['Hour', 'Amount_scaled']
        df_features = df_features[expected_cols]

        probs = model.predict_proba(df_features)[:, 1]
        df_in['fraud_probability'] = probs
        df_in['prediction'] = (probs >= 0.5).astype(int)
        st.dataframe(df_in.sort_values('fraud_probability', ascending=False))
    else:
        st.warning("Paste some values or upload a CSV first.")

st.divider()
st.caption(
    "Trained on the Kaggle Credit Card Fraud Detection dataset. "
    "For demonstration purposes only — not a production fraud system."
)