# app.py — Noon Fraud-Check Tool  (Phase 4)
# Run locally:  streamlit run app.py
# Needs: fraud_model.joblib  — save YOUR final classifier like this in your notebook:
#   joblib.dump({'model': clf, 'columns': list(X_train.columns)}, 'fraud_model.joblib')
import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="Noon Fraud-Check", page_icon="🕵️")

@st.cache_resource
def load():
    return joblib.load("fraud_model.joblib")

M = load()
model, COLUMNS = M["model"], M["columns"]

def check(tx: dict):
    X = pd.get_dummies(pd.DataFrame([tx]), columns=["payment_method"]).reindex(columns=COLUMNS, fill_value=0)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    return pred, prob

st.title("🕵️ Noon — Transaction Fraud Check")
st.write("Enter a transaction's details. The model returns a **FRAUD / LEGIT** verdict.")

c1, c2 = st.columns(2)
with c1:
    order_amount     = st.number_input("Order amount (AED)", 10, 2000, 120)
    account_age_days = st.number_input("Account age (days)", 0, 2000, 200)
    hour_of_day      = st.slider("Hour of day", 0, 23, 13)
    num_orders_last_7d = st.slider("Orders in last 7 days", 0, 30, 3)
    delivery_distance_km = st.number_input("Delivery distance (km)", 0.0, 40.0, 5.0)
with c2:
    payment_method   = st.selectbox("Payment method", ["card","wallet","cod","applepay"])
    discount_applied = st.selectbox("Discount applied?", [0,1])
    prev_refunds     = st.slider("Previous refunds", 0, 10, 0)
    device_change    = st.selectbox("New/changed device?", [0,1])
    location_mismatch= st.selectbox("Billing/delivery mismatch?", [0,1])

tx = dict(order_amount=order_amount, account_age_days=account_age_days, hour_of_day=hour_of_day,
          num_orders_last_7d=num_orders_last_7d, payment_method=payment_method,
          delivery_distance_km=delivery_distance_km, discount_applied=discount_applied,
          prev_refunds=prev_refunds, device_change=device_change, location_mismatch=location_mismatch)

pred, prob = check(tx)
if pred == 1:
    st.error(f"🚨 FRAUD — flagged for review  (risk score {prob*100:.0f}%)")
else:
    st.success(f"✅ LEGIT  (risk score {prob*100:.0f}%)")
st.caption("Verdict comes from the classifier you trained in the lab notebook.")
