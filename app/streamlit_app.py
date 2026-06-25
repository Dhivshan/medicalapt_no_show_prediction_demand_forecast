# Imports
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Load Models ---
no_show_model = joblib.load("C://Users//cxsha//PycharmProjects//medical_appt_noshowpredict_demandforecasting//models//no_show_model.pkl")
forecast_model = joblib.load("C://Users//cxsha//PycharmProjects//medical_appt_noshowpredict_demandforecasting//models//demand_forecast_model.pkl")

# --- App Layout ---
st.set_page_config(page_title="Medical Appointments Dashboard", layout="wide")
st.title("📊 Medical Appointments Prediction & Forecasting")

# Tabs for modules
tab1, tab2 = st.tabs(["No-Show Predictor", "Demand Forecaster"])

# --- No-Show Predictor ---
with tab1:
    st.header("No-Show Risk Prediction")

    # Input form
    with st.form("no_show_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        disability = st.selectbox("Disability", ["Yes", "No", "Unknown"])
        needs_companion = st.selectbox("patient_needs_companion", ["Yes", "No"])
        specialty = st.selectbox("Specialty",
                                 ["Physiotherapy", "Psychology", "Speech Therapy", "Occupational Therapy", "Unknown"])
        appointment_shift = st.selectbox("Appointment Shift", ["Morning", "Afternoon", "Evening"])
        place = st.text_input("City", "Hyderabad")
        sms_received = st.selectbox("SMS Received", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        with st.spinner("Predicting no-show risk..."):
            # Example feature vector (simplified)
            input_data = pd.DataFrame([{
                "gender": gender,
                "age": age,
                "disability": disability,
                "patient_needs_companion": needs_companion,
                "specialty": specialty,
                "appointment_shift": appointment_shift,
                "place": place,
                "SMSreceived": 1 if sms_received == "Yes" else 0
            }])

            # Predict probability
            prob = no_show_model.predict_proba(input_data)[:, 1][0]
            risk_score = round(prob * 100, 2)

            st.success(f"Predicted No-Show Risk: {risk_score}%")

            if risk_score > 70:
                st.warning("⚠️ High Risk: Consider sending reminders or rescheduling.")
            elif risk_score > 40:
                st.info("Moderate Risk: Monitor closely.")
            else:
                st.success("Low Risk: Patient likely to attend.")


# --- Demand Forecaster (SARIMA only) ---
with tab2:
    st.header("Daily Appointment Demand Forecasting (SARIMA)")

    # Forecast horizon
    forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=60, value=30)

    if st.button("Generate Forecast"):
        with st.spinner("Generating SARIMA forecast..."):
            # Load and prepare demand data
            df = pd.read_csv("C://Users//cxsha//PycharmProjects//medical_appt_noshowpredict_demandforecasting//data//Medical_appointment_data.csv")
            df["appointment_date_continuous"] = pd.to_datetime(df["appointment_date_continuous"])
            daily_demand = df.groupby("appointment_date_continuous").size().reset_index(name="appointments")

            train_size = int(len(daily_demand) * 0.8)
            train = daily_demand.iloc[:train_size]
            test = daily_demand.iloc[train_size:]

            # SARIMA forecast: predict for the next N days beyond training
            sarima_forecast = forecast_model.predict(
                start=len(train),
                end=len(train) + forecast_days - 1,
                dynamic=False
            )

            # Build date range for plotting
            future_dates = pd.date_range(
                start=train["appointment_date_continuous"].iloc[-1] + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D"
            )

            # Plot SARIMA forecast
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(train["appointment_date_continuous"], train["appointments"], label="Train")
            ax.plot(test["appointment_date_continuous"], test["appointments"], label="Test", color="black")
            ax.plot(future_dates, sarima_forecast, label="SARIMA Forecast", color="blue")
            ax.set_title("SARIMA Forecasted Appointment Demand")
            ax.set_xlabel("Date")
            ax.set_ylabel("Appointments")
            ax.legend()
            st.pyplot(fig)

            st.success("SARIMA forecast generated successfully!")


