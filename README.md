# 🏥 Medical Appointment-No show Prediction / Demand Forecast
Medical Appointment- No show prediction / Demand Forecast
This project builds machine learning models and a Streamlit dashboard to:
- Predict patient **no-show risk** for medical appointments.
- Forecast **daily appointment demand** across specialties and cities.
- Provide actionable insights for clinic staff to reduce inefficiencies and improve patient care.

---

## 📂 Project Structure

project-root/ │ ├── notebooks/ │ ├── eda.ipynb │ ├── preprocessing.ipynb │ ├── classification_training.ipynb │ ├── forecasting_training.ipynb │ └── evaluation.ipynb │ |├── models/ │ ├── preprocessor_cat.pkl │ └── preprocessor_nume.pkl │ ├── models/ │ ├── no_show_model.pkl │ └── demand_forecast_model.pkl │ ├── app/ │ ├── streamlit_app.py │ ├── data/ │ └── Medical_appointment.csv │ ├── requirements.txt ├── README.md

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medicalapt_no_show_prediction_demand_forecast.git
   cd medicalapt_no_show_prediction_demand_forecast
2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3.	Install dependencies
•	pip install -r requirements.txt
4.Run Streamlit app
streamlit run app/streamlit_app.py

## 📊 Data Dictionary (Key Columns)
-	Target
• no_show: Appointment outcome (yes = missed, no = attended)
-	Patient
•	gender: Male/Female
•	age: Patient age
•	under_12, over_60: Age flags
•	disability: Disability status
•	needs_companion: Whether patient requires a companion
-	Appointment
•	specialty: Medical specialty
•	appointment_time: Time of appointment
•	appointment_shift: Morning/Afternoon/Evening
•	appointment_date_continuous: Continuous calendar date
-	Location
•	place: City of appointment
-	Health (binary)
•	Hypertension, Diabetes, Alcoholism, Handcap, Scholarship, SMSreceived
-	Weather
•	avg_temp, max_temp, rain, heat_intensity, rain_intensity
•	rainy_day_before, storm_day_before
## 📈 Models & Evaluation
-	Classification (No-Show Prediction)
•	Algorithms: Logistic Regression, Random Forest, XGBoost
•	Metrics: F1, ROC-AUC, Precision, Recall, Confusion Matrix
•	Target: F1 > 0.70, ROC-AUC > 0.75
-	Forecasting (Demand Prediction)
•	Algorithms: SARIMA, Prophet
•	Metrics: RMSE, MAE, MAPE, R²
•	Target: MAPE < 20%, R² > 0.65
## 🖥️ Streamlit Dashboard
-	No-Show Predictor
•	Input patient details → risk score + tiered interpretation
-	Demand Forecaster
•	Select horizon (days) → forecast daily appointment demmand
-	Visualizations
•	Feature importance, demand trends, seasonal

## 📌 Business Impact
•	Reduce no-shows via targeted reminders.
•	Improve staffing efficiency with demand forecasts.
•	Support financial sustainability and patient care quality.
