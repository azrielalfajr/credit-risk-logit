import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input fields for the user
st.title("Prediksi Risiko Kredit Anda")
st.image('header.png')

st.sidebar.header("Masukkan Data Anda")
person_age = st.sidebar.number_input("Usia", min_value=20, max_value=99, value=25)
person_income = st.sidebar.number_input("Pendapatan per Tahun ($)", min_value=4000, max_value=6000000, value=50000)
person_emp_length = st.sidebar.number_input("Lama Bekerja (tahun)", min_value=0, max_value=60, value=5)
loan_amnt = st.sidebar.number_input("Jumlah Pinjaman ($)", min_value=1000, max_value=500000, value=10000)
loan_int_rate = st.sidebar.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, value=10.0)
loan_percent_income = st.sidebar.number_input("Rasio Pinjaman terhadap Pendapatan (%)", min_value=0.0, max_value=100.0, value=15.0)

cb_person_default_on_file = st.sidebar.selectbox("Catatan Gagal Bayar", options=["Gagal bayar", "Tidak pernah"])
cb_person_default_on_file = 1 if cb_person_default_on_file == 1 else 0

cb_person_cred_hist_length = st.sidebar.number_input("Lama Riwayat Kredit (tahun)", min_value=0, max_value=50, value=10)

person_home_ownership = st.sidebar.selectbox("Kepemilikan Rumah", options=['Lainnya', 'Milik pribadi', 'Sewa','Hipotek'])
home_ownership_OTHER = 1 if person_home_ownership == 'OTHER' else 0
home_ownership_OWN = 1 if person_home_ownership == 'OWN' else 0
home_ownership_RENT = 1 if person_home_ownership == 'RENT' else 0
home_ownership_RENT = 1 if person_home_ownership == 'MORTGAGE' else 0


loan_intent = st.sidebar.selectbox("Alasan Pengajuan Pinjaman", options=['Pendidikan', 'Renovasi Rumah', 'Kesehatan', 'Alasan Pribadi', 'Modal Usaha'])
loan_intent_EDUCATION = 1 if loan_intent == 'EDUCATION' else 0
loan_intent_HOMEIMPROVEMENT = 1 if loan_intent == 'HOMEIMPROVEMENT' else 0
loan_intent_MEDICAL = 1 if loan_intent == 'MEDICAL' else 0
loan_intent_PERSONAL = 1 if loan_intent == 'PERSONAL' else 0
loan_intent_VENTURE = 1 if loan_intent == 'VENTURE' else 0

loan_grade = st.sidebar.selectbox("Peringkat Pinjaman", options=['A','B', 'C', 'D', 'E', 'F', 'G'])
loan_grade_B = 1 if loan_grade == 'A' else 0
loan_grade_B = 1 if loan_grade == 'B' else 0
loan_grade_C = 1 if loan_grade == 'C' else 0
loan_grade_D = 1 if loan_grade == 'D' else 0
loan_grade_E = 1 if loan_grade == 'E' else 0
loan_grade_F = 1 if loan_grade == 'F' else 0
loan_grade_G = 1 if loan_grade == 'G' else 0

# Prepare the input data for prediction
input_data = np.array([[
    person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income, 
    cb_person_default_on_file, cb_person_cred_hist_length, 
    home_ownership_OTHER, home_ownership_OWN, home_ownership_RENT, 
    loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL, loan_intent_PERSONAL, loan_intent_VENTURE, 
    loan_grade_B, loan_grade_C, loan_grade_D, loan_grade_E, loan_grade_F, loan_grade_G
]])
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = model.predict_proba(input_data_scaled)

# Display the prediction
st.subheader("Prediksi Default:")
with st.container(border=True):
    st.write("Ya ☹️" if prediction[0] == 1 else "Tidak 😀")

# Display the prediction probability
st.subheader("Tingkat Probabilitas:")
with st.container(border=True):
    st.write(f"Probabilitas Tidak Default: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Probabilitas Default: {prediction_proba[0][1]*100:.2f}%")
