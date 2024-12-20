import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Pisahkan fitur (X) dan label (Y)
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standarisasi data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Pisahkan data latih dan data uji
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Latih model SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluasi akurasi model
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Simpan model dan scaler
pickle.dump(classifier, open('diabetes_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

# Streamlit Web App
st.title('Prediksi Diabetes')

col1,col2 = st.columns(2)

# Input dari pengguna
with col1:
    Pregnancies = st.text_input('Input nilai Pregnancies')
with col2:
    Glucose = st.text_input('Input nilai Glucose')
with col1:
    BloodPressure = st.text_input('Input nilai Blood Pressure')
with col2:
    SkinThickness = st.text_input('Input nilai Skin Thickness')
with col1:
    Insulin = st.text_input('Input nilai Insulin')
with col2:
    BMI = st.text_input('Input nilai BMI')
with col1:
    DiabetesPedigreeFunction = st.text_input('Input nilai Diabetes Pedigree Function')
with col2:
    Age = st.text_input('Input nilai Age')
# Prediksi
diab_diagnosis = ''

if st.button('Test Prediksi Diabetes'):
    try:
        # Konversi input ke float
        input_data = [
            float(Pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
        ]

        # Standardisasi data
        scaler = pickle.load(open('scaler.sav', 'rb'))
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_as_numpy_array)

        # Prediksi menggunakan model
        diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
        diab_prediction = diabetes_model.predict(input_data_scaled)

        # Interpretasi hasil
        if diab_prediction[0] == 1:
            diab_diagnosis = 'Pasien Terkena Diabetes'
        else:
            diab_diagnosis = 'Pasien Tidak Terkena Diabetes'

        st.success(diab_diagnosis)
    except ValueError:
        st.error('Harap masukkan nilai input yang valid (angka).')


