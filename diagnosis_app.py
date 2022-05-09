import numpy as np
import streamlit as st
from keras.models import load_model, Model
import time

# Load model
model: Model = load_model('data/model.h5')


def diagnose_heart(input_data):
    # input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
    # input_data = (52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3)

    # Changing the input data to a numpy array
    numpy_data = np.asarray(input_data)

    # Reshaping the numpy array as we are predicting for only on instance
    input_reshaped = numpy_data.reshape(1, 13)
    print(input_reshaped.shape)

    prediction = (model.predict(input_reshaped) > 0.5).astype(int)
    print(f'Prediction -- {prediction[0]}')

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    if prediction[0] == 0:
        st.success('The person does not have heart disease.')
    else:
        st.error('The person has a heart disease.')


# Add title
st.title('Heart Diagnosis App')
# st.header('Heart Diagnosis App')

# Getting the input data from the user
age = st.number_input('Age', 1, 120, format='%i')
sex = st.number_input('Sex: 1 – male, 0 – female', 0, 1, format='%i')
cp = st.number_input('Chest pain type (0, 1, 2, 3 values)', 0, 3, format='%i')
trestbps = st.number_input('Resting blood pressure (mmHg)', 0, format='%i')
chol = st.number_input('Serum cholesterol (mg / dl)', 0, format='%i')
fbs = st.number_input('Fasting blood sugar > 120mg / dl: 1 – true, 0 – false', 0, 1, format='%i')
restecg = st.number_input('Resting electro-cardiographic results (values 0,1,2)', 0, 2, format='%i')
thalach = st.number_input('Maximum heart rate achieved', 0, format='%i')
exang = st.number_input('Exercise induced angina: 1 – yes, 0 – no', 0, 1, format='%i')
oldpeak = st.number_input('ST depression', 0.0, format='%f')
slope = st.number_input('Slope', 0, format='%i')
ca = st.number_input('Number of major vessels (0 - 3)', 0, 3, format='%i')
thal = st.number_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)', 0, 2, format='%i')

if sex == 'Male':
    sex = 1  # for male
else:
    sex = 0  # for female

if fbs == 'True':
    fbs = 1
else:
    fbs = 0

if exang == 'Yes':
    exang = 1
else:
    exang = 0

if thal == '0':
    thal = 0
elif thal == '1':
    thal = 1
else:
    thal = 2

# creating a button for Prediction
if st.button('Heart Disease Test Result'):
    diagnose_heart((age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal))


# age = st.number_input('Age', 1, 120)
# sex = st.selectbox('Sex: 1 – male, 0 – female', ['Select', 'Male', 'Female'])
# cp = st.number_input('Chest pain type (4 values)', 1, 4)
# trestbps = st.text_input('Resting blood pressure (mmHg)')
# chol = st.text_input('Serum cholesterol (mg / dl)')
# fbs = st.selectbox('Fasting blood sugar > 120mg / dl: 1 – true, 0 – false', ['Select', 'True', 'False'])
# restecg = st.number_input('Resting electrocardiographic results (values 0,1,2)', 0, 2)
# thalach = st.text_input('Maximum heart rate achieved')
# exang = st.selectbox('Exercise induced angina: 1 – yes, 0 – no', ['Select', 'Yes', 'No'])
# oldpeak = st.number_input('ST depression', format='%f')
# slope = st.text_input('Slope')
# ca = st.number_input('Number of major vessels (0 - 3)', 0, 3)
# thal = st.selectbox('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)',
#                     [0, 1, 2])