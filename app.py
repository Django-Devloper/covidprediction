import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

model = load_model('covid_prediction.keras')
binary_select = ('True' , 'False')
known_contact_option = ('Contact with confirmed','Abroad')
pickle_dict = {
    'Cough_symptoms':'Cough_symptomsencoder.pkl',
    'Fever':'Feverencoder.pkl',
    'Sore_throat':'Sore_throatencoder.pkl',
    'Shortness_of_breath':'Shortness_of_breathencoder.pkl',
    'Headache':'Headacheencoder.pkl',
    'Age_60_above':'Age_60_aboveencoder.pkl',
    'Sex':'Sexencoder.pkl',
    'Known_contact':'Known_contactencoder.pkl',
}

def pickel_reder(column_name,pickler,input_data):
    with open(pickler,'rb') as file :
        picker_lable =  pickle.load(file)
    input_data[column_name] = picker_lable.transform([input_data[column_name]])
    return input_data

st.title('covid Prediction ')
cough= st.checkbox('Do you have symptoms of Cough ?:',binary_select)
fever =st.checkbox('Do you have symptoms of Fever ?:',binary_select)
sore_throat =st.checkbox('Do you have Sore throat ?:',binary_select)
shortness_of_breath =st.checkbox('Do you have Shortness of Breath ?:',binary_select)
headache = st.checkbox('Do you have Headache ?:',binary_select)
age_60_above = st.radio('are you above 60 ?:',('Yes','No'))
sex =st.radio('Whats your Gender ?:',('male' ,'female'))
known_contact = st.radio('is you know in abroad or have to met ?:',known_contact_option)
analysis = st.button('Analysis Report' ,use_container_width=True, type='primary')
if analysis:
    input_data = {
        'Cough_symptoms':cough,
        'Fever':fever,
        'Sore_throat':sore_throat,
        'Shortness_of_breath':shortness_of_breath,
        'Headache':headache,
        'Age_60_above':age_60_above,
        'Sex':sex,
        'Known_contact':known_contact
    }
    for column_name , pickler in pickle_dict.items():
        input_data = pickel_reder(column_name,pickler,input_data)
    input_data_df = pd.DataFrame(input_data)
    with open('scallerencoder.pkl' ,'rb') as file:
        scaller = pickle.load(file)
    scalled_data =scaller.transform(input_data_df)
    predict = model.predict(scalled_data)
    if predict > .5:
        st.info(f'You are COVID Posotive. Prediction: {int(predict[0][0]*100)} %')
    else:
        st.info(f'You are not Covide Positive. Prediction: {int(predict[0][0]*100)} %')