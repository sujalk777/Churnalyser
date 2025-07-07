import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

st.title('Churn Modelling of Bank by Deep Learning')
col1, col2 = st.columns(2)
with col1:
    CreditScore=st.number_input('Credit Score',min_value=350, max_value=1000, value=350)
with col2:
    Geography=st.selectbox('Geography',('France','Germany','Spain'))
Gender=st.selectbox('Gender',('Male','Female'))
col3,col4=st.columns(2)
with col3:
    Age=st.number_input('Age',min_value=18, max_value=100, value=18)
with col4:
    Tenure=st.slider('Tenure',min_value=0, max_value=10, value=0,step=1)
Balance=st.number_input('Balance',min_value=0.0, max_value=1000000.0, value=0.0)
NumOfProducts=st.slider('Number of Products',min_value=1, max_value=4, value=1,step=1)
col5,col6=st.columns(2)
with col5:
    HasCrCard=st.radio('Has Credit Card',(0,1))
with col6:
    IsActiveMember=st.radio('Is Active Member',(0,1))
EstimatedSalary=st.number_input('Estimated Salary',min_value=0.0, max_value=1000000.0, value=0.0)
model=load_model('my_model.h5')
data={'CreditScore':CreditScore,'Geography':Geography,'Gender':Gender,'Age':Age,'Tenure':Tenure,'Balance':Balance,'NumOfProducts':NumOfProducts,'HasCrCard':HasCrCard,'IsActiveMember':IsActiveMember,'EstimatedSalary':EstimatedSalary}
df=pd.DataFrame(data,index=[0])
prepocess=pickle.load(open('preprocessor.pkl','rb'))
df=prepocess.transform(df)
prediction=model.predict(df)
if st.button('Predict'):
    if prediction>0.5:
        st.error('Customer will leave the bank')
    else:
        st.success('Customer will stay with the bank')
    propba=st.subheader(str(round(prediction[0][0]*100,2))+'%')
    st.write('Probability of customer leaving the bank:'+ 'propba')
