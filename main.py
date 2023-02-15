import streamlit as st
from joblib import dump, load

st.title("Test")

LABELS = ['setosa', 'versicolor', 'virginica'];
clf = load("DT.joblib")

sp_l = st.slider('Sepal Length: (cm)', min_value=0, max_value= 10)

sp_w = st.slider('Sepal Width: (cm)', min_value=0, max_value= 10)

pe_l = st.slider('Petal Length: (cm)', min_value=0, max_value= 10)

pe_w = st.slider('Petal Width: (cm)', min_value=0, max_value= 10)


prediction = clf.predict([sp_l,sp_w,pe_l,pe_w])

st.write(LABELS[prediction])


