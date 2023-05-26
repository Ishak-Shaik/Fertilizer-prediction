import streamlit as st
import pandas as pd
import numpy as np

st.title('Fertilizer Prediction')
st.markdown('An Machine Learning model to predict fertilizer \
With the reading of Potassium, Nitrogen, Phosphorus')


st.header('Soil contents')
col1, col2 ,col3= st.columns(3)
with col1:
    st.text('Nitrogen content :')
    N = st.slider('N', 0, 50, 1)

with col2:
    st.text('Potassium content :')
    K = st.slider('K', 0, 30, 1)

with col3:
    st.text('Phosphorus content :')
    P = st.slider('P', 0, 50, 1)


from prediction import predict


Fertilizer_Names=['10 - 26 - 26','14 - 35 - 14','70 - 70 - 70','20 - 20','28 - 28','DAP','UREA']
result = predict(np.array([[N, K, P]]))

st.text('Fertilizer type you should use is : '+Fertilizer_Names[int(result)])