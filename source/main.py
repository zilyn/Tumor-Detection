import streamlit as st
import pandas as pd

# set title
st.title('Pneumonia classification')
# set header
st.header('Please upload a chest X-ray image')
# upload file
st.file_uploader('', type=['jpeg', 'jpg', 'png'])
# load classifier
model = load_model('./source/pneumonia_classifier.h5')
# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
