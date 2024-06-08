import streamlit as st
from PIL import Image
from predict import predict


st.title("Tumor Detection usingg U-Net++")


uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image_path = "uploaded_image.png"
    image.save(image_path)
    
    st.markdown("<h1 style='text-align: center; color: white;'>Predicted Image</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>The predicted Image will be displayed below, this might take some seconds, kindly hold on.</p>", unsafe_allow_html=True)
    
    with st.spinner('Applying the juice...'):
        mask = predict(image_path)
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize to [0.0, 1.0]
    st.image(mask, caption='Predicted Image.', use_column_width=True)
