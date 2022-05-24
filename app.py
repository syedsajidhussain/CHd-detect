# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:12:56 2022

@author: syed



    
    """
import streamlit as st
from PIL import Image

from main import teachable_machine_classification
st.title("Image Classification Using VGG-classifier")
st.header("Congenital Heart Disease detection Example")
st.text("Upload a Heart Echo Image for image detection as CHD or NO-CHD")

uploaded_file = st.file_uploader("Choose a Heart Echo frame ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded frame.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'chd_model.h5')
        if label == 1:
            st.write("The echo frame is CHD")
        else:
            st.write("The echo frame is a No-CHD")
    
    
    
    
    
    
    
    
    
    
    