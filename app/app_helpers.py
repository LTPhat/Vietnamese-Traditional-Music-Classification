import streamlit as st
import streamlit.components.v1 as components
import base64
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



# Decode images
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set background's app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# ----------------------- Components's styles-------------------------------

title_style = """
    <style>
    .title {
        display: inline;
        color: red;
        text-align: center;
        font-size: 53px; 
        font-family: 'Brush Script MT', cursive;
    }
    </style>
    """

request_style = """
    <style>
    .request {
        display: inline;
        color: black;
        text-align: center;
        font-family: 'Brush Script MT', cursive;
    }
    </style>
"""
result_style = """
    <style>
    .result{
        display: inline;
        color: black;
        text-align: center;
        font-size: 40px;
        font-family: 'Brush Script MT', cursive;
    }
    </style>
"""