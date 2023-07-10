import streamlit as st
import streamlit.components.v1 as components
import base64
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


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

# ---------------------- Create dataframe from input user

def create_data_input(filename_list, label_list, list_data, index_previous):
    """
    Input:
    filename_list: List of filename uploaded by user
    label_list: Predict respective to filename_list
    list_data: Dataframe before adding new prediction
    index_previous: Index of last row of list_data before adding new prediction
    """
    data = []
    for file, label in zip(filename_list, label_list):
        data.append([file, label])
    datapoint = pd.DataFrame(data = np.array(data), index = range(index_previous + 1, index_previous + len(data) + 1), columns=list_data.columns)
    return datapoint


def save_prediction(excel_url, list_data, data_point):
    """
    Input:
    - excel_url: Excel file dir
    - list_data: Original excel data
    - data_point: New prediction data
    """
    # Create input datapoint
    list_data = pd.concat((list_data, data_point),axis=0)
    list_data.to_excel(excel_url)
    print("Update new prediction!")
    return 
