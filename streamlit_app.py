# streamlit run streamlit_app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests

st.title("🎈 Berry Analysis")
st.write(
    # "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.write("This is currently a work in progress app, built to analyze and classify images uploaded by the user.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
# else:
#     image = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

edges = cv2.Canny(np.array(image), 100, 200)
tab1, tab2 = st.tabs(["Detected edges", "Original"])
tab1.image(edges, use_column_width=True)
tab2.image(image, use_column_width=True)