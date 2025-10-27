# streamlit run streamlit_app.py

import streamlit as st
# import cv2
import numpy as np
from PIL import Image
import requests


def crop_metadata_bar(img_array):
    """
    Crops the image to a fixed height of 3192px from the top.
    """
    
    img_height = img_array.shape[0]
    target_height = 3192
    
    if img_height > target_height:
       
        st.success(f"Cropped image from {img_height}px to {target_height}px height.")
        return img_array[:target_height, :, :]
    else:
        st.info("Image height is already at or below target crop height. No crop needed.")
        return img_array


def resize_image(img_pil, base_width):
    """
    Resizes image according to new base_width (base 64) while maintaining originalo aspect ratio.
    """
    original_width, original_height = img_pil.size

    wpercent = (base_width / float(original_width))
    hsize = int((float(original_height) * float(wpercent)))

    img_resized = img_pil.resize((base_width, hsize)), Image.Resampling.LANCZOS

    st.success(f"Image resized to {base_width} x {hsize} from {original_width} x {original_height}.")
    return img_resized





st.title("🎈 Berry Analysis")
st.write(
    # "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.write("This is currently a work in progress app, built to analyze and classify images uploaded by the user.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)

    image_unprocessed = image

    #Adds a checkbox to crop bar if present
    st.header("1. Crop Image")
    crop_image = st.checkbox("Crop black metadata bar")

    image_to_display = image 

    if crop_image:
        image_array = np.array(image)
        cropped_array = crop_metadata_bar(image_array)
        image_to_display = cropped_array


# else:
#     image = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

# edges = cv2.Canny(np.array(image), 100, 200)
# tab1, tab2 = st.tabs(["Detected edges", "Original"])
# tab1.image(edges, use_column_width=True)
# tab2.image(image, use_column_width=True)

# img_uploaded.image(image, use_column_width=True)

    st.image(image_to_display, caption="Uploaded Berry Image", width="stretch")

