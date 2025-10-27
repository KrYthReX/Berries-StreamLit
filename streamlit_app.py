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

    img_resized = img_pil.resize((base_width, hsize), Image.Resampling.LANCZOS)

    st.success(f"Image resized to {base_width} x {hsize} from {original_width} x {original_height}.")
    return img_resized


def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1


if 'page' not in st.session_state:
    st.session_state.page = 1
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None    


st.title("🎈 Berry Analysis")
st.write(
    # "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


def upload_process_img():
    """
    First page: Holds image uploading, metabar crop, and resizing functionality. Documentation WIP.
    """

    st.write("This is currently a work in progress app, built to analyze and classify images uploaded by the user.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

        image_unprocessed = image
        image_to_display = image 

        st.image(image_to_display, caption="Uploaded Berry Image", width="stretch")

        #Adds a checkbox to crop bar if present
        st.header("1. Crop Out Metadata Bar in Image")
        crop_image = st.checkbox("Crop black metadata bar")



        if crop_image:
            image_array = np.array(image_unprocessed)
            cropped_array = crop_metadata_bar(image_array)
            # image_to_display = cropped_array
            image_unprocessed = Image.fromarray(cropped_array)

        st.header("2. Resize Image")

        # resize_options = ["original", 64, 128, 256, 512, 1024] #1024 is max option, as it was used for training.
        resize_options = ["original", 64, 128, 256, 512, 1024, 2048, 4096] #1024 is max option, as it was used for training.

        selected_width = st.selectbox("Resize image (keep orig. aspect ratio):",
                        options = resize_options)

        image_to_display = image_unprocessed

        if selected_width != "original":
            base_width = int(selected_width)

            image_to_display = resize_image(image_unprocessed, base_width)

        st.header("3. Processed Image")
        st.image(image_to_display, caption="Processed Image", width="stretch")

        st.session_state.processed_image = image_to_display
        st.button("Next", on_click=next_page)


    # else:
    #     image = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

    # edges = cv2.Canny(np.array(image), 100, 200)
    # tab1, tab2 = st.tabs(["Detected edges", "Original"])
    # tab1.image(edges, use_column_width=True)
    # tab2.image(image, use_column_width=True)

    # img_uploaded.image(image, use_column_width=True)


def page_analysis():
    """
    Page 2: Image analysis with trained models. WIP.
    """
    st.title("Berry Analysis - Analyze")
    st.write("WIP: Analyze the processed image")

    image_to_analyze = st.session_state.processed_image

    if image_to_analyze:
        st.write("Processed image")
        st.image(image_to_analyze, width="stretch")

        if st.button("Run Model"):

            st.success("Placeholder")


        #WIP: models are not staged within Streamlit.

    else:
        st.warning("Placeholder for no image uploaded")
    
    st.button("Back", on_click=prev_page)



if st.session_state.page == 1:
    upload_process_img()
elif st.session_state.page == 2:
    page_analysis()