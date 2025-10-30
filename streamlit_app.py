# streamlit run streamlit_app.py

import streamlit as st
# import cv2
import numpy as np
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
import pandas as pd
import torch.nn as nn
from huggingface_hub import hf_hub_download

from models_J import ViT, ResNet50

# Berry Class Stages/Classes
PHENOLOGY_STAGES = [
    "breaking_leaf_buds", "increasing_leaf_size", "colored_leaves",
    "open_flowers", "ripe_fruits", "ripe_fruit_max",
    "pre_season", "post_season"
]

REPO_ID = "andrewkota/Berries_ViT32_TestStreamlit"
MODEL_FILENAME = "vit_bs32_ep40_katlian.pth"

@st.cache_resource
def load_model(repo_id, filename, model_class):
    """
    Loads in trained mode in eval mode

    model_path to .pth file
    """

    st.info(f"Downloading/loading model: {filename} from {repo_id}...")

    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.error("Please check your REPO_ID, MODEL_FILENAME, and that the file is public.")
        return None

    st.info(f"File downloaded. Loading model from {model_path}...")

    try:
        # Initialize your model class
        model = model_class(num_classes=len(PHENOLOGY_STAGES))
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set to evaluation mode
        model.eval()
        st.success(f"Model {filename} has been loaded.")
        return model
    
    except Exception as e:
        st.error(f"Error loading model from state_dict: {e}")
        st.error("Test.")
        return None
    


def preprocess_image(image_pil):
    """
    Converts an image to pytorch tensor
    """

    test_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_tensor = test_transform(image_pil).unsqueeze(0)
    return image_tensor


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


resnet_model = "Placeholder for ResNet50"
# vit_model = "Placeholder for ViT"
vit_model = load_model(REPO_ID, MODEL_FILENAME, ViT)
cnn_model = "Placeholder for CNN"
exp_model = "Placeholder for Expert"

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
    Page 2: Image analysis with trained models.
    """
    st.title("Berry Analysis - Analyze")
    st.write("Select a model to analyze your processed image.")

    image_to_analyze = st.session_state.processed_image

    if image_to_analyze:
        st.write("Processed image:")
        st.image(image_to_analyze, width="stretch")

        # --- ADDED MODEL SELECTION ---
        model_choice = st.selectbox(
            "Choose a model:",
            ("ViT (ViT_B_16)", "ResNet50 (Placeholder)")
        )

        if st.button(f"Run {model_choice} Model"):
            
            # Select the correct model
            model_to_run = None
            if model_choice.startswith("ViT"):
                model_to_run = vit_model
            else:
                model_to_run = resnet_model
            
            # Checks if model is loaded
            if model_to_run is None or isinstance(model_to_run, str):
                st.error(f"Model '{model_choice}' is not loaded or failed to load.")
                st.warning("Placeholder - model integration not fully functional.")
            else:
                with st.spinner("Model is analyzing..."):
                    
                    # 1. Preprocess the image
                    image_tensor = preprocess_image(image_to_analyze)
                    
                    # 2. Run inference
                    with torch.no_grad():
                        logits = model_to_run(image_tensor)
                        # Convert to probabilities
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                    
                    # 3. Get top prediction
                    top_prob = probabilities[0].max().item()
                    top_class_index = probabilities[0].argmax().item()
                    top_class_name = PHENOLOGY_STAGES[top_class_index]

                    st.success(f"Analysis Complete! Model: {model_choice}")
                    
                    # 4. Display Metric
                    st.metric(
                        label=f"Top Prediction",
                        value=f"{top_class_name}",
                        delta=f"Confidence: {top_prob:.2%}",
                        delta_color="normal"
                    )

                    # 5. Display Chart
                    st.subheader("All Class Probabilities")
                    probs_df = pd.DataFrame(probabilities[0].numpy(), index=PHENOLOGY_STAGES, columns=["Probability"])
                    st.bar_chart(probs_df)

    else:
        st.warning("No image was processed. Please go back to Step 1.")
    
    st.button("Back", on_click=prev_page)



if st.session_state.page == 1:
    upload_process_img()
elif st.session_state.page == 2:
    page_analysis()